# DSV4.1 DSpark 在线训练最小适配方案

本文以 HuggingFace 的 `DeepSeek-V4-Flash-DSpark/inference` 和
`DeepSeek-V4.1-Flash/inference` 为模型结构依据，以 online 仓现有 DSV4
训练实现为代码基础。vLLM-Ascend 只负责 producer 和 hidden-state 传输，
不把 vLLM 的推理 cache、TP/EP 或 NPU kernel 搬到训练模型中。

## 1. 结论

DSV4.1 DSpark 不需要完整重写。online 现有 DSV4 DSpark 的以下部分可以继续使用：

- DSpark block/anchor 采样和 attention mask；
- `main_proj`、noise token、Markov head、confidence head；
- CE/L1/tau loss 和 batch collator；
- online Mooncake 数据接收和 target cache 格式；
- trainer、FSDP、checkpoint 和 evaluator 框架。

需要修改的模型语义主要是 delayed-mHC：

1. `hc_pre` 接收上一阶段的 `pre_mix`；
2. Attention 产生的 `attn_pre` 传给 FFN；
3. FFN 产生的 `ffn_pre` 传给下一层；
4. 最后一层使用 `pre_mix + hc_collapse` 得到单路 hidden state。

此前方案中列出的 paged KV cache、packed cache、context parallel、TP/EP、Mooncake connector 等属于 vLLM 推理侧，不属于 online DSpark 训练模型，不应作为 online 的改动项。

## 2. DSV4.1 与 DSV4 DSpark 的训练结构差异

### 2.1 DSV4 的层内流程

```text
hidden_states -> hc_pre -> attention -> hc_post -> hc_pre -> MoE -> hc_post
```

当前 DSV4 layer 对外返回的 `residual` 参数实际会在函数开始处被
`residual = hidden_states.clone()` 覆盖，跨层传递没有实际作用。有效的
residual 只是在当前 Attention/MoE 子层内部供 `hc_post` 使用。

### 2.2 DSV4.1 的层内流程

```text
hidden_states + pre_mix
  -> hc_pre(..., pre_mix) -> attention -> hc_post
  -> hc_pre(..., attn_pre) -> MoE -> hc_post
  -> hidden_states + ffn_pre
```

训练版接口应从：

```python
hidden_states, residual = layer(...)
```

改为：

```python
hidden_states, pre_mix = layer(hidden_states, pre_mix, ...)
```

初始值为：

```python
pre_mix = hidden_states.new_zeros(tokens, hc_mult, dtype=torch.float32)
pre_mix[:, 0] = 1.0
```

### 2.3 最终输出

DSV4 训练模型使用独立 `hc_head` 重新计算最终融合权重。DSV4.1 使用最后
一层产生的 `pre_mix`：

```python
final_hidden = (pre_mix.unsqueeze(-1) * hidden_states.float()).sum(-2)
final_hidden = final_norm(final_hidden)
```

## 3. hidden-state 数据协议

训练配置使用 zero-based layer id：

```python
target_layer_ids = [37, 38, 39]
```

producer 使用 one-based capture id：

| capture id | 内容 |
| ---: | --- |
| 38 | decoder layer 37 输出 |
| 39 | decoder layer 38 输出 |
| 40 | decoder layer 39 输出 |
| 41 | 最终 `hc_collapse + norm` 输出 |

前三层拼接为 `target_hidden_states [tokens, 3 * hidden_size]`，第四层单独
保存为 `target_last_hidden_states [tokens, hidden_size]`。capture id 41 不能
加入 `target_layer_ids`，它不是 `main_proj` 的输入层。

## 4. online 仓的最小修改

### 4.1 配置

新增 DSV4.1 配置，设置：

```python
target_layer_ids = [37, 38, 39]
online_vllm_layer_ids = [38, 39, 40, 41]
num_draft_layers = 3
block_size = 5
mask_token_id = 128799
online_vllm_model = "dsv41"
```

配置 builder 从 DSV4.1 config 保留 hidden size、HC、attention、MoE 和 RoPE
字段，并添加 DSpark 字段，不复制 vLLM 的推理 cache 字段。

### 4.2 训练模型

优先在现有 `deepspec/modeling/dspark/deepseek_v4/modeling.py` 上抽象
delayed-mHC，而不是复制整个模型目录：

- 为 decoder layer 增加可选 `pre_mix` 输入；
- 返回新的 `pre_mix`，不再把无效的跨层 residual 当作状态；
- 保留当前 PyTorch/SDPA、autograd、FSDP 和 NPU fallback；
- 增加 DSV4.1 的 `hc_collapse` 最终输出；
- 只有在 HuggingFace DSV4.1 inference 的 Q/K/V、RoPE、MoE 数学结构确实
  与 DSV4 不同时，才修改对应训练模块。

不能因为 vLLM 中存在 `DeepseekV41Attention` 就直接复制它；vLLM 类依赖
推理 cache 和并行运行时，训练版只需要 autograd-safe 的等价计算。

### 4.3 trainer/evaluator 和数据链路

复用现有 trainer 的 batch forward、loss、FSDP、embedding/lm_head 冻结、
checkpoint、proposal 和 confidence 统计。`online_feature_dataset.py` 和
`target_cache_dataset.py` 已经是模型无关的，只需校验四层 hidden state 并按
上述协议拆分；loss、collator、shard 格式无需修改。

## 5. vLLM-Ascend 侧

当前 DSV4.1 producer 侧为 vLLM-Ascend 的最小适配，代码改动集中在一个
commit：

```text
40b85a6bb1fdf3c7634678027040862441bfd22e
feat(dspark): export final DeepSeek V4.1 hidden state
```

改动文件：

```text
vllm_ascend/models/deepseek_v41/model.py
```

改动内容是在 DSV4.1 主模型 forward 的最终 `hc_collapse + norm` 之后，支持
使用 `num_hidden_layers + 1` 作为 final hidden state 的特殊 capture ID：

```python
final_hidden_capture_id = config.num_hidden_layers + 1
capture_final_hidden = (
    final_hidden_capture_id in self.aux_hidden_state_layers
)
...
hidden_states = self.norm(hidden_states)
if capture_final_hidden:
    aux_hidden_states.append(hidden_states)
```

DSV4.1 有 40 个 zero-based decoder layers，因此 producer 的 capture ID 为：

| capture ID | 内容 |
| ---: | --- |
| 38 | decoder layer 37 对应的辅助 hidden state |
| 39 | decoder layer 38 对应的辅助 hidden state |
| 40 | decoder layer 39 对应的辅助 hidden state |
| 41 | 最终 `hc_collapse + norm` hidden state |

其中 41 不是第 41 个 decoder layer，而是 `num_hidden_layers + 1` 的 final
hidden state sentinel。该输出与 online 训练端的
`target_last_hidden_states` 对应。

producer 侧其余能力直接复用已有实现：

- `extract_hidden_states` speculative method；
- `HiddenStateCacheSpec`；
- `DeepSpecMooncakeConnector`；
- Mooncake TCP 传输；
- DSV4.1 原有 TP/DP/EP、Engram 和 NPU inference kernel。

因此 online 训练端不需要复制 vLLM 的 paged KV cache、slot mapping、TP/EP
调度或推理 kernel。vLLM-Ascend 适配的唯一必要改动是让 final hidden state
进入与前三个辅助 hidden state 相同的传输列表；前三个 decoder hidden state
仍使用原有 DSV4.1 capture 流程。

### 5.1 HiddenStateCacheSpec、cache planner 与 Mooncake producer

离线 hidden-state 导出还依赖以下 commit：

```text
21ad9a356745da9b2d38bcf4622a7c442e57e55b
feat(dspark): support DeepSeek V4.1 offline hidden-state export
```

这个 commit 与 `40b85a6...` 分工不同：`40b85a6` 让模型 forward 产生最终
hidden state，`21ad9a356...` 负责把模型产生的四组状态规划到独立 cache，
从 cache 中取出并发布到 Mooncake。两者共同组成完整链路：

```text
DeepseekV41Model forward
  -> aux_hidden_states [38, 39, 40, 41]
  -> HiddenStateCacheSpec
  -> DeepSpecMooncakeConnector
  -> Mooncake
  -> online OnlineFeatureDataset
```

该 commit 修改了四个文件：

1. `vllm_ascend/distributed/kv_transfer/deepspec_mooncake_connector.py`
   新增 `DeepSpecMooncakeConnector`。它是仅写入的 producer connector，
   从 `CacheOnlyAttentionLayer` 的独立 hidden-state cache 读取每个请求的
   hidden states 和 token IDs，在 TP rank 0 生成安全的 request key，并通过
   Mooncake `put_tensor` 发布 `hidden_states`、`token_ids` 以及描述 shape、
   dtype 和状态的 manifest。发布失败时清理已写入对象并写入 error manifest；
   它不实现 consumer load。
2. `vllm_ascend/models/deepseek_v41/cache_config.py`
   引入 `HiddenStateCacheSpec` 的识别和分组，将 hidden-state cache 从 MLA/
   普通 state cache 中分离，并在 cache planner 中单独计算每块大小、容量和
   tensor descriptor。`_specs_from_groups()` 同时兼容嵌套的普通 cache group
   和直接的 hidden-state group，避免把 hidden-state spec 当作带
   `kv_cache_specs` 的普通 group 访问。
3. `vllm_ascend/worker/model_runner_v1.py`
   让 V1 NPU model runner 正确初始化和传递 cache-only layer 的 Tensor cache，
   同时为 reshape 路径预先定义 hybrid cache 标志，并跳过 hidden-state
   standalone layout 的 MLA layer stride 处理。
4. `vllm_ascend/worker/v2/attn_utils.py`
   在 V2 cache reshape 中识别 hidden-state cache group，避免将其按 MLA
   layer tuple 计算 stride，保持 V1/V2 cache layout 行为一致。

因此，`40b85a6...` 解决“最后一层 hidden state 从哪里产生”，
`21ad9a356...` 解决“这些状态如何单独分配、提取并传输”。online 训练端只
消费 Mooncake 中约定的 tensor 和 manifest，不需要实现 vLLM 的 cache planner、
worker reshape、slot mapping 或 Mooncake producer connector。

## 6. 实施顺序和验收

1. 确认 vLLM-Ascend 包含 `40b85a6bb1fdf3c7634678027040862441bfd22e` 和
   `21ad9a356745da9b2d38bcf4622a7c442e57e55b`。
2. producer 使用 capture IDs `[38, 39, 40, 41]`。
3. 确认 online 收到四层状态，前三层拼接为
   `[tokens, 3 * hidden_size]`，最后一层为 `[tokens, hidden_size]`。
4. 确认 `main_proj` 的输入维度为 `3 * hidden_size`。
5. 单卡完成一次 forward/backward，再验证 FSDP、checkpoint 和在线传输。
