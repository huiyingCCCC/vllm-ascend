# DSV4.1 离线保存 hidden states 适配方案

本文说明如何在 DSV4 既有 hidden-state 导出机制上适配 DSV4.1，并使用 `vllm-ascend + Mooncake + online` 完成 DSpark 离线特征保存。

## 一、整体适配方案

### 1. 复用 DSV4 的总体链路

整体链路保持 DSV4 的设计：

```text
DSV4.1 target model
        |
        | extract_hidden_states
        v
CacheOnlyAttentionLayer
        |
        | 独立 HiddenStateCacheSpec cache group
        v
DeepSpecMooncakeConnector
        |
        | Mooncake store
        v
prepare_target_cache_vllm.py
        |
        v
manifest.json + samples.idx + shard-*.bin
        |
        v
DSpark training
```

不修改 upstream vLLM 的模型实现，而是在 Ascend 插件侧补齐 DSV4.1 特有的 cache group、cache layout 和 connector 处理。

### 2. 适配 DSV4.1 的差异

DSV4.1 适配集中在四点：

1. DSV4.1 训练目标层使用 zero-based `[37, 38, 39]`，producer capture id 对应 `[38, 39, 40]`。
2. 最终 `hc_collapse + norm` hidden state 通过 capture id `41` 单独保存为 verifier target。
3. hidden-state cache 是单 tensor、独立 cache group，不应套用 DSV4 MLA 的 K/V tuple layout。
4. vLLM 当前版本的 `CacheOnlyAttentionBackend` 没有 `get_kv_cache_shape()`，因此由 Ascend 侧专用 reshape 分支构造 cache view。

### 3. 数据约定

producer wire 数据为 `[tokens, 4, hidden_size]`。online 侧拆分为：

```text
states[:, :3] -> target_hidden_states      [tokens, 3, hidden_size]
states[:, 3]  -> target_last_hidden_states [tokens, hidden_size]
```

前三层随后展平为 `[tokens, 3 * hidden_size]` 写入训练 cache。第 41 个 sentinel 不能拼入前三层训练目标。

## 二、层号映射和配置

producer 使用 `extract_hidden_states`，从 cache-only cache 中提取四个 capture：

| Capture ID | 含义 | 最终字段 |
| ---: | --- | --- |
| 38 | zero-based decoder layer 37 | `target_hidden_states` 第 1 段 |
| 39 | zero-based decoder layer 38 | `target_hidden_states` 第 2 段 |
| 40 | zero-based decoder layer 39 | `target_hidden_states` 第 3 段 |
| 41 | 最终 `hc_collapse + norm` sentinel | `target_last_hidden_states` |

训练目标层保持：

```python
target_layer_ids = [37, 38, 39]
```

因此最终形状为：

```text
target_hidden_states      [tokens, 3 * hidden_size]
target_last_hidden_states [tokens, hidden_size]
```

## vLLM-Ascend 修改

当前分支为 `v0.30.0rc2`，相关提交：

```text
40b85a6bb1  feat(dspark): export final DeepSeek V4.1 hidden state
b74b27b50   feat(dspark): support DeepSeek V4.1 hidden-state export
7e1e492f0   fix(dspark): handle direct hidden-state cache groups
ec19db4e8   fix(dspark): bypass backend shape lookup for hidden states
fed621467   fix(dspark): match hidden-state cache layout
ed52eb392   fix(worker): initialize hybrid cache layout flag
32b546f44   fix(dspark): bind hidden-state cache as tensor
```

### `vllm_ascend/distributed/kv_transfer/deepspec_mooncake_connector.py`

新增 producer-only `DeepSpecMooncakeConnector`：

1. 识别唯一的 `HiddenStateCacheSpec` group。
2. 注册 `CacheOnlyAttentionLayer` 的独立 cache tensor。
3. 请求结束时记录 request、token ids 和 block ids。
4. 按 cache layout `[blocks, hidden_layers, block_tokens, hidden]` 读取。
5. 向 Mooncake 发布 `hidden_states`、`token_ids` 和 manifest。

读取方式是：

```python
cache[block_ids, :, token_offsets]
```

不是普通 attention KV cache 的 K/V tuple 访问方式。

### `vllm_ascend/models/deepseek_v41/cache_config.py`

- 引入并识别 `HiddenStateCacheSpec`。
- 把 hidden-state spec 从 DSV4.1 MLA/full-context tuple 中分离。
- 支持直接的 hidden-state group，不访问不存在的 `kv_cache_specs` 属性。
- 为 hidden-state cache 创建独立 `KVCacheTensor` descriptor。
- 将 hidden-state page bytes 纳入每 block 容量计算。

### `vllm_ascend/worker/model_runner_v1.py`

- hidden-state layer 绕过普通 MLA tuple reshape，使用单 tensor view：
  `[num_blocks, num_hidden_state_layers, block_size, hidden_size]`。
- 初始化 `hybrid_with_attn_and_mamba=False`，避免特殊 allocation 提前返回后 reshape 访问未初始化属性。
- DSV4.1 attention/compressor layer 仍绑定单元素 list；`CacheOnlyAttentionLayer` 绑定裸 Tensor：

```python
if isinstance(layer, CacheOnlyAttentionLayer):
    layer.kv_cache = kv_caches[name]
else:
    layer.kv_cache = [kv_caches[name]]
```

最后一项修复了：

```text
AttributeError: 'list' object has no attribute 'dtype'
```

### `vllm_ascend/worker/v2/attn_utils.py`

v2 reshape 路径同样排除 `HiddenStateCacheSpec`，避免调用当前 upstream 版本没有的：

```python
CacheOnlyAttentionBackend.get_kv_cache_shape()
```

没有修改 `/data/c00444317/deepspec/vllm` upstream 源码。

## online 修改

当前 online 分支为 `dsv41`。

### `config/dspark/dspark_deepseek_v41_flash.py`

关键配置：

```python
block_size=5
num_draft_layers=3
target_layer_ids=[37, 38, 39]
mask_token_id=128799
num_anchors=512
```

Mooncake 配置默认 `protocol="tcp"`。`local_hostname` 和 master 地址应通过运行时环境变量或 CLI 显式设置。

### `scripts/serve/launch_vllm_dsv41_deepspec.sh`

启动参数启用：

```text
--tokenizer-mode deepseek_v41
--speculative-config method=extract_hidden_states
--kv-connector DeepSpecMooncakeConnector
--kv-role kv_producer
```

并固定：

```text
eagle_aux_hidden_state_layer_ids=[38,39,40,41]
```

### `scripts/data/prepare_target_cache_vllm.py`

脚本逐条请求 producer，从 Mooncake 读取特征，校验 token ids、形状、dtype 和有限性，然后写入 target-cache shard。最终生成：

```text
manifest.json
samples.idx
shard-00000.bin
```

shard 内每个样本的顺序为：

```text
input_ids                 int32
attention_mask            uint8
loss_mask                 uint8
target_hidden_states      bfloat16
target_last_hidden_states bfloat16
```

`samples.idx` 保存每个字段的 offset，不能单独解析 `shard-00000.bin`。

## 启动和保存流程

```bash
mooncake_master --port 50051
```

```bash
export MOONCAKE_MASTER_SERVER=7.242.97.148:50051
export MOONCAKE_LOCAL_HOSTNAME=7.242.97.148
export ASCEND_BASE_PORT=61000

cd /data/c00444317/deepspec/online
bash scripts/serve/launch_vllm_dsv41_deepspec.sh
```

另开终端执行：

```bash
cd /data/c00444317/deepspec/online
python scripts/data/prepare_target_cache_vllm.py \
  --config config/dspark/dspark_deepseek_v41_flash.py \
  --train-data-path train_datasets/ds_v41/perfectblend_train_regen.jsonl \
  --output-dir "$HOME/.cache/deepspec/dsv41_target_cache" \
  --endpoint http://127.0.0.1:8077/v1 \
  --model-name dsv41 \
  --mooncake-local-hostname "$MOONCAKE_LOCAL_HOSTNAME" \
  --mooncake-master-server "$MOONCAKE_MASTER_SERVER" \
  --concurrency 1 \
  --min-loss-tokens 14 \
  --validate-outputs \
  --fail-on-error
```

`protocol=tcp` 控制 Store 对象传输协议；Ascend Mooncake wheel 仍可能初始化 `AscendDirectTransport`，所以不同服务仍需使用不同 `ASCEND_BASE_PORT`。

## 验证 cache

在 online 仓目录执行：

```bash
python - <<'PY'
import os, sys, torch
sys.path.insert(0, os.getcwd())
from deepspec.data.target_cache_dataset import CacheDataset

cache = CacheDataset(os.path.expanduser('~/.cache/deepspec/dsv41_target_cache'))
print('samples:', len(cache))
print('layers:', cache.target_layer_ids)
print('hidden_size:', cache.hidden_size)
assert cache.target_layer_ids == [37, 38, 39]
sample = cache[0]
print({k: (tuple(v.shape), str(v.dtype)) for k, v in sample.items()})
assert sample['target_hidden_states'].shape == (
    sample['input_ids'].shape[0], 3 * cache.hidden_size)
assert sample['target_last_hidden_states'].shape == (
    sample['input_ids'].shape[0], cache.hidden_size)
assert sample['target_hidden_states'].dtype == torch.bfloat16
assert sample['target_last_hidden_states'].dtype == torch.bfloat16
assert torch.isfinite(sample['target_hidden_states'].float()).all()
assert torch.isfinite(sample['target_last_hidden_states'].float()).all()
cache.close()
print('cache validation passed')
PY
```

如果只存在 `_tmp/rank_0/shard-local-00000.bin` 和 `samples.local.idx`，说明脚本尚未完成最终合并，不能直接用于训练。

## 已知限制

- 需要匹配的 vLLM、vLLM-Ascend、Mooncake 和 online 分支版本。
- 当前没有修改 upstream vLLM。
- 真实 NPU 权重、Mooncake 和完整训练链路需要在目标机器上做端到端验证。
- 第 41 个 capture 是最后一层 sentinel，不能拼入前三个训练目标层。
