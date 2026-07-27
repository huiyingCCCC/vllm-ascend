# SPDX-License-Identifier: Apache-2.0
"""Source regressions for the DSpark context-cache Triton kernels."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRITON_UTILS = ROOT / "vllm_ascend" / "ops" / "triton" / "spec_decode" / "utils.py"


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(TRITON_UTILS.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {TRITON_UTILS}")


def _cache_position_store_mask(function_name: str) -> str:
    function = _function(function_name)
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or (node.func.value.id, node.func.attr) != ("tl", "store"):
            continue
        if not node.args or "cache_positions_ptr" not in ast.unparse(node.args[0]):
            continue
        mask = next(keyword.value for keyword in node.keywords if keyword.arg == "mask")
        return ast.unparse(mask)
    raise AssertionError(f"cache-position tl.store not found in {function_name}")


def test_dspark_cache_position_stores_use_scalar_masks() -> None:
    for function_name in (
        "store_dspark_context_kv_kernel",
        "restore_dspark_context_cache_kernel",
    ):
        mask = _cache_position_store_mask(function_name)
        assert "is_first_head_block" in mask
        assert "head_offsets" not in mask
