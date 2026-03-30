"""Qwen3-30B-A3B-Instruct-2507 (MoE) HuggingFace model integration tests.

Tests Qwen3MoeForCausalLM through the PT2/FX path (LUMINAL_EXPORT_MODE=pt2).
The Qwen3 MoE architecture uses Mixture-of-Experts with top-k routing,
grouped matrix multiplication, and operations like argsort/histc that may
not yet be supported in the PT2 translator.

Run with:
    LUMINAL_EXPORT_MODE=pt2 LUMINAL_BACKEND=cuda \
      uv run pytest tests/test_qwen3_moe.py -v -s
"""

import torch
import torch._dynamo

from luminal import luminal_backend


def _make_qwen3_moe_config(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_hidden_layers: int,
    intermediate_size: int,
    vocab_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    moe_intermediate_size: int,
    decoder_sparse_step: int = 1,
    norm_topk_prob: bool = False,
):
    """Create a Qwen3MoeConfig with use_cache=False and eager attention."""
    from transformers import Qwen3MoeConfig

    return Qwen3MoeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        decoder_sparse_step=decoder_sparse_step,
        norm_topk_prob=norm_topk_prob,
        max_position_embeddings=128,
        use_cache=False,
        attn_implementation="eager",
    )


def _run_hf_qwen3_moe_test(config, device: torch.device, atol: float):
    """Run a HuggingFace Qwen3MoeForCausalLM test with the given config."""
    from transformers import Qwen3MoeForCausalLM

    model = Qwen3MoeForCausalLM(config).eval().to(device)
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    assert torch.allclose(out.logits, ref.logits, atol=atol), (
        f"max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
    )


def test_hf_qwen3_moe_tiny(device: torch.device):
    """Qwen3MoeForCausalLM -- tiny MoE (64 hidden, 4 experts, 2 active, 1 layer)."""
    config = _make_qwen3_moe_config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        vocab_size=256,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
    )
    _run_hf_qwen3_moe_test(config, device, atol=1e-5)


def test_hf_qwen3_moe_small(device: torch.device):
    """Qwen3MoeForCausalLM -- small MoE (256 hidden, 8 experts, 2 active, 1 layer)."""
    config = _make_qwen3_moe_config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=1,
        intermediate_size=512,
        vocab_size=1024,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
    )
    _run_hf_qwen3_moe_test(config, device, atol=1e-4)


def test_hf_qwen3_moe_medium(device: torch.device):
    """Qwen3MoeForCausalLM -- medium MoE (256 hidden, 8 experts, 2 active, 2 layers)."""
    config = _make_qwen3_moe_config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=512,
        vocab_size=1024,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
    )
    _run_hf_qwen3_moe_test(config, device, atol=1e-4)


def test_hf_qwen3_moe_real_config_1layer(device: torch.device):
    """Qwen3MoeForCausalLM -- real Qwen3-30B-A3B architecture, 1 layer.

    Uses AutoConfig.from_pretrained with num_hidden_layers overridden to 1.
    Random weights -- tests the full-width 128-expert architecture compiles.
    """
    from transformers import AutoConfig, Qwen3MoeForCausalLM

    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
    config.num_hidden_layers = 1
    config.use_cache = False
    config._attn_implementation = "eager"

    model = Qwen3MoeForCausalLM(config).eval().to(device)
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    assert torch.allclose(out.logits, ref.logits, atol=1e-3), (
        f"max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
    )
