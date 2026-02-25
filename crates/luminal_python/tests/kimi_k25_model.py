"""Tiny Kimi-K2.5 text backbone (DeepSeek-V3 fork) for testing.

Uses HuggingFace's AutoConfig with trust_remote_code to load the real
Kimi-K2.5 model architecture, then overrides dimensions to be tiny for
fast test execution.

Key architectural features exercised:
  - MLA (Multi-Latent Attention): Q/KV compressed through LoRA-like bottleneck
  - MoE (Mixture of Experts): routed + shared experts in FFN layers
"""

import importlib

import torch
import torch.nn as nn
from transformers import AutoConfig


def create_kimi_k25_text_model() -> nn.Module:
    """Create a tiny DeepSeek-V3 model using Kimi-K2.5's custom code."""
    # Load real config (downloads custom modeling code from HF)
    config = AutoConfig.from_pretrained(
        "moonshotai/Kimi-K2.5", trust_remote_code=True
    )

    # Override text_config for tiny test dimensions
    tc = config.text_config

    # Core dimensions
    tc.hidden_size = 64  # = num_heads * v_head_dim = 4 * 16
    tc.num_hidden_layers = 2
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 4

    # MLA bottleneck dimensions
    tc.q_lora_rank = 32
    tc.kv_lora_rank = 16
    tc.qk_nope_head_dim = 8
    tc.qk_rope_head_dim = 4
    tc.v_head_dim = 16

    # FFN dimensions
    tc.intermediate_size = 128

    # Vocabulary and sequence
    tc.vocab_size = 256
    tc.max_position_embeddings = 64

    # MoE configuration
    tc.n_routed_experts = 4
    tc.n_shared_experts = 1
    tc.num_experts_per_tok = 2
    tc.moe_intermediate_size = 32
    tc.first_k_dense_replace = 1  # layer 0 dense, layer 1+ MoE
    tc.n_group = 2  # divides n_routed_experts
    tc.topk_group = 1

    # Disable features that complicate ONNX export
    tc.num_nextn_predict_layers = 0  # no multi-token prediction
    tc.rope_scaling = None  # basic RoPE, not YaRN
    tc._attn_implementation = "eager"  # no flash attention
    tc.use_cache = False

    # Import DeepseekV3ForCausalLM from the downloaded custom code
    config_module = type(config).__module__
    modeling_module = config_module.replace(
        "configuration_kimi_k25", "modeling_deepseek"
    )
    DeepseekV3ForCausalLM = importlib.import_module(
        modeling_module
    ).DeepseekV3ForCausalLM

    model = DeepseekV3ForCausalLM(tc)
    model.eval()
    return model


class KimiK25TextModel(nn.Module):
    """Wrapper that returns raw logits tensor (HF models return dataclass)."""

    def __init__(self):
        super().__init__()
        self.model = create_kimi_k25_text_model()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, use_cache=False)
        return outputs.logits
