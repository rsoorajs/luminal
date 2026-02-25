"""Tiny Llama 3 model for testing.

Uses HuggingFace's AutoConfig with the local config.json to load the Llama 3
architecture, then overrides dimensions to be tiny for fast test execution.
"""

import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


def create_llama3_model() -> nn.Module:
    """Create a tiny Llama 3 model using the local config.json."""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config = AutoConfig.from_pretrained(config_dir)

    # Override for tiny test dimensions
    config.hidden_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.intermediate_size = 128
    config.vocab_size = 256
    config.max_position_embeddings = 64

    # Use float32 so ONNX initializers are f32 (luminal doesn't handle bf16)
    config.torch_dtype = "float32"

    # Disable features that complicate ONNX export
    config.rope_scaling = {"rope_type": "default", "rope_theta": 500000.0}
    config.use_cache = False
    config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_config(config)
    model.to(torch.float32)
    model.eval()
    return model


class Llama3Model(nn.Module):
    """Wrapper that returns raw logits tensor (HF models return dataclass)."""

    def __init__(self):
        super().__init__()
        self.model = create_llama3_model()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, use_cache=False)
        return outputs.logits
