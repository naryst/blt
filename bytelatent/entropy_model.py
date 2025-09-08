# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import logging
import os

import torch

from bytelatent.transformer import LMTransformer, LMTransformerArgs

logger = logging.getLogger()


def load_entropy_model(entropy_model_checkpoint_dir, state_dict_path, device="cpu"):
    with open(os.path.join(entropy_model_checkpoint_dir, "params.json")) as fr:
        reloaded = json.loads(fr.read())

    # Avoid globally changing default dtype unless explicitly requested
    prev_default_dtype = torch.get_default_dtype()
    use_bf16_default = os.environ.get("BLT_ENTROPY_DEFAULT_BF16", "0") == "1"
    if use_bf16_default:
        torch.set_default_dtype(torch.bfloat16)
    model_params = reloaded["entropy_model"]
    logger.warning(
        "Update checkpoint to load attn and sliding window args from checkpoint"
    )
    entropy_model_args = LMTransformerArgs(
        dim=model_params["dim"],
        n_layers=model_params["n_layers"],
        n_heads=model_params["n_heads"],
        max_seqlen=model_params["max_seqlen"],
        ffn_dim_multiplier=model_params["ffn_dim_multiplier"],
        vocab_size=model_params["vocab_size"],
        attn_bias_type="local_block_causal",
        attn_impl="sdpa",
        sliding_window=512,
    )
    entropy_model = LMTransformer(entropy_model_args)

    entropy_model.load_state_dict(
        torch.load(state_dict_path, map_location=device)["model"], strict=False
    )
    entropy_model.to(device)
    entropy_model = entropy_model.eval()
    # no grads for the model:
    for param in entropy_model.parameters():
        param.requires_grad = False
    # Restore previous global default dtype to prevent dtype mismatches elsewhere
    if use_bf16_default and torch.get_default_dtype() != prev_default_dtype:
        torch.set_default_dtype(prev_default_dtype)
    return entropy_model, entropy_model_args
