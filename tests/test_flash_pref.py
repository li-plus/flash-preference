import functools
import os
from typing import Optional, Sequence
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flash_attn import flash_attn_varlen_func
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    GemmaConfig,
    GemmaForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
)
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import get_module_class_from_name

from flash_pref.shared_prefix import get_prefix_lens, repeat_sequence, shared_prefix, to_shared, to_unshared


def create_qwen2_model(vocab_size: int, attn_implementation: str):
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        use_cache=False,
        attn_implementation=attn_implementation,
    )
    model = Qwen2ForCausalLM(config)
    return model


def create_qwen2_vl_model(vocab_size: int, attn_implementation: str):
    # https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/config.json
    config = Qwen2VLConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        use_cache=False,
        attn_implementation=attn_implementation,
        image_token_id=vocab_size - 1,
        video_token_id=vocab_size - 2,
        vision_start_token_id=vocab_size - 3,
        vision_end_token_id=vocab_size - 4,
        rope_scaling=dict(type="mrope", mrope_section=[8, 12, 12]),
        vision_config=dict(depth=4, embed_dim=128, hidden_size=512, mlp_ratio=2, num_heads=8),
    )
    model = Qwen2VLForConditionalGeneration(config)
    return model


def create_qwen2_5_vl_model(vocab_size: int, attn_implementation: str):
    # https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json
    config = Qwen2_5_VLConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        use_cache=False,
        attn_implementation=attn_implementation,
        image_token_id=vocab_size - 1,
        video_token_id=vocab_size - 2,
        vision_start_token_id=vocab_size - 3,
        vision_end_token_id=vocab_size - 4,
        rope_scaling=dict(type="mrope", mrope_section=[8, 12, 12]),
        vision_config=dict(depth=4, hidden_size=128, intermediate_size=256, num_heads=8, out_hidden_size=512),
    )
    model = Qwen2_5_VLForConditionalGeneration(config)
    return model


def create_llama_model(vocab_size: int, attn_implementation: str):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        use_cache=False,
        attn_implementation=attn_implementation,
    )
    model = LlamaForCausalLM(config)
    return model


def create_gemma_model(vocab_size: int, attn_implementation: str):
    # https://huggingface.co/google/gemma-2b-it/blob/main/config.json
    config = GemmaConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        hidden_act="gelu",
        # initializer_range=0.001,
        use_cache=False,
        attn_implementation=attn_implementation,
    )
    model = GemmaForCausalLM(config)
    return model


def create_gemma2_model(vocab_size: int, attn_implementation: str):
    # https://huggingface.co/google/gemma-2-2b-it/blob/main/config.json
    config = Gemma2Config(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        # initializer_range=0.001,
        use_cache=False,
        attn_implementation=attn_implementation,
    )
    model = Gemma2ForCausalLM(config)
    return model


def create_model(model_type: str, dtype: torch.dtype, vocab_size: int, attn_implementation: str) -> PreTrainedModel:
    model_factory = {
        "qwen2": create_qwen2_model,
        "llama": create_llama_model,
        "gemma": create_gemma_model,
        "gemma2": create_gemma2_model,
        "qwen2_vl": create_qwen2_vl_model,
        "qwen2_5_vl": create_qwen2_5_vl_model,
    }
    model_builder = model_factory[model_type]
    with torch.device("cuda"):
        model = model_builder(vocab_size=vocab_size, attn_implementation=attn_implementation).to(dtype=dtype)
    return model


def to_interleaved(x: torch.Tensor, group_size: int):
    return x.unflatten(dim=0, sizes=(-1, group_size)).transpose(0, 1).contiguous().flatten(end_dim=1)


def make_hidden_states(prefix_lens: Sequence[int], response_lens: Sequence[int], hidden_size: int, interleaved: bool):
    group_size, remainder = divmod(len(response_lens), len(prefix_lens))
    assert remainder == 0

    hidden_states = []
    for prefix_idx, prefix_len in enumerate(prefix_lens):
        prefix_states = torch.randn(prefix_len, hidden_size, device="cuda")
        for response_len in response_lens[prefix_idx * group_size : (prefix_idx + 1) * group_size]:
            response_states = torch.randn(response_len, hidden_size, device="cuda")
            seq_states = torch.cat((prefix_states, response_states), dim=0)
            hidden_states.append(seq_states)

    hidden_states = pad_sequence(hidden_states, batch_first=True).cuda()
    sequence_lens = (np.repeat(prefix_lens, group_size) + response_lens).tolist()

    if interleaved:
        hidden_states = to_interleaved(hidden_states, group_size=group_size)
        sequence_lens = to_interleaved(torch.tensor(sequence_lens), group_size=group_size).tolist()

    return hidden_states, sequence_lens


def make_inputs(
    prefix_lens: Sequence[int],
    response_lens: Sequence[int],
    image_grid_thw: Optional[Sequence[Sequence[int]]],
    image_nums: Optional[Sequence[int]],
    interleaved: bool,
    config: PretrainedConfig,
):
    group_size, remainder = divmod(len(response_lens), len(prefix_lens))
    assert remainder == 0

    input_ids = []
    attention_mask = []
    for seq_idx, response_len in enumerate(response_lens):
        prefix_idx, response_idx = divmod(seq_idx, group_size)
        prefix_len = prefix_lens[prefix_idx]
        seq_ids = torch.cat((torch.arange(prefix_len), torch.arange(response_idx, response_idx + response_len)))
        seq_mask = torch.ones(prefix_len + response_len, dtype=torch.long)
        input_ids.append(seq_ids)
        attention_mask.append(seq_mask)

    input_ids = pad_sequence(input_ids, batch_first=True).cuda()
    attention_mask = pad_sequence(attention_mask, batch_first=True).cuda()

    multimodal_inputs = {}
    if config.model_type in ("qwen2_vl", "qwen2_5_vl") and image_grid_thw is not None and image_nums is not None:
        # make shared pixel_values based on shared image_grid_thw
        image_grid_thw = torch.tensor(image_grid_thw)

        pixel_values = []
        for thw in image_grid_thw:
            vision_config = config.vision_config
            num_patches = thw.prod()
            patch_size = (
                vision_config.in_channels
                * vision_config.temporal_patch_size
                * vision_config.patch_size
                * vision_config.patch_size
            )
            pixel_values.append(torch.randn(num_patches, patch_size))
        pixel_values = torch.cat(pixel_values)

        assert sum(image_nums) == len(image_grid_thw)
        image_grid_thw_splits = image_grid_thw.split(image_nums, dim=0)

        # set vision token into input_ids
        vision_tokens = []
        for image_grid_thw_split in image_grid_thw_splits:
            curr_vision_tokens = []
            for thw in image_grid_thw_split:
                curr_vision_tokens += (
                    [config.vision_start_token_id]
                    + [config.image_token_id] * (thw.prod() // vision_config.spatial_merge_size**2)
                    + [config.vision_end_token_id]
                )
            vision_tokens.append(torch.tensor(curr_vision_tokens))

        vision_start_idx = 7
        for prefix_idx, curr_vision_tokens in enumerate(vision_tokens):
            for seq_ids in input_ids[prefix_idx * group_size : (prefix_idx + 1) * group_size]:
                seq_ids[vision_start_idx : vision_start_idx + len(curr_vision_tokens)] = curr_vision_tokens

        # replicate pixel_values & image_grid_thw for all responses
        if not interleaved:
            pixel_values_sizes = [x.prod(dim=-1).sum() for x in image_grid_thw_splits]
            pixel_values_splits = pixel_values.split(pixel_values_sizes)
            pixel_values = torch.cat(repeat_sequence(pixel_values_splits, group_size))
            image_grid_thw = torch.cat(repeat_sequence(image_grid_thw_splits, group_size))
        else:
            pixel_values = pixel_values.tile(group_size, 1)
            image_grid_thw = image_grid_thw.tile(group_size, 1)

        multimodal_inputs.update(pixel_values=pixel_values.cuda(), image_grid_thw=image_grid_thw.cuda())

    if interleaved:
        input_ids = to_interleaved(input_ids, group_size=group_size)
        attention_mask = to_interleaved(attention_mask, group_size=group_size)

    labels = torch.where(attention_mask.bool(), input_ids, -100)

    return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **multimodal_inputs)


@pytest.mark.parametrize(
    "prefix_lens,response_lens",
    [
        ((256,), (64, 32)),
        ((128, 64, 200), (64, 32, 23, 100, 123, 25)),
        ((128, 200), (64, 32, 23, 100, 123, 25)),  # 3 responses per prefix
        ((0, 16, 16, 16), (8, 8, 0, 0, 0, 1, 1, 0)),  # empty prefix or response
    ],
)
@pytest.mark.parametrize("interleaved", [True, False])
def test_to_shared_unshared(prefix_lens, response_lens, interleaved: bool):
    hidden_size = 1024
    hidden_states, sequence_lens = make_hidden_states(
        prefix_lens=prefix_lens, response_lens=response_lens, hidden_size=hidden_size, interleaved=interleaved
    )
    shared_states = to_shared(
        hidden_states=hidden_states, prefix_lens=prefix_lens, sequence_lens=sequence_lens, interleaved=interleaved
    )
    assert shared_states.shape == (1, sum(prefix_lens) + sum(response_lens), hidden_size)

    unshared_states = to_unshared(
        hidden_states=shared_states, prefix_lens=prefix_lens, sequence_lens=sequence_lens, interleaved=interleaved
    )
    torch.testing.assert_close(unshared_states, hidden_states)


def test_fused_flash_attn():
    num_heads = 8
    head_dim = 64
    prefix_len = 512
    chosen_len = 64
    rejected_len = 32
    prefix_qkv = torch.randn(3, prefix_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    chosen_qkv = torch.randn(3, chosen_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    rejected_qkv = torch.randn(3, rejected_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    # naive attn
    qkv = torch.cat((prefix_qkv, chosen_qkv, prefix_qkv, rejected_qkv), dim=1)
    cu_seqlens = torch.tensor([0, prefix_len + chosen_len, prefix_len + rejected_len], device="cuda").cumsum(
        dim=0, dtype=torch.int32
    )
    max_seqlen = prefix_len + max(chosen_len, rejected_len)
    q, k, v = qkv
    output_ref = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    # fused attn
    q, _, _ = torch.cat((prefix_qkv, chosen_qkv, rejected_qkv), dim=1)
    cu_seqlens_q = torch.tensor([0, prefix_len + chosen_len, rejected_len], device="cuda").cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens_k = cu_seqlens
    max_seqlen_q = max(prefix_len + chosen_len, rejected_len)
    max_seqlen_k = max_seqlen
    output_opt = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=True,
    )
    prefix_output, chosen_output, rejected_output = output_opt.split([prefix_len, chosen_len, rejected_len])
    output_opt = torch.cat([prefix_output, chosen_output, prefix_output, rejected_output])

    # check
    torch.testing.assert_close(output_opt, output_ref)


def _test_flash_pref(
    model_type,
    dtype,
    rtol,
    atol,
    attn_implementation,
    prefix_lens,
    response_lens,
    image_grid_thw,
    image_nums,
    interleaved,
    use_liger_kernel,
    parallel_mode: Optional[str] = None,
):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if parallel_mode is not None:
        dist.init_process_group(backend="nccl")

    model = create_model(model_type=model_type, dtype=dtype, vocab_size=2048, attn_implementation=attn_implementation)
    model.gradient_checkpointing_enable()

    if use_liger_kernel:
        _apply_liger_kernel_to_instance(model)

    if parallel_mode == "fsdp":
        transformer_layer_cls = {get_module_class_from_name(model, name) for name in model._no_split_modules}
        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls)
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            sync_module_states=True,
            use_orig_params=True,
        )
    elif parallel_mode == "ddp":
        model = DDP(model)
    else:
        assert parallel_mode is None, f"Unknown parallel_mode: {parallel_mode}"

    inputs = make_inputs(
        prefix_lens=prefix_lens,
        response_lens=response_lens,
        image_grid_thw=image_grid_thw,
        image_nums=image_nums,
        interleaved=interleaved,
        config=unwrap_model(model).config,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    group_size = len(response_lens) // len(prefix_lens)

    derived_prefix_lens = get_prefix_lens(
        input_ids=input_ids,
        sequence_lens=attention_mask.sum(-1),
        group_size=group_size,
        interleaved=interleaved,
    )
    torch.testing.assert_close(derived_prefix_lens.cpu(), torch.tensor(prefix_lens))

    # reference
    ref_output = model(**inputs)
    ref_output.loss.backward()

    ref_grads = {}
    for name, param in model.named_parameters():
        ref_grads[name] = param.grad
        param.grad = None  # zero grad

    # prefix sharing
    with shared_prefix(
        model, input_ids=input_ids, attention_mask=attention_mask, group_size=group_size, interleaved=interleaved
    ):
        opt_output = model(**inputs)
        opt_output.loss.backward()

    opt_grads = {}
    for name, param in model.named_parameters():
        opt_grads[name] = param.grad
        param.grad = None  # zero grad

    # check
    torch.testing.assert_close(opt_output.loss, ref_output.loss, rtol=rtol, atol=atol)
    if not use_liger_kernel:
        torch.testing.assert_close(
            opt_output.logits * attention_mask.unsqueeze(-1),
            ref_output.logits * attention_mask.unsqueeze(-1),
            rtol=rtol,
            atol=atol,
        )
    torch.testing.assert_close(opt_grads, ref_grads, rtol=rtol, atol=atol)

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "model_type",
    [
        "qwen2",
        "llama",
        "gemma",
        "gemma2",
        "qwen2_vl",
        "qwen2_5_vl",
    ],
)
@pytest.mark.parametrize(
    "dtype,rtol,atol,attn_implementation",
    [
        (torch.float32, None, None, "eager"),
        (torch.float32, None, None, "sdpa"),
        (torch.bfloat16, 1e-2, 2e-2, "flash_attention_2"),
    ],
)
@pytest.mark.parametrize(
    "prefix_lens,response_lens,image_grid_thw,image_nums",
    [
        # batch size 1
        ((256,), (64, 32), [[1, 8, 16]], (1,)),
        # batch size 3
        ((128, 64, 200), (64, 32, 23, 100, 123, 25), [[1, 8, 16], [1, 12, 20], [1, 12, 8]], (2, 0, 1)),
        # 3 responses per prefix + w/ or w/o liger kernel
        ((128, 200), (64, 32, 23, 100, 123, 25), [[1, 8, 16], [1, 12, 20], [1, 12, 8]], (2, 1)),
        # empty prefix or response
        ((0, 16, 16, 16), (8, 8, 0, 0, 0, 1, 1, 0), None, None),
    ],
)
@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("use_liger_kernel", [False])
def test_flash_pref(
    model_type,
    dtype,
    rtol,
    atol,
    attn_implementation,
    prefix_lens,
    response_lens,
    image_grid_thw,
    image_nums,
    interleaved,
    use_liger_kernel,
):
    _test_flash_pref(
        model_type=model_type,
        dtype=dtype,
        rtol=rtol,
        atol=atol,
        attn_implementation=attn_implementation,
        prefix_lens=prefix_lens,
        response_lens=response_lens,
        image_grid_thw=image_grid_thw,
        image_nums=image_nums,
        interleaved=interleaved,
        use_liger_kernel=use_liger_kernel,
    )


def _test_flash_pref_parallel_wrapper(rank, world_size, *args):
    with patch.dict(
        os.environ,
        {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": os.getenv("MASTER_ADDR", "127.0.0.1"),
            "MASTER_PORT": os.getenv("MASTER_PORT", "29500"),
            "LOCAL_RANK": str(rank),
        },
    ):
        _test_flash_pref(*args)


@pytest.mark.parametrize(
    "model_type",
    [
        "qwen2",
        "llama",
        "gemma",
        "gemma2",
        "qwen2_vl",
        "qwen2_5_vl",
    ],
)
@pytest.mark.parametrize(
    "dtype,rtol,atol,attn_implementation",
    [
        (torch.bfloat16, 1e-2, 2e-2, "flash_attention_2"),
    ],
)
@pytest.mark.parametrize(
    "prefix_lens,response_lens,image_grid_thw,image_nums,use_liger_kernel",
    [
        ((256,), (64, 32), [[1, 8, 16]], (1,), False),
        ((256,), (64, 32), [[1, 8, 16]], (1,), True),
    ],
)
@pytest.mark.parametrize("interleaved", [False])
@pytest.mark.parametrize("parallel_mode", ["fsdp", "ddp"])
def test_flash_pref_parallel(
    model_type,
    dtype,
    rtol,
    atol,
    attn_implementation,
    prefix_lens,
    response_lens,
    image_grid_thw,
    image_nums,
    interleaved,
    use_liger_kernel,
    parallel_mode,
):

    world_size = 2

    mp.spawn(
        _test_flash_pref_parallel_wrapper,
        args=(
            world_size,
            model_type,
            dtype,
            rtol,
            atol,
            attn_implementation,
            prefix_lens,
            response_lens,
            image_grid_thw,
            image_nums,
            interleaved,
            use_liger_kernel,
            parallel_mode,
        ),
        nprocs=world_size,
    )
