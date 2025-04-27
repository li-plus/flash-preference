from __future__ import annotations

import inspect
from contextlib import ExitStack, contextmanager
from itertools import chain
from typing import TYPE_CHECKING, Optional, Sequence, Tuple
from unittest.mock import patch

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaRotaryEmbedding


def to_shared(
    hidden_states: torch.Tensor, prefix_lens: Sequence[int], sequence_lens: Sequence[int], interleaved: bool
) -> torch.Tensor:
    batch_size = len(hidden_states)
    assert batch_size % len(prefix_lens) == 0, f"Inconsistent number of responses for each prompt"
    group_size = batch_size // len(prefix_lens)

    max_len = hidden_states.shape[1]

    split_sizes = []
    for seq_idx, seq_len in enumerate(sequence_lens):
        if not interleaved:
            prefix_idx, response_idx = divmod(seq_idx, group_size)
        else:
            response_idx, prefix_idx = divmod(seq_idx, len(prefix_lens))
        prefix_len = prefix_lens[prefix_idx]
        if response_idx == 0:
            split_sizes += [0, seq_len, max_len - seq_len]
        else:
            split_sizes += [prefix_len, seq_len - prefix_len, max_len - seq_len]

    split_states = hidden_states.flatten(end_dim=1).split(split_sizes)
    shared_states = torch.cat(split_states[1::3]).unsqueeze(0)
    return shared_states


def to_unshared(
    hidden_states: torch.Tensor,
    prefix_lens: Sequence[int],
    sequence_lens: Sequence[int],
    interleaved: bool,
    padding: bool = True,
) -> torch.Tensor:
    assert len(hidden_states) == 1, f"Expect batch_size to be 1, but got {len(hidden_states)}"
    hidden_states = hidden_states.squeeze(0)

    assert (
        len(sequence_lens) % len(prefix_lens) == 0
    ), f"Expect equal number of responses for each prefix, but got {len(prefix_lens)} prefixes and {len(sequence_lens)} sequences"
    group_size = len(sequence_lens) // len(prefix_lens)

    split_sizes = []
    for seq_idx, seq_len in enumerate(sequence_lens):
        if not interleaved:
            prefix_idx, response_idx = divmod(seq_idx, group_size)
        else:
            response_idx, prefix_idx = divmod(seq_idx, len(prefix_lens))
        prefix_len = prefix_lens[prefix_idx]
        if response_idx == 0:
            split_sizes.append(prefix_len)
        split_sizes.append(seq_len - prefix_len)

    split_states = hidden_states.split(split_sizes)

    unshared_states = []
    if not interleaved:
        # transform [A, a1, a2, B, b1, b2, C, c1, c2] to [A, a1, A, a2, B, b1, B, b2, C, c1, C, c2]
        for i, states in enumerate(split_states):
            if i % (group_size + 1) == 0:
                prefix_states = states
                continue
            unshared_states += [prefix_states, states]
    else:
        # transform [A, a1, B, b1, C, c1, a2, b2, c2] to [A, a1, B, b1, C, c1, A, a2, B, b2, C, c2]
        prefix_states = split_states[: 2 * len(prefix_lens) : 2]
        unshared_states += split_states[: 2 * len(prefix_lens)]
        for i, states in enumerate(split_states[2 * len(prefix_lens) :]):
            prefix_idx = i % len(prefix_lens)
            unshared_states += [prefix_states[prefix_idx], states]

    unshared_states = torch.cat(unshared_states)
    if padding:
        unshared_states = pad_sequence(unshared_states.split(sequence_lens), batch_first=True)
    return unshared_states


@contextmanager
def patch_flash_attention_forward(prefix_lens: Sequence[int], sequence_lens: Sequence[int], interleaved: bool):
    from flash_attn import flash_attn_varlen_func
    from transformers.integrations import flash_attention
    from transformers.modeling_flash_attention_utils import (
        _flash_supports_window_size,
        deterministic_g,
        fa_peft_integration_check,
        flash_241,
    )
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    from transformers.models.qwen2_vl import modeling_qwen2_vl

    shared_attn_cu_seqlens_q = [0]
    shared_attn_cu_seqlens_k = [0]
    group_size = len(sequence_lens) // len(prefix_lens)
    for seq_idx, seq_len in enumerate(sequence_lens):
        if not interleaved:
            prefix_idx, response_idx = divmod(seq_idx, group_size)
        else:
            response_idx, prefix_idx = divmod(seq_idx, len(prefix_lens))
        prefix_len = prefix_lens[prefix_idx]
        if response_idx == 0:
            shared_attn_cu_seqlens_q.append(seq_len)
        else:
            shared_attn_cu_seqlens_q.append(seq_len - prefix_len)
        shared_attn_cu_seqlens_k.append(seq_len)

    shared_attn_max_length_q = max(shared_attn_cu_seqlens_q)
    shared_attn_max_length_k = max(shared_attn_cu_seqlens_k)

    shared_attn_cu_seqlens_q = torch.tensor(shared_attn_cu_seqlens_q, device="cuda").cumsum(dim=0, dtype=torch.int32)
    shared_attn_cu_seqlens_k = torch.tensor(shared_attn_cu_seqlens_k, device="cuda").cumsum(dim=0, dtype=torch.int32)

    def _flash_attention_forward_wrapper(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        use_top_left_mask: bool = False,
        softcap: Optional[float] = None,
        deterministic: Optional[bool] = None,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        target_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if not use_top_left_mask:
            causal = is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
            causal = is_causal and query_length != 1

        # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
        use_sliding_windows = (
            _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
        )
        flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

        if flash_241:
            if deterministic is None:
                deterministic = deterministic_g
            flash_kwargs["deterministic"] = deterministic

        if softcap is not None:
            flash_kwargs["softcap"] = softcap

        # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
        query_states, key_states, value_states = fa_peft_integration_check(
            query_states, key_states, value_states, target_dtype
        )

        batch_size = query_states.size(0)

        # ===== extra logic for prefix sharing =====
        assert causal, "Prefix sharing does not support causal=False"
        key_states = to_unshared(
            key_states, prefix_lens=prefix_lens, sequence_lens=sequence_lens, interleaved=interleaved, padding=False
        )
        value_states = to_unshared(
            value_states, prefix_lens=prefix_lens, sequence_lens=sequence_lens, interleaved=interleaved, padding=False
        )
        # ===== extra logic end =====

        query_states = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
        key_states = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
        value_states = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=shared_attn_cu_seqlens_q,
            cu_seqlens_k=shared_attn_cu_seqlens_k,
            max_seqlen_q=shared_attn_max_length_q,
            max_seqlen_k=shared_attn_max_length_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

        return attn_output

    original_params = inspect.signature(flash_attention._flash_attention_forward).parameters.keys()
    patch_params = inspect.signature(_flash_attention_forward_wrapper).parameters.keys()
    if original_params != patch_params:
        raise RuntimeError(
            "The signature of _flash_attention_forward in transformers is different from the patch function in flash-preference. Please check your transformers version."
        )

    with (
        patch.object(flash_attention, "_flash_attention_forward", _flash_attention_forward_wrapper),
        patch.object(modeling_qwen2_vl, "_flash_attention_forward", _flash_attention_forward_wrapper),
        patch.object(modeling_qwen2_5_vl, "_flash_attention_forward", _flash_attention_forward_wrapper),
    ):
        yield


@contextmanager
def patch_rotary_emb(
    rotary_emb: LlamaRotaryEmbedding, prefix_lens: Sequence[int], sequence_lens: Sequence[int], interleaved: bool
):
    def rotary_emb_forward_wrapper(self: LlamaRotaryEmbedding, *args, **kwargs):
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = old_forward(*args, **kwargs)
        cos, sin = position_embeddings
        is_mrope = cos.ndim == 4  # qwen2vl & qwen2.5vl
        if is_mrope:
            assert len(cos) == 3 and len(sin) == 3
            cos = cos.permute(1, 2, 3, 0)
            sin = sin.permute(1, 2, 3, 0)
        cos = to_shared(
            cos.expand(len(sequence_lens), *cos.shape[1:]),
            prefix_lens=prefix_lens,
            sequence_lens=sequence_lens,
            interleaved=interleaved,
        )
        sin = to_shared(
            sin.expand(len(sequence_lens), *sin.shape[1:]),
            prefix_lens=prefix_lens,
            sequence_lens=sequence_lens,
            interleaved=interleaved,
        )
        if is_mrope:
            cos = cos.permute(3, 0, 1, 2)
            sin = sin.permute(3, 0, 1, 2)
        return cos, sin

    old_forward = rotary_emb.forward
    with patch.object(rotary_emb, "forward", rotary_emb_forward_wrapper.__get__(rotary_emb)):
        yield


def repeat_sequence(seq: Sequence, repeats: int):
    return type(seq)(chain.from_iterable(zip(*[seq] * repeats)))


@contextmanager
def patch_qwen_visual_forward(
    model, sequence_lens: Sequence[int], prefix_lens: Sequence[int], input_ids: torch.Tensor, interleaved: bool
):

    def qwen_visual_forward_wrapper(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # share visual states
        group_size = len(sequence_lens) // len(prefix_lens)
        if not interleaved:
            image_nums = (input_ids == model.config.vision_start_token_id).sum(dim=-1).tolist()
            num_patches = [x.prod(dim=-1).sum() for x in grid_thw.split(image_nums)]
            shared_states = torch.cat(hidden_states.split(num_patches)[::group_size])
            shared_grid_thw = torch.cat(grid_thw.split(image_nums)[::group_size])
        else:
            shared_states = hidden_states.unflatten(dim=0, sizes=(group_size, -1))[0]
            shared_grid_thw = grid_thw.unflatten(dim=0, sizes=(group_size, -1))[0]

        # forward
        shared_output: torch.Tensor = old_forward(hidden_states=shared_states, grid_thw=shared_grid_thw)

        # unshare visual states
        if not interleaved:
            num_tokens = [x // self.config.spatial_merge_size**2 for x in num_patches[::group_size]]
            output = torch.cat(repeat_sequence(shared_output.split(num_tokens), group_size))
        else:
            output = shared_output.tile(group_size, 1)

        return output

    old_forward = model.visual.forward
    with patch.object(model.visual, "forward", qwen_visual_forward_wrapper.__get__(model.visual)):
        yield


@contextmanager
def patch_layer_attention(
    model, prefix_lens: Sequence[int], sequence_lens: Sequence[int], shared_attention: bool, interleaved: bool
):

    def wrap_layer_forward(
        self: LlamaDecoderLayer,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # share prefix before the first layer
        if self.self_attn.layer_idx == 0:
            hidden_states = to_shared(
                hidden_states, prefix_lens=prefix_lens, sequence_lens=sequence_lens, interleaved=interleaved
            )

        hidden_states, *extra = old_forward_map[self](hidden_states, *args, **kwargs)

        # unshare prefix after the last layer
        if self.self_attn.layer_idx == self.self_attn.config.num_hidden_layers - 1:
            hidden_states = to_unshared(
                hidden_states, prefix_lens=prefix_lens, sequence_lens=sequence_lens, interleaved=interleaved
            )

        return hidden_states, *extra

    def wrap_attention_forward(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        if not shared_attention:
            hidden_states = to_unshared(
                hidden_states, prefix_lens=prefix_lens, sequence_lens=sequence_lens, interleaved=interleaved
            )
        hidden_states, *extra = old_forward_map[self](hidden_states, *args, **kwargs)
        if not shared_attention:
            hidden_states = to_shared(
                hidden_states, prefix_lens=prefix_lens, sequence_lens=sequence_lens, interleaved=interleaved
            )
        return hidden_states, *extra

    old_forward_map = {}

    contexts = []
    for layer in model.model.layers:
        old_forward_map[layer] = layer.forward
        old_forward_map[layer.self_attn] = layer.self_attn.forward

        contexts += [
            patch.object(layer, "forward", wrap_layer_forward.__get__(layer)),
            patch.object(layer.self_attn, "forward", wrap_attention_forward.__get__(layer.self_attn)),
        ]

    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield


def get_prefix_lens(
    input_ids: torch.Tensor, sequence_lens: torch.Tensor, group_size: int, interleaved: bool
) -> torch.Tensor:
    if not interleaved:
        input_ids = input_ids.unflatten(dim=0, sizes=(-1, group_size))
        sequence_lens = sequence_lens.unflatten(dim=0, sizes=(-1, group_size))
    else:
        input_ids = input_ids.unflatten(dim=0, sizes=(group_size, -1)).transpose(0, 1)
        sequence_lens = sequence_lens.unflatten(dim=0, sizes=(group_size, -1)).transpose(0, 1)

    min_lens = sequence_lens.min(dim=1).values
    prefix_mask = (input_ids == input_ids.roll(shifts=1, dims=1)).all(dim=1)
    prefix_mask_min = prefix_mask.min(dim=-1)
    prefix_lens = torch.where(prefix_mask_min.values, min_lens, prefix_mask_min.indices)

    return prefix_lens


@contextmanager
def shared_prefix(
    model: PreTrainedModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    group_size: int = 2,
    interleaved: bool = False,
    enabled: bool = True,
):
    """Apply prefix sharing to model forward and backward.

    For preference data with multiple responses that share the same prompt, model within this context manager computes
    the shared prefix only once by removing duplicated prefixes. For example, sequences [A|a1, A|a2, B|b1, B|b2], where
    A/B denote prompts and a1/a2/b1/b2 denote the corresponding responses, are reorganized into [A|a1|a2, B|b1|b2]
    before forward and backward and restored afterwards, reducing computation and memory footprint without loss of accuracy.

    Args:
        model (PreTrainedModel): The causal language model.
        input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary. Same as the `input_ids` passed into
            model forward function.
        attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices. Same as the
            `attention_mask` passed into the model forward function.
        group_size (int, optional): The size of the group in which all sequences share the same prefix. For pairwise
            data with a chosen and a rejected response, the group size is 2. For preference data with multiple
            responses per prompt, the group size should equal the number of responses per prompt. Defaults to 2.
        interleaved (bool, optional): Describes the sequence order of `input_ids`. If true, all groups are merged
            interleaved in the order like [A|a1, B|b1, A|a2, B|b2]. Otherwise, all groups are directly concatenated in
            the order like [A|a1, A|a2, B|b1, B|b2]. Defaults to False.
        enabled (bool, optional): Whether prefix sharing is enabled within this context manager. Defaults to True.
    """
    if not enabled:
        yield
        return

    if isinstance(model, FSDP):
        model = model.module

    if isinstance(model, DDP):
        model = model.module

    # ensure right padding
    if not attention_mask[:, 0].all():
        raise RuntimeError("Expect input_ids to be right padded, but got padding tokens at the beginning")

    # ensure padded to the longest
    if not attention_mask[:, -1].any():
        raise RuntimeError(
            "Expect input_ids to be padded to the longest in the batch, but got extra padding tokens at the end"
        )

    # prepare context
    sequence_lens = attention_mask.sum(dim=-1)
    prefix_lens = get_prefix_lens(
        input_ids=input_ids, sequence_lens=sequence_lens, group_size=group_size, interleaved=interleaved
    )
    shared_attention = model.config._attn_implementation == "flash_attention_2"
    prefix_lens = prefix_lens.tolist()
    sequence_lens = sequence_lens.tolist()

    # monkey patch
    contexts = [
        patch_layer_attention(
            model,
            prefix_lens=prefix_lens,
            sequence_lens=sequence_lens,
            shared_attention=shared_attention,
            interleaved=interleaved,
        )
    ]

    if model.config.model_type in ("qwen2_vl", "qwen2_5_vl"):
        contexts.append(
            patch_qwen_visual_forward(
                model,
                sequence_lens=sequence_lens,
                prefix_lens=prefix_lens,
                input_ids=input_ids,
                interleaved=interleaved,
            )
        )

    if shared_attention:
        contexts += [
            patch_rotary_emb(
                model.model.rotary_emb, prefix_lens=prefix_lens, sequence_lens=sequence_lens, interleaved=interleaved
            ),
            patch_flash_attention_forward(
                prefix_lens=prefix_lens, sequence_lens=sequence_lens, interleaved=interleaved
            ),
        ]

    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield
