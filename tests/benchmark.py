from itertools import product
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
import torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from tabulate import tabulate
from test_flash_pref import make_inputs
from torch.utils.benchmark import Timer
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, PreTrainedModel

from flash_pref import shared_prefix

apply_liger_kernel_to_qwen2()

HERE = Path(__file__).resolve().parent


def get_auto_model_class(config):
    if type(config) in AutoModelForVision2Seq._model_mapping:
        return AutoModelForVision2Seq
    return AutoModelForCausalLM


def benchmark_len(model: PreTrainedModel, prefix_lens: Sequence[int], response_lens: Sequence[int]):
    inputs = make_inputs(
        prefix_lens=prefix_lens,
        response_lens=response_lens,
        image_grid_thw=None,
        image_nums=None,
        interleaved=False,
        config=model.config,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    def run_forward_backward(enable_shared_prefix: bool):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with shared_prefix(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            enabled=enable_shared_prefix,
        ):
            stmt = """
output = model(**inputs)
output.loss.backward()
"""
            elapsed = Timer(stmt, globals=dict(model=model, inputs=inputs)).timeit(5).mean

        peak_memory = torch.cuda.max_memory_allocated()

        for param in model.parameters():
            param.grad = None  # zero grad

        return dict(elapsed=elapsed, peak_memory=peak_memory)

    ref_stats = run_forward_backward(enable_shared_prefix=False)
    opt_stats = run_forward_backward(enable_shared_prefix=True)

    return dict(
        ref_gb=ref_stats["peak_memory"] / 1e9,
        opt_gb=opt_stats["peak_memory"] / 1e9,
        ref_ms=ref_stats["elapsed"] * 1e3,
        opt_ms=opt_stats["elapsed"] * 1e3,
        speedup=ref_stats["elapsed"] / opt_stats["elapsed"],
    )


def benchmark(
    model_id: str,
    seqlens: Sequence[Tuple[int, int]],
    out_path: Path,
):
    table = []
    done_keys = set()
    if out_path.is_file():
        # resume from last run
        df = pd.read_csv(out_path)
        table = df.to_dict(orient="records")
        done_keys = set(tuple(x) for x in df[["prefix_len", "response_len"]].values.tolist())

    config = AutoConfig.from_pretrained(model_id, use_cache=False)
    if visual_config := getattr(config, "visual", None) is not None:
        visual_config.torch_dtype = torch.bfloat16
    auto_model_cls = get_auto_model_class(config)
    with torch.device("cuda"):
        model: PreTrainedModel = auto_model_cls.from_config(
            config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    model.gradient_checkpointing_enable()

    for prefix_len, response_len in tqdm(seqlens):
        if (prefix_len, response_len) in done_keys:
            continue
        row = dict(model_id=model_id, prefix_len=prefix_len, response_len=response_len)
        stats = benchmark_len(model, prefix_lens=[prefix_len], response_lens=[response_len] * 2)
        row.update(stats)
        table.append(row)

        df = pd.DataFrame(table)
        df.to_csv(out_path, index=False)

    df = pd.DataFrame(table)
    df = df.sort_values(by=["prefix_len", "response_len"])
    df.to_csv(out_path, index=False)

    print(tabulate(df, headers="keys", tablefmt="psql", floatfmt=".3f"))


def main():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    out_path = HERE / "perf.csv"

    seqlens = list(product([2**i for i in range(6, 15)], repeat=2))
    benchmark(model_id=model_id, seqlens=seqlens, out_path=out_path)


if __name__ == "__main__":
    main()
