from pathlib import Path
from typing import Any, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

HERE = Path(__file__).resolve().parent

df = pd.read_csv(HERE / "perf.csv")
speedup_df = df.pivot(index="prefix_len", columns="response_len", values="speedup")

df["saved_gb"] = df["ref_gb"] - df["opt_gb"]
mem_df = df.pivot(index="prefix_len", columns="response_len", values="saved_gb")


def format_seqlen(x):
    if x < 1024:
        return str(x)
    return f"{x // 1024}k"


fig, (ax1, ax2) = cast(Tuple[Any, Tuple[Axes, Axes]], plt.subplots(1, 2, figsize=(16, 6)))

for dfi in [speedup_df, mem_df]:
    dfi.columns.name = "Response Length"
    dfi.index.name = "Prefix Length"

x_labels = [format_seqlen(x) for x in speedup_df.columns]
y_labels = [format_seqlen(y) for y in speedup_df.index]

ax1.set_title("Performance Speedup")
sns.heatmap(speedup_df, xticklabels=x_labels, yticklabels=y_labels, cmap=None, annot=True, fmt=".2f", ax=ax1)

ax2.set_title("Memory Saved (GB)")
sns.heatmap(mem_df, xticklabels=x_labels, yticklabels=y_labels, cmap="mako", annot=True, fmt=".2f", ax=ax2)

save_path = HERE.parent / "docs/perf.png"
plt.savefig(save_path)
print(f"Performance graph saved to {save_path}")
