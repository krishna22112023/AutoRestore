import matplotlib.pyplot as plt

# Data: mean total inference time (s) and performance metrics
methods = [
    "planned_no_optimisation",
    "planned_opt_no_verify",
    "planned_opt_verify",
    "finetuned_planned_opt_verify",
    "random_planning",
]

# Mean total inference time in seconds (std removed)
total_inference_time = [
    84.59,
    100.23,
    141.98,
    107.01,
    73.70,
]

# Performance metrics
metrics = {
    "PSNR (dB)": [
        63.8793,
        63.9316,
        63.9337,
        63.7702,
        63.9129,
    ],
    "SSIM": [
        0.4719,
        0.4664,
        0.4723,
        0.4653,
        0.4739,
    ],
    "LPIPS (↓)": [
        0.3229,
        0.3216,
        0.3180,
        0.3250,
        0.3269,
    ],
    "MANIQA (↑)": [
        0.1961,
        0.1923,
        0.1983,
        0.1993,
        0.1938,
    ],
    "CLIPIQA (↑)": [
        0.2058,
        0.2088,
        0.2146,
        0.2091,
        0.2001,
    ],
    "MUSIQ (↑)": [
        26.9772,
        27.0952,
        26.7065,
        26.9063,
        26.3808,
    ],
}

# Colors for each method
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(methods))]

# Mapping from internal method name to display label
label_map = {
    "planned_no_optimisation": "planning only",
    "planned_opt_no_verify": "planning + optimization",
    "planned_opt_verify": "planning + optimization + verify",
    "finetuned_planned_opt_verify": "finetuned_planning + optimization + verify",
    "random_planning": "random planning",
}

# Create figure with 2x3 subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 9))
axs = axs.flatten()

for idx, (metric_name, values) in enumerate(metrics.items()):
    for i, method in enumerate(methods):
        axs[idx].scatter(
            total_inference_time[i],
            values[i],
            color=colors[i],
            label=label_map[method] if idx == 0 else None,  # add legend labels only once
            s=60,
            edgecolor="k",
        )
    axs[idx].set_xlabel("Total Inference Time (s)", fontsize=12)
    axs[idx].set_ylabel(metric_name, fontsize=12)
    axs[idx].set_title(f"Inference Speed vs {metric_name}", fontsize=14)
    axs[idx].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Common legend added only once (from first axis handles)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("/home/krishna/workspace/AutoRestore/eval/ablation_exp/d3/ablation_exp_d3.png", bbox_inches="tight")
plt.show()