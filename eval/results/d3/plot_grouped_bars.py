import matplotlib.pyplot as plt
import numpy as np

# Methods and their display labels
methods = [
    "planned_no_optimisation",
    "planned_opt_no_verify",
    "random_planning",
    "planned_opt_verify",
    "finetuned_planned_opt_verify",
]
label_map = {
    "planned_no_optimisation": "planning only",
    "planned_opt_no_verify": "planning + optimisation",
    "planned_opt_verify": "planning + optimisation + verify",
    "finetuned_planned_opt_verify": "finetuned planning + optimisation + verify",
    "random_planning": "random planning",
}

# Degradation categories (4 groups)
categories = [
    "dark_defocus_blur_jpeg compression_artifact",
    "haze_motion_blur_low_resolution",
    "motion_blur_defocus_blur_noise",
    "rain_noise_low_resolution",
]

# Metric data organised as {metric_name: {category: [values per method]}}
metric_data = {
    "LPIPS (↓)": {
        categories[0]: [0.4003, 0.3994, 0.3998, 0.3836, 0.4070],
        categories[1]: [0.2078, 0.2159, 0.2016, 0.1954, 0.2059],
        categories[2]: [0.4638, 0.4583, 0.4734, 0.4674, 0.4625],
        categories[3]: [0.2196, 0.2126, 0.2330, 0.2256, 0.2247],
    },
    "PSNR (dB) (↑)": {
        categories[0]: [63.5075, 63.6196, 63.5293, 63.6564, 63.3087],
        categories[1]: [63.2105, 63.1096, 63.5137, 63.3215, 63.2246],
        categories[2]: [63.4557, 63.5292, 63.6322, 63.5525, 63.5194],
        categories[3]: [65.3436, 65.4682, 64.9764, 65.2046, 65.0284],
    },
    "SSIM (↑)": {
        categories[0]: [0.4195, 0.4182, 0.4175, 0.4381, 0.4008],
        categories[1]: [0.5097, 0.4915, 0.5297, 0.5134, 0.5142],
        categories[2]: [0.4085, 0.4065, 0.4123, 0.4091, 0.4094],
        categories[3]: [0.5497, 0.5494, 0.5363, 0.5285, 0.5369],
    },
    "MANIQA (↑)": {
        categories[0]: [0.3072, 0.3116, 0.3075, 0.3275, 0.3113],
        categories[1]: [0.4506, 0.4145, 0.4497, 0.4527, 0.4505],
        categories[2]: [0.3332, 0.3309, 0.3255, 0.3303, 0.3316],
        categories[3]: [0.5172, 0.5141, 0.5228, 0.5212, 0.5253],
    },
    "CLIPIQA (↑)": {
        categories[0]: [0.4314, 0.4357, 0.4278, 0.4678, 0.4366],
        categories[1]: [0.7491, 0.7394, 0.7493, 0.7586, 0.7490],
        categories[2]: [0.5000, 0.4970, 0.4835, 0.4928, 0.4962],
        categories[3]: [0.8132, 0.8130, 0.8177, 0.8193, 0.8205],
    },
    "MUSIQ (↑)": {
        categories[0]: [61.4210, 60.8915, 60.8324, 62.7745, 61.1226],
        categories[1]: [63.3618, 61.2118, 63.5648, 63.2736, 63.3986],
        categories[2]: [61.5898, 61.4374, 59.8140, 61.3778, 60.7935],
        categories[3]: [63.9231, 63.5961, 63.9602, 64.0299, 64.2547],
    },
}

# Plot settings
# Desired ordering (left column then right column)
ordered_metrics = [
    "LPIPS (↓)",
    "PSNR (dB) (↑)",
    "SSIM (↑)",
    "MANIQA (↑)",
    "CLIPIQA (↑)",
    "MUSIQ (↑)",
]

num_metrics = len(ordered_metrics)
num_methods = len(methods)
num_categories = len(categories)

# Generate color map for methods
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(num_methods)]

fig, axes = plt.subplots(3, 2, figsize=(14, 14), sharey=False)
axes = axes.flatten()

# Custom y-axis limits to zoom in
y_limits = {
    "LPIPS (↓)": (0.1, 0.5),
    "PSNR (dB) (↑)": (55, 70),
    "SSIM (↑)": (0.3, 0.6),
    "MANIQA (↑)": (0.2, 0.6),
    "CLIPIQA (↑)": (0.2, 0.9),
    "MUSIQ (↑)": (40, 80),
}

bar_width = 0.12
bar_gap = 0.02  # gap between methods within a category

for ax, metric_name in zip(axes, ordered_metrics):
    cat_values = metric_data[metric_name]
    # Compute base positions for each category group
    indices = np.arange(num_categories)

    # To enable star annotation, keep track of bar positions and heights
    positions_by_method = []  # list of np.ndarray of positions per method
    values_by_method = []     # list of list of values per method

    for i_method, method in enumerate(methods):
        # Determine positions with offset for each method
        positions = indices - ((num_methods - 1) / 2.0) * (bar_width + bar_gap) + i_method * (bar_width + bar_gap)
        # Retrieve the values for this method across categories
        vals = [cat_values[cat][i_method] for cat in categories]
        ax.bar(positions, vals, width=bar_width, color=colors[i_method], label=label_map[method])

        positions_by_method.append(positions)
        values_by_method.append(vals)

    # After plotting all bars, add star markers on bars with max value per category
    for idx_cat in range(num_categories):
        # Gather (value, position) for each method for this category
        vals_positions = [
            (values_by_method[i][idx_cat], positions_by_method[i][idx_cat]) for i in range(num_methods)
        ]
        # Identify max value and its x-position
        max_val, max_pos = max(vals_positions, key=lambda vp: vp[0])
        # Plot star marker slightly above the bar top
        ax.plot(
            max_pos,
            max_val * 1.01,
            marker="*",
            markersize=8,
            color="black",
            zorder=5,
        )

    # Set x-ticks and labels to category names
    ax.set_xticks(indices)
    ax.set_xticklabels([cat.replace("_", "\n") for cat in categories], fontsize=9)
    ax.set_title(metric_name, fontsize=12, pad=10)
    # Apply custom y-axis limits
    if metric_name in y_limits:
        ax.set_ylim(y_limits[metric_name])
    ax.set_xlabel("Degradation Type", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate group names on top of each group
    '''ymax = ax.get_ylim()[1]
    for idx, cat in enumerate(categories):
        ax.text(idx, ymax * 1.02, f"{cat.split('_')[0]}", ha="center", va="bottom", fontsize=8, rotation=45)'''

    # Adjust y-axis label only for first subplot to save space
    if ax is axes[0]:
        ax.set_ylabel("Metric Value", fontsize=10)

# Create a single legend above all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

# Adjust layout to make room for legend on top
fig.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig("/home/krishna/workspace/AutoRestore/eval/ablation_exp/d3/grouped_bar_metrics.png", bbox_inches="tight")
plt.show()
