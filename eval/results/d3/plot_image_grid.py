import os
from typing import List
import matplotlib.pyplot as plt
from PIL import Image

# Absolute base paths (modify here if directory structure changes)
BASE_DIR = "/home/krishna/workspace/AutoRestore"
DATA_DIR = os.path.join(BASE_DIR, "eval/data/d3")
GT_DIR = os.path.join(BASE_DIR, "eval/gnd_truth")

# Rows: degradation scenarios
DEGRADATION_FOLDERS: List[str] = [
    "dark_defocus_blur_jpeg compression_artifact",
    "haze_motion_blur_low_resolution",
    "motion_blur_defocus_blur_noise",
    "rain_noise_low_resolution",
]

RAW_METHOD_FOLDER = "planned_no_optimisation"  # folder where raw images live

# Columns: methods (internal folder names)
METHOD_MAP = {
    "raw input": RAW_METHOD_FOLDER,
    "planning only": "planned_no_optimisation",
    "planning + optimization": "planned_opt_no_verify",
    "planning + optimization + verify": "lambda_0_7",
    "finetuned_planning + optimization + verify": "finetuned_planned_opt_verify",
    "random planning": "random_planning",
}

# Target subplot grid size
N_ROWS = len(DEGRADATION_FOLDERS)
N_COLS = len(METHOD_MAP)


def _load_and_resize(path: str, size):
    """Load image from path and resize keeping aspect ratio via thumbnail."""
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    # Paste into exact-size canvas for consistent shape
    canvas = Image.new("RGB", size, (255, 255, 255))
    x_off = (size[0] - img.width) // 2
    y_off = (size[1] - img.height) // 2
    canvas.paste(img, (x_off, y_off))
    return canvas


def plot_image_grid(img_name: str, resize_to=(256, 256)):
    """Plot a 4×5 grid of processed images for the given file name.

    Parameters
    ----------
    img_name : str
        Name of the image file (e.g., "001.png").
    resize_to : tuple[int, int], optional
        Desired unified size (width, height) of each plotted image, by default (256, 256).
    """
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 3, N_ROWS * 3))

    col_titles = list(METHOD_MAP.keys())

    # Utility to split long titles onto two lines (skip for 'raw input')
    _format_title = lambda t: t if t == "raw input" else t.replace(" + ", "\n+ ")

    for col_idx, col_title in enumerate(col_titles):
        axes[0, col_idx].set_title(
            _format_title(col_title),
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

    for row_idx, degradation in enumerate(DEGRADATION_FOLDERS):
        row_label = degradation.replace("_", " ")  # human readable
        axes[row_idx, 0].set_ylabel(
            row_label,
            fontsize=12,
            fontweight="bold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=130,
        )

        for col_idx, col_title in enumerate(col_titles):
            ax = axes[row_idx, col_idx]
            method_folder = METHOD_MAP[col_title]
            subdir = "raw" if col_title == "raw input" else "processed"
            img_path = os.path.join(
                DATA_DIR,
                degradation,
                method_folder,
                subdir,
                img_name,
            )
            # If processed not found fall back to raw (for processed cols)
            if not os.path.exists(img_path) and subdir == "processed":
                img_path = os.path.join(
                    DATA_DIR,
                    degradation,
                    method_folder,
                    "raw",
                    img_name,
                )
            try:
                img = _load_and_resize(img_path, resize_to)
                ax.imshow(img)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    fig.subplots_adjust(left=0.25, top=0.88)  # extra space for row labels and titles
    return fig


if __name__ == "__main__":
    # Example usage
    fig = plot_image_grid("001.png")
    save_path = os.path.join(BASE_DIR, "eval/ablation_exp/d3/ablation_exp_d3_image_grid.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
