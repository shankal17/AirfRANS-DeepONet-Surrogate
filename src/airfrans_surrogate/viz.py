"""Shared figure style: thin marks, recessive axes, fixed categorical order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots in fixed order (never cycled or re-ranked).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def setup():
    """Apply the shared style to matplotlib's global rcParams."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8,
    })


def naca_label(row):
    """Readable airfoil name from a cases.csv row.

    Args:
        row (Series): cases.csv row with "naca" (parameters joined by "_") and "series".

    Returns:
        str: e.g. "NACA 4-digit (3.1, 1, 18.252)".
    """
    
    params = str(row.naca).split("_")
    
    return f"NACA {'4' if row.series == 4 else '5'}-digit ({', '.join(params)})"
