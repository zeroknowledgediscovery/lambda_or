#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================
# USER OPTIONS
# =============================
CSV_PATH = "./lambdaor_outputs/multisweep_summary.csv"

# Method colors (fill + edge will match)
COLOR_SPOR = "#ff8822"   # lambda-OR
COLOR_COR  = "#bbbbbb"   # naive OR

# Overall sizing
BASE_FONTSIZE = 15
FIGSIZE = (14.5, 11.5)

# Panel boundary styling
SPINE_COLOR = "#111111"
SPINE_LW = 2.0

# Panel title styling
TITLE_PAD = 6

# Panel letter styling (decoupled from title)
PANEL_LETTER_FONTSIZE = 24
PANEL_LETTER_X = -0.10
PANEL_LETTER_Y = 1.08

# Violin visibility
VIOLIN_WIDTH = 1.05
VIOLIN_GAP = 0.15

# Quartile line styling (constant)
QUARTILE_COLOR = "#eeeeee"
QUARTILE_ALPHA = 0.8
QUARTILE_LW = .5

# =============================
# GLOBAL STYLE
# =============================
mpl.rcParams.update({
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE + 2,
    "axes.labelsize": BASE_FONTSIZE + 1,
    "xtick.labelsize": BASE_FONTSIZE - 1,
    "ytick.labelsize": BASE_FONTSIZE - 1,
    "legend.fontsize": BASE_FONTSIZE,
})

# Use LaTeX so manuscript math renders the same
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = (
    "\\usepackage{amsmath}\n"
    "\\newcommand{\\SPOR}{\\textrm{$\\lambda$-OR}}\n"
    "\\newcommand{\\COR}{\\operatorname{OR}}\n"
)

# =============================
# LOAD + PREPARE DF
# =============================
df = pd.read_csv(CSV_PATH)

# derived axes used throughout the paper
df["e_p"] = 1.0 - df["p_sel"].astype(float)
df["e_q"] = 1.0 - df["q_sel"].astype(float)

# tick-label rounding (adjust if you want)
df["e_p"] = df["e_p"].round(3)
df["e_q"] = df["e_q"].round(3)
if "theta" in df.columns:
    df["theta"] = df["theta"].round(3)

spo_label = r"$\SPOR$"
naive_label = r"$\COR$"

label_map = {
    "lambda-OR": spo_label,
    "lambda-or": spo_label,
    "lambda_or": spo_label,
    "naive OR": naive_label,
    "naive": naive_label,
    "OR": naive_label,
}
df["method_label"] = df["method"].map(label_map).fillna(df["method"].astype(str))

# split=True requires exactly two hue levels
df = df[df["method_label"].isin([spo_label, naive_label])].copy()

# Ordered categorical x-axes (stable order)
def ordered_cat(series):
    vals = sorted(series.dropna().unique())
    return pd.Categorical(series, categories=vals, ordered=True)

df["e_p_cat"] = ordered_cat(df["e_p"])
df["e_q_cat"] = ordered_cat(df["e_q"])
if "theta" in df.columns:
    df["theta_cat"] = ordered_cat(df["theta"])

# n on x: categorical if not too many unique values
if "n" in df.columns and df["n"].nunique(dropna=True) <= 30:
    df["n_cat"] = ordered_cat(df["n"])
    n_x = "n_cat"
else:
    n_x = "n"

palette = {spo_label: COLOR_SPOR, naive_label: COLOR_COR}

# =============================
# AXIS LABELS (symbol + descriptive phrase)
# =============================
xlabel_map = {
    "e_p_cat": r"\textbf{$\mathbf{1-p_{\mathrm{sel}}}$ (False negative rate)}",
    "e_q_cat": r"\textbf{$\mathbf{1-q_{\mathrm{sel}}}$ (False positive rate)}",
    "theta_cat": r"\textbf{$\log \operatorname{OR}$}",
    "n_cat": r"\textbf{$\mathbf{n}$ (Sample size)}",
    "n": r"\textbf{$\mathbf{n}$ (Sample size)}",
}
ylabel_map = {
    "bias_mean": r"\textbf{$\langle \mathrm{bias} \rangle$}",
    "rmse": r"\textbf{$\mathrm{RMSE}$}",
    "cover_95": r"\textbf{$\mathrm{Coverage}_{95\%}$}",
}

# =============================
# PANEL TITLES (WITHOUT LETTERS)
# =============================
panel_titles_core = [
    r"\textbf{Bias vs. }$\mathbf{1-p_{\mathrm{sel}}}$",
    r"\textbf{RMSE vs. }$\mathbf{1-p_{\mathrm{sel}}}$",
    r"\textbf{Bias vs. }$\mathbf{1-q_{\mathrm{sel}}}$",
    r"\textbf{RMSE vs. }$\mathbf{1-q_{\mathrm{sel}}}$",
    r"\textbf{RMSE vs. }$\boldsymbol{\theta}$",
    r"\textbf{Bias vs. }$\boldsymbol{\theta}$",
    r"\textbf{Coverage vs. }$\boldsymbol{\theta}$",
    r"\textbf{Bias vs. }$\mathbf{n}$",
    r"\textbf{Coverage vs. }$\mathbf{1-q_{\mathrm{sel}}}$",
]

# =============================
# PLOT
# =============================
sns.set_style("white")  # no grid, like Fig 1

fig, axes = plt.subplots(3, 3, figsize=FIGSIZE)
axes = axes.flatten()

plots = [
    ("e_p_cat", "bias_mean"),
    ("e_p_cat", "rmse"),
    ("e_q_cat", "bias_mean"),
    ("e_q_cat", "rmse"),
    ("theta_cat", "rmse"),
    ("theta_cat", "bias_mean"),
    ("theta_cat", "cover_95"),
    (n_x, "bias_mean"),
    ("e_q_cat", "cover_95"),
]

for i, (xvar, yvar) in enumerate(plots):
    ax = axes[i]

    sns.violinplot(
        x=xvar,
        y=yvar,
        data=df,
        hue="method_label",
        split=True,
        inner="quart",
        gap=VIOLIN_GAP,
        density_norm="count",
        palette=palette,
        cut=0,
        width=VIOLIN_WIDTH,
        linewidth=1.0,
        ax=ax,
    )

    # Quartile line styling (constant gray, semi-transparent, thin)
    for line in ax.lines:
        line.set_color(QUARTILE_COLOR)
        line.set_alpha(QUARTILE_ALPHA)
        line.set_linewidth(QUARTILE_LW)

    # Title (no letter embedded)
    ax.set_title(panel_titles_core[i], pad=TITLE_PAD, fontweight="normal")

    # Panel letter (large, not bold, directly left of title)
    letter = chr(97 + i)
    ax.text(
        PANEL_LETTER_X,
        PANEL_LETTER_Y,
        f"{letter}.",
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_FONTSIZE,
        fontweight="normal",
        ha="left",
        va="center",
    )

    ax.grid(False)

    ax.set_xlabel(xlabel_map.get(xvar, xvar), fontweight="bold")
    ax.set_ylabel(ylabel_map.get(yvar, yvar), fontweight="bold")

    # Darker, bolder panel borders
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(SPINE_LW)

    # Violin edges match fill
    for coll in ax.collections:
        try:
            fc = coll.get_facecolor()
            if fc is None or len(fc) == 0:
                continue
            coll.set_edgecolor(fc)
            coll.set_linewidth(1.2)
        except Exception:
            pass

    if ax.legend_ is not None:
        ax.legend_.remove()

# Single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.965),
)

fig.subplots_adjust(top=0.88, wspace=0.32, hspace=0.55)

fig.savefig("simulation_violin_grid.pdf", bbox_inches="tight")
fig.savefig("simulation_violin_grid.png", dpi=300, bbox_inches="tight")
