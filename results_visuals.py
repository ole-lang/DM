import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")


# ------------------------------
# Data loading
# ------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(df.head())
    print(df.describe())
    return df


# ------------------------------
# Plots
# ------------------------------
def plot_corr_matrix(df: pd.DataFrame, num_cols):
    """Correlation matrix + heatmap."""
    corr = df[num_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_r2_distribution(df: pd.DataFrame, r2_cols):
    """Boxplot + points for R² per model."""
    df_r2 = df.melt(
        id_vars=["file"],
        value_vars=r2_cols,
        var_name="model",
        value_name="r2",
    )

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_r2, x="model", y="r2")
    sns.stripplot(
        data=df_r2,
        x="model",
        y="r2",
        color="black",
        alpha=0.4,
        jitter=True,
    )
    plt.title("Distribution of R² per Model")
    plt.xlabel("Model")
    plt.ylabel("R²")
    plt.tight_layout()
    plt.show()
    
def plot_model_r2_heatmap(df):
    """
    Plots a heatmap showing the R² and aggregated R² scores
    for each model across all files.
    Removes values outside [-1, 1] to avoid scale distortion.
    """
    models = ["linear", "rf", "ada", "mlp"]
    
    heatmap_data = []
    file_labels = []

    for _, row in df.iterrows():
        row_values = []
        for m in models:
            val_r2 = row[f"{m}_r2"]
            val_agg = row[f"{m}_aggregated_r2"]

            # Cleanup: replace invalid values with NaN
            val_r2 = val_r2 if 0 <= val_r2 <= 1 else np.nan
            val_agg = val_agg if 0 <= val_agg <= 1 else np.nan
            
            row_values.append(val_r2)
            row_values.append(val_agg)

        heatmap_data.append(row_values)
        file_labels.append(row.get("file", f"File {_}"))

    heatmap_data = np.array(heatmap_data)

    # X-axis labels
    x_labels = []
    for m in models:
        x_labels.append(f"{m.capitalize()} R²")
        x_labels.append(f"{m.capitalize()} Aggregated R²")

    # ---- Plot ----
    plt.figure(figsize=(14, max(6, len(file_labels) * 0.3)))
    im = plt.imshow(heatmap_data, aspect="auto", cmap="viridis")

    plt.colorbar(im, label="R² Score")

    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(file_labels)), file_labels, fontsize=6)

    plt.title("Heatmap of Normal and Aggregated R² Scores per File")
    plt.tight_layout()
    plt.show()




def plot_aggregated_r2_distribution(df: pd.DataFrame, agg_cols):
    """Boxplot + points for aggregated R² per model."""
    df_r2_agg = df.melt(
        id_vars=["file"],
        value_vars=agg_cols,
        var_name="model",
        value_name="aggregated_r2",
    )

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_r2_agg, x="model", y="aggregated_r2")
    sns.stripplot(
        data=df_r2_agg,
        x="model",
        y="aggregated_r2",
        color="black",
        alpha=0.4,
        jitter=True,
    )
    plt.title("Distribution of Aggregated R² per Model")
    plt.xlabel("Model")
    plt.ylabel("Aggregated R²")
    plt.tight_layout()
    plt.show()


def plot_r2_vs_num_rows(df: pd.DataFrame, r2_cols, labels):
    """Subplots: R² vs number of rows for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, col, label in zip(axes, r2_cols, labels):
        ax.scatter(df["num_rows"], df[col], alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("Number of rows (num_rows)")
        ax.set_ylabel("R²")

    fig.suptitle("R² vs Number of Rows for Each Model", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_mean_r2_per_rows_bin(df: pd.DataFrame, r2_cols, labels):
    """Line chart of mean R² per num_rows-bin."""
    bins = np.linspace(df["num_rows"].min(), df["num_rows"].max(), 8)
    labels_bins = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)]

    df_binned = df.copy()
    df_binned["rows_bin"] = pd.cut(
        df_binned["num_rows"],
        bins=bins,
        labels=labels_bins,
        include_lowest=True,
    )

    mean_r2_per_bin = df_binned.groupby("rows_bin")[r2_cols].mean()

    plt.figure(figsize=(10, 6))
    for col, label in zip(r2_cols, labels):
        plt.plot(
            mean_r2_per_bin.index.astype(str),
            mean_r2_per_bin[col],
            marker="o",
            label=label,
        )

    plt.title("Mean R² per Number-of-rows Bin")
    plt.xlabel("Number-of-rows bin")
    plt.ylabel("Mean R²")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_mean_r2_custom_ranges(df: pd.DataFrame, r2_cols, labels):
    """Mean R² for custom row intervals."""
    
    bins = [0, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000]

    labels_bins = [
        "0–5000 rows",       # 1
        "5000–7500 rows",    # 2
        "7500–10000 rows",   # 3
        "10000–12500 rows",  # 4
        "12500–15000 rows",  # 5
        "15000–17500 rows",  # 6
        "17500–20000 rows",  # 7
        "20000–22500 rows",  # 8
        "22500–25000 rows"   # 9  ← exakt rätt antal
    ]
    
    df_copy = df.copy()
    df_copy["rows_range"] = pd.cut(
        df_copy["num_rows"],
        bins=bins,
        labels=labels_bins,
        include_lowest=True
    )

    mean_r2_per_range = df_copy.groupby("rows_range")[r2_cols].mean()

    plt.figure(figsize=(12, 6))
    for col, label in zip(r2_cols, labels):
        plt.plot(
            mean_r2_per_range.index.astype(str),
            mean_r2_per_range[col],
            marker="o",
            label=label
        )

    plt.title("Mean R² per Defined Row Range")
    plt.xlabel("Row Range")
    plt.ylabel("Mean R²")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_r2_vs_mean_speed(df: pd.DataFrame, r2_cols, labels):
    """Subplots: R² vs mean_speed for each model."""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()  # flatten 2x2 grid

    for ax, col, label in zip(axes, r2_cols, labels):
        ax.scatter(df["mean_speed"], df[col], alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("Mean Speed (mean_speed)")
        ax.set_ylabel("R²")

    fig.suptitle("R² vs Mean Speed for Each Model", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for the main title
    plt.show()


def plot_r2_vs_std_speed(df: pd.DataFrame, r2_cols, labels):
    """Subplots: R² vs std_speed for each model."""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()  # flatten 2x2 grid

    for ax, col, label in zip(axes, r2_cols, labels):
        ax.scatter(df["std_speed"], df[col], alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("Speed Standard Deviation (std_speed)")
        ax.set_ylabel("R²")

    fig.suptitle("R² vs Speed Variation (std_speed) for Each Model", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for main title
    plt.show()

# ------------------------------
# Main
# ------------------------------
def main():
    csv_path = "Results_with_acc.csv"
    df = load_data(csv_path)

    num_cols = [
        "linear_r2",
        "rf_r2",
        "ada_r2",
        "mlp_r2",
        "linear_aggregated_r2",
        "rf_aggregated_r2",
        "ada_aggregated_r2",
        "mlp_aggregated_r2",
        "num_rows",
        "mean_speed",
        "std_speed",
    ]

    r2_cols = ["linear_r2", "rf_r2", "ada_r2", "mlp_r2"]
    agg_cols = [
        "linear_aggregated_r2",
        "rf_aggregated_r2",
        "ada_aggregated_r2",
        "mlp_aggregated_r2",
    ]
    labels = ["Linear", "Random Forest", "AdaBoost", "MLP"]

    # plot_corr_matrix(df, num_cols)
    # plot_r2_distribution(df, r2_cols)
    # plot_aggregated_r2_distribution(df, agg_cols)
    # plot_r2_vs_num_rows(df, r2_cols, labels)
    # plot_mean_r2_per_rows_bin(df, r2_cols, labels)
    # plot_mean_r2_custom_ranges(df,r2_cols,labels)
    # plot_r2_vs_mean_speed(df, r2_cols, labels)
    # plot_r2_vs_std_speed(df, r2_cols, labels)
    plot_model_r2_heatmap(df)


if __name__ == "__main__":
    main()
