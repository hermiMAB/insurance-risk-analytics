"""
visualization.py - EDA plots for ACIS Insurance Risk Analytics
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("reports/figures").mkdir(parents=True, exist_ok=True)

PALETTE = {
    "primary"  : "#2563EB",
    "accent"   : "#DC2626",
    "secondary": "#16A34A",
    "neutral"  : "#64748B",
}


def plot_univariate_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, color in [(axes[0], "TotalPremium", PALETTE["primary"]),
                            (axes[1], "TotalClaims",  PALETTE["accent"])]:
        data = df[col].dropna()

        # FIX: don't filter out zeros for TotalClaims — use >= 0
        data = data[(data >= 0) & (data <= data.quantile(0.95))]

        # FIX: guard against empty data
        if data.empty or data.nunique() < 2:
            ax.set_title(f"{col} — no data to plot", fontweight="bold")
            continue

        sns.histplot(data, bins=50, kde=True, color=color, ax=ax)
        ax.axvline(data.median(), color="black", linestyle="--",
                   label=f"Median  R{data.median():,.0f}")
        ax.set_title(col, fontsize=13, fontweight="bold", color=color)
        ax.set_xlabel("ZAR (capped at 95th pct)", fontsize=11)
        ax.set_ylabel("Policy count", fontsize=11)
        ax.legend(fontsize=9)
        ax.text(0.97, 0.95, f"skew = {data.skew():.2f}",
                transform=ax.transAxes, ha="right", va="top",
                color=PALETTE["neutral"])

    plt.suptitle("Premium & Claims Distributions", fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/01_univariate.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_province_risk(df):
    # FIX: create HasClaim on the fly if missing
    if "HasClaim" not in df.columns:
        df = df.copy()
        df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    prov = (
        df.groupby("Province", observed=True)["HasClaim"]
          .mean().mul(100).reset_index()
          .rename(columns={"HasClaim": "ClaimRate"})
          .sort_values("ClaimRate", ascending=False)
    )
    sns.barplot(x="ClaimRate", y="Province", data=prov,
                palette="Blues_r", ax=ax1)
    ax1.set_title("Claim Rate (%) by Province", fontsize=13,
                  fontweight="bold", color=PALETTE["primary"])
    ax1.set_xlabel("Claim Rate (%)", fontsize=11)
    ax1.set_ylabel("Province", fontsize=11)

    veh = (
        df.groupby("VehicleType", observed=True)["HasClaim"]
          .mean().mul(100).reset_index()
          .rename(columns={"HasClaim": "ClaimRate"})
          .sort_values("ClaimRate", ascending=False)
    )
    sns.barplot(x="ClaimRate", y="VehicleType", data=veh,
                palette="Oranges_r", ax=ax2)
    ax2.set_title("Claim Rate (%) by Vehicle Type", fontsize=13,
                  fontweight="bold", color=PALETTE["primary"])
    ax2.set_xlabel("Claim Rate (%)", fontsize=11)
    ax2.set_ylabel("Vehicle Type", fontsize=11)

    plt.tight_layout()
    plt.savefig("reports/figures/03_province_risk.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_premium_vs_claims(df):
    """Scatter by Province + top postal codes by Loss Ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # scatter by province
    cap = df[["TotalPremium", "TotalClaims"]].quantile(0.99).max()
    d   = df[(df["TotalPremium"] > 0) & (df["TotalPremium"] <= cap) &
             (df["TotalClaims"] >= 0)  & (df["TotalClaims"]  <= cap)]

    provinces = d["Province"].dropna().unique()
    palette   = dict(zip(provinces, sns.color_palette("tab10", len(provinces))))
    for prov, grp in d.groupby("Province", observed=True):
        ax1.scatter(grp["TotalPremium"], grp["TotalClaims"],
                    label=str(prov), color=palette[prov],
                    alpha=0.3, s=8, linewidths=0)
    ax1.plot([0, cap], [0, cap], color="black", linestyle="--",
             linewidth=1.2, label="Break-even")
    ax1.set_title("Premium vs Claims by Province", fontsize=13,
                  fontweight="bold", color=PALETTE["primary"])
    ax1.set_xlabel("Total Premium (ZAR)", fontsize=11)
    ax1.set_ylabel("Total Claims (ZAR)", fontsize=11)
    ax1.legend(fontsize=7, markerscale=3)

    # top postal codes by loss ratio
    zip_lr = (
        df.groupby("PostalCode", observed=True)
          .agg(P=("TotalPremium", "sum"),
               C=("TotalClaims",  "sum"),
               N=("PolicyID",     "count"))
          .query("N >= 30")
          .assign(LR=lambda x: x.C / x.P)
          .nlargest(10, "LR")
          .reset_index()
    )
    colors = [PALETTE["accent"] if v >= 1 else PALETTE["primary"]
              for v in zip_lr["LR"]]
    ax2.barh(zip_lr["PostalCode"].astype(str), zip_lr["LR"],
             color=colors, edgecolor="white")
    ax2.axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    ax2.set_title("Top 10 Postal Codes — Loss Ratio", fontsize=13,
                  fontweight="bold", color=PALETTE["primary"])
    ax2.set_xlabel("Loss Ratio", fontsize=11)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig("reports/figures/02_premium_vs_claims.png", dpi=150, bbox_inches="tight")
    plt.show()





def plot_outliers(df):
    """Box plots for the 4 key financial columns."""
    cols = [c for c in ["TotalPremium", "TotalClaims",
                         "SumInsured", "CalculatedPremiumPerTerm"]
            if c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, col in zip(axes.flatten(), cols):
        data   = df[col].dropna().astype(float)
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        n_out  = int(((data < q1 - 1.5*(q3-q1)) | (data > q3 + 1.5*(q3-q1))).sum())
        ax.boxplot(data, vert=False, patch_artist=True,
                   boxprops    = dict(facecolor=PALETTE["primary"], alpha=0.5),
                   medianprops = dict(color=PALETTE["accent"], linewidth=2),
                   flierprops  = dict(marker=".", markersize=2, alpha=0.2))
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_yticks([])
        ax.text(0.97, 0.8,
                f"{n_out:,} outliers ({n_out/len(data)*100:.1f}%)",
                transform=ax.transAxes, ha="right",
                fontsize=9, color=PALETTE["accent"])

    plt.suptitle("Outlier Detection — Key Financial Columns", fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/04_outliers.png", dpi=150, bbox_inches="tight")
    plt.show()