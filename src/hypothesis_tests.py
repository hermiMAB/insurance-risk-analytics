import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, chi2_contingency
from itertools import combinations

def prep_insurance_data(df):
    """
    Prepares the raw insurance dataframe by calculating Margin, 
    Claim Frequency flags, and creating a claims-only subset.
    """
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
    
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    df['Has_Claim'] = np.where(df['TotalClaims'] > 0, 'Yes', 'No')
    
    claims_only_df = df[df['TotalClaims'] > 0].copy()
    
    return df, claims_only_df

# =====================================================================
# 1. CHI-SQUARE TEST (Frequency / Categorical Independence)
# =====================================================================
def test_chi_square(df, category_col, target_col='Has_Claim', alpha=0.05):
    """
    Runs a Chi-Square test with Cramér's V Effect Size Interpretation.
    """
    contingency = pd.crosstab(df[category_col], df[target_col])
    chi2_stat, p_value_chi2, dof, expected = chi2_contingency(contingency)

    print('======================================================')
    print(f'     CHI-SQUARED TEST: {category_col} vs {target_col}')
    print('======================================================')
    print(f'H₀: {category_col} and {target_col} are INDEPENDENT')
    print(f'H₁: There IS an association between them')
    print('---')
    print(f'Chi² statistic:     {chi2_stat:.4f}')
    print(f'Degrees of freedom: {dof}')
    print(f'P-value:            {p_value_chi2:.6f}')
    print('---')
    
    if p_value_chi2 < alpha:
        print(f'DECISION: p ({p_value_chi2:.6f}) < α ({alpha}) → REJECT H₀')
        print(f'There IS a significant association between {category_col} and {target_col}.\n')
        
        # --- Effect Size: Cramér's V ---
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2_stat / (n * min_dim))
        
        print('=== Effect Size (Cramér\'s V) ===')
        print(f"Value: {cramers_v:.4f}")
        
        if cramers_v < 0.1:
            print("Interpretation: NEGLIGIBLE association")
        elif cramers_v < 0.3:
            print("Interpretation: SMALL association")
        elif cramers_v < 0.5:
            print("Interpretation: MEDIUM association")
        else:
            print("Interpretation: LARGE association")
            
    else:
        print(f'DECISION: p ({p_value_chi2:.6f}) ≥ α ({alpha}) → FAIL TO REJECT H₀')
        print('No significant association found.')

    print('\n=== Expected Frequencies (under H₀) ===')
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    print(expected_df.round(1))
    
    min_expected = expected.min()
    if min_expected >= 5:
        print('\n(All expected counts ≥ 5 → Chi-squared assumption satisfied)')
    else:
        print(f'\n(WARNING: Minimum expected count is {min_expected:.1f}. Assumption violated!)')
        
    return p_value_chi2


# =====================================================================
# 2. T-TEST (Severity / 2 Groups)
# =====================================================================
def test_ttest(df, category_col, numeric_col, group1_name, group2_name, alpha=0.05):
    """
    Runs an Independent Samples T-Test with Cohen's d and Percentage Diff.
    """
    g1_data = df[df[category_col] == group1_name][numeric_col].dropna()
    g2_data = df[df[category_col] == group2_name][numeric_col].dropna()
    
    print('======================================================')
    print(f'  INDEPENDENT SAMPLES T-TEST: {category_col} ({group1_name} vs {group2_name})')
    print('======================================================')
    
    print('\n=== Group Sizes ===')
    print(f"{group1_name}: {len(g1_data):,}")
    print(f"{group2_name}: {len(g2_data):,}\n")

    print('=== Descriptive Statistics ===')
    print(f"{group1_name:<10} — Mean: {g1_data.mean():.2f}, Median: {g1_data.median():.2f}, Std: {g1_data.std():.2f}")
    print(f"{group2_name:<10} — Mean: {g2_data.mean():.2f}, Median: {g2_data.median():.2f}, Std: {g2_data.std():.2f}\n")

    t_stat, p_value = stats.ttest_ind(g1_data, g2_data, equal_var=False)

    print('H₀: μ_1 = μ_2 (Means are equal)')
    print('H₁: μ_1 ≠ μ_2 (Means are different)')
    print('---')
    print(f'T-statistic: {t_stat:.4f}')
    print(f'P-value:     {p_value:.6f}')
    print(f'Alpha:       {alpha}')
    print('---')
    
    if p_value < alpha:
        print(f'DECISION: p ({p_value:.6f}) < α ({alpha}) → REJECT H₀')
        print(f'There IS a statistically significant difference in average {numeric_col}.\n')
    else:
        print(f'DECISION: p ({p_value:.6f}) ≥ α ({alpha}) → FAIL TO REJECT H₀')
        print(f'No significant difference in average {numeric_col}.\n')

    # --- Effect Size Calculation ---
    pooled_std = np.sqrt((g1_data.std()**2 + g2_data.std()**2) / 2)
    cohens_d = (g2_data.mean() - g1_data.mean()) / pooled_std
    mean_diff = g2_data.mean() - g1_data.mean()
    pct_diff = (mean_diff / g1_data.mean()) * 100 if g1_data.mean() != 0 else 0

    print('=== Effect Size ===')
    direction = f"{group2_name} higher" if mean_diff > 0 else f"{group1_name} higher"
    print(f"Mean difference:    {abs(mean_diff):.2f} ({direction})")
    print(f"Percentage diff:    {abs(pct_diff):.1f}% higher for {group2_name if mean_diff > 1 else group1_name}")
    print(f"Cohen's d:          {abs(cohens_d):.3f}")
    
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        print("Interpretation: SMALL effect size")
    elif abs_d < 0.5:
        print("Interpretation: SMALL-MEDIUM effect size")
    elif abs_d < 0.8:
        print("Interpretation: MEDIUM effect size")
    else:
        print("Interpretation: LARGE effect size")
        
    return p_value


# =====================================================================
# 3. ANOVA (Severity / 3+ Groups)
# =====================================================================
def test_anova(df, category_col, numeric_col, alpha=0.05):
    """
    Runs a One-Way ANOVA with Eta-Squared and Pairwise % Differences.
    """
    grouped = df.dropna(subset=[category_col, numeric_col]).groupby(category_col)
    region_groups = {name: group[numeric_col].values for name, group in grouped if len(group) > 0}
    groups_list = list(region_groups.values())

    print('======================================================')
    print(f'     ONE-WAY ANOVA: {numeric_col} across {category_col}')
    print('======================================================')

    print('\n=== Group Sizes ===')
    for name, data in region_groups.items():
        print(f"{name}: {len(data):,}")

    print('\n=== Descriptive Statistics ===')
    for name, data in region_groups.items():
        print(f"{str(name):<20} — Mean: {np.mean(data):.2f}, Median: {np.median(data):.2f}, Std: {np.std(data, ddof=1):.2f}")

    f_stat, p_value = f_oneway(*groups_list)

    print('\nH₀: Average is equal across all groups')
    print('H₁: At least one group mean is different')
    print('---')
    print(f'F-statistic: {f_stat:.4f}')
    print(f'P-value:     {p_value:.6f}')
    print(f'Alpha:       {alpha}')
    print('---')

    # --- Effect Size: Eta-Squared (η²) ---
    k = len(groups_list)
    N = sum(len(g) for g in groups_list)
    df_bn = k - 1
    df_wn = N - k
    eta_squared = (f_stat * df_bn) / ((f_stat * df_bn) + df_wn)

    print('\n=== Overall Effect Size (Eta-Squared η²) ===')
    print(f"Value: {eta_squared:.4f}")
    if eta_squared < 0.01:
        print("Interpretation: NEGLIGIBLE effect")
    elif eta_squared < 0.06:
        print("Interpretation: SMALL effect")
    elif eta_squared < 0.14:
        print("Interpretation: MEDIUM effect")
    else:
        print("Interpretation: LARGE effect")

    if p_value < alpha:
        print(f'\nDECISION: p ({p_value:.6f}) < α ({alpha}) → REJECT H₀')
        print(f'There IS a statistically significant difference in {numeric_col} across {category_col}.\n')
        
        # --- Post-hoc Test: Bonferroni Corrected Pairwise T-Tests ---
        pairs = list(combinations(region_groups.keys(), 2))
        
        print('=== Pairwise T-Tests (Bonferroni corrected) ===')
        print(f"Number of comparisons: {len(pairs)}")
        bonferroni_alpha = alpha / len(pairs)
        print(f"Corrected alpha: {bonferroni_alpha:.6f}\n")
        
        print(f"{'Pair':<35} {'Mean Diff':>12} {'% Diff':>10} {'p-value':>12} {'Significant?':>14}")
        print('-' * 88)

        for g1, g2 in pairs:
            _, pv = stats.ttest_ind(region_groups[g1], region_groups[g2], equal_var=False)
            
            mean_g1 = np.mean(region_groups[g1])
            mean_g2 = np.mean(region_groups[g2])
            mean_diff = mean_g2 - mean_g1
            
            # Calculate percentage difference relative to Group 1
            pct_diff = (mean_diff / mean_g1) * 100 if mean_g1 != 0 else 0
            
            sig = 'YES ***' if pv < bonferroni_alpha else ('YES *' if pv < alpha else 'No')
            pair_name = f"{str(g1)[:15]} vs {str(g2)[:15]}"
            
            print(f"{pair_name:<35} {mean_diff:>+12.2f} {pct_diff:>+9.1f}% {pv:>12.6f} {sig:>14}")

    else:
        print(f'\nDECISION: p ({p_value:.6f}) ≥ α ({alpha}) → FAIL TO REJECT H₀')
        print(f'No significant difference found. No post-hoc tests needed.\n')

    return p_value