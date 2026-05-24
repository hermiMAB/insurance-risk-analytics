import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define your brand palette to keep charts consistent
PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#d62728',
    'background': '#f4f4f4'
}

def prepare_plotting_data(df: pd.DataFrame):
    """
    Cleans and prepares dataframes specifically for visualizations.
    Returns: df_clean, df_plot, zip_summary
    """
    print("Preparing data for visualizations...")
    
    # A. Filter Operational Anomalies (Keep active policies and valid claims)
    df_clean = df[(df['TotalPremium'] > 0) & (df['TotalClaims'] >= 0)].copy()

    # B. Create Capped Dataset for Distribution Plots (Remove top 1% skew)
    premium_cap = df_clean['TotalPremium'].quantile(0.99)
    df_plot = df_clean[df_clean['TotalPremium'] <= premium_cap].copy()

    # C. High-Cardinality Aggregation (Zip Code Summary for Scatter Plot)
    zip_summary = df_clean.groupby('PostalCode').agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum'),
        PolicyCount=('PolicyID', 'count')
    ).reset_index()

    # Filter out noise (postal codes with fewer than 10 policies)
    zip_summary = zip_summary[zip_summary['PolicyCount'] >= 10]

    print("✅ Preprocessing complete. Dataframes ready for plotting.")
    return df_clean, df_plot, zip_summary


def plot_univariate_distributions(df: pd.DataFrame):
    """Plots histograms for numerical and bar charts for categorical columns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Numerical: Total Premium
    sns.histplot(df['TotalPremium'], bins=40, color=PALETTE['primary'], ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Total Premium', fontweight='bold')
    
    # 2. Numerical: Total Claims (Excluding 0s and extreme outliers)
    positive_claims = df[df['TotalClaims'] > 0]
    # Dynamically cap at the 95th percentile of actual claims so the plot isn't squished
    claims_cap = positive_claims['TotalClaims'].quantile(0.95)
    claims_to_plot = positive_claims[positive_claims['TotalClaims'] <= claims_cap]
    
    sns.histplot(claims_to_plot['TotalClaims'], bins=40, color=PALETTE['accent'], ax=axes[0, 1])
    axes[0, 1].set_title(f'Distribution of Claims (Values > $0 and <= ${claims_cap:,.0f})', fontweight='bold')
    
    # 3. Categorical: Top 10 Vehicle Types
    top_vehicles = df['VehicleType'].value_counts().nlargest(10).index
    sns.countplot(y='VehicleType', data=df, order=top_vehicles, palette='Blues_r', ax=axes[1, 0])
    axes[1, 0].set_title('Top Vehicle Types', fontweight='bold')
    axes[1, 0].set_ylabel('')
    
    # 4. Categorical: Top 10 Cover Types
    top_covers = df['CoverType'].value_counts().nlargest(10).index
    sns.countplot(y='CoverType', data=df, order=top_covers, palette='Oranges_r', ax=axes[1, 1])
    axes[1, 1].set_title('Top Cover Types', fontweight='bold')
    axes[1, 1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()

def plot_premium_vs_claims(zip_summary: pd.DataFrame):
    """Plots scatter relationships and prints correlation for Zip Codes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Scatter Plot (Now with Log Scales)
    sns.scatterplot(
        x='TotalPremium', 
        y='TotalClaims', 
        size='PolicyCount', 
        sizes=(20, 400),
        alpha=0.6, 
        color=PALETTE['primary'],
        data=zip_summary, 
        ax=axes[0]
    )
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title('Total Premium vs Total Claims by Postal Code (Log Scale)', fontweight='bold')
    axes[0].set_xlabel('Total Premium (Aggregated)')
    axes[0].set_ylabel('Total Claims (Aggregated)')
    
    # 2. Correlation Matrix
    corr_matrix = zip_summary[['TotalPremium', 'TotalClaims']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('Correlation Matrix (ZipCode Level)', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_province_risk(df: pd.DataFrame):
    """Compares premium, auto make, and cover type across provinces using stacked bars."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    # 1. Average Premium by Province
    prov_prem = df.groupby('Province')['TotalPremium'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(x='TotalPremium', y='Province', data=prov_prem, palette='Blues_r', ax=axes[0])
    axes[0].set_title('Average Premium by Province', fontweight='bold')
    axes[0].set_ylabel('')
    
    # 2. Top Auto Make per Province (Stacked Bar Chart)
    top_makes = df['make'].value_counts().nlargest(5).index
    make_prov = pd.crosstab(df['Province'], df[df['make'].isin(top_makes)]['make'], normalize='index')
    make_prov.plot(kind='bar', stacked=True, colormap='YlGnBu', ax=axes[1])
    axes[1].set_title('Top 5 Auto Makes Distribution by Province', fontweight='bold')
    axes[1].set_ylabel('Proportion')
    axes[1].set_xlabel('')
    axes[1].legend(title='Make', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Top Cover Type per Province (Stacked Bar Chart)
    top_covers = df['CoverType'].value_counts().nlargest(5).index
    cover_prov = pd.crosstab(df['Province'], df[df['CoverType'].isin(top_covers)]['CoverType'], normalize='index')
    cover_prov.plot(kind='bar', stacked=True, colormap='Oranges', ax=axes[2])
    axes[2].set_title('Top 5 Cover Types Distribution by Province', fontweight='bold')
    axes[2].set_ylabel('Proportion')
    axes[2].set_xlabel('')
    axes[2].legend(title='Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_outliers(df: pd.DataFrame):
    """Uses box plots to detect outliers on key numerical features."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Log scale is used here because insurance outliers are extreme.
    
    sns.boxplot(x=df['TotalPremium'], color=PALETTE['primary'], ax=axes[0])
    axes[0].set_xscale('log')
    axes[0].set_title('Outliers: Total Premium (Log Scale)', fontweight='bold')
    
    # Only plot claims > 0 to avoid log(0) errors
    claims_only = df[df['TotalClaims'] > 0]
    sns.boxplot(x=claims_only['TotalClaims'], color=PALETTE['accent'], ax=axes[1])
    axes[1].set_xscale('log')
    axes[1].set_title('Outliers: Total Claims (Log Scale)', fontweight='bold')
    
    sns.boxplot(x=df['CustomValueEstimate'], color=PALETTE['secondary'], ax=axes[2])
    axes[2].set_xscale('log')
    axes[2].set_title('Outliers: Custom Value Estimate (Log Scale)', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
def plot_loss_ratio(df: pd.DataFrame):
    """Calculates and plots the Loss Ratio by Province, VehicleType, and Gender."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # Helper function to calculate aggregate loss ratio
    def get_loss_ratio(group_col):
        grouped = df.groupby(group_col)[['TotalPremium', 'TotalClaims']].sum()
        # Loss Ratio = (Total Claims / Total Premium) * 100
        grouped['LossRatio'] = (grouped['TotalClaims'] / grouped['TotalPremium']) * 100
        return grouped['LossRatio'].sort_values(ascending=False).reset_index()

    # 1. By Province
    lr_prov = get_loss_ratio('Province')
    sns.barplot(x='LossRatio', y='Province', data=lr_prov, palette='Blues_r', ax=axes[0])
    axes[0].set_title('Loss Ratio (%) by Province', fontweight='bold')
    axes[0].set_xlabel('Loss Ratio (%)')
    axes[0].set_ylabel('')

    # 2. By Vehicle Type
    lr_veh = get_loss_ratio('VehicleType')
    sns.barplot(x='LossRatio', y='VehicleType', data=lr_veh, palette='Oranges_r', ax=axes[1])
    axes[1].set_title('Loss Ratio (%) by Vehicle Type', fontweight='bold')
    axes[1].set_xlabel('Loss Ratio (%)')
    axes[1].set_ylabel('')

    # 3. By Gender
    lr_gen = get_loss_ratio('Gender')
    sns.barplot(x='Gender', y='LossRatio', data=lr_gen, palette='Purples_r', ax=axes[2])
    axes[2].set_title('Loss Ratio (%) by Gender', fontweight='bold')
    axes[2].set_ylabel('Loss Ratio (%)')
    axes[2].set_xlabel('')

    plt.tight_layout()
    plt.show()

def plot_temporal_trends(df: pd.DataFrame):
    """Plots claim frequency and severity over time."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Ensure TransactionMonth is datetime
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    
    # 1. Claim Frequency (Percentage of policies with claims > 0 per month)
    monthly_counts = df.groupby('TransactionMonth').size()
    monthly_claims = df[df['TotalClaims'] > 0].groupby('TransactionMonth').size()
    frequency = (monthly_claims / monthly_counts * 100).fillna(0)
    
    axes[0].plot(frequency.index, frequency.values, marker='o', color=PALETTE['primary'], linewidth=2)
    axes[0].set_title('Claim Frequency Over Time', fontweight='bold')
    axes[0].set_ylabel('Claim Frequency (%)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Claim Severity (Average claim amount for claims > 0 per month)
    severity = df[df['TotalClaims'] > 0].groupby('TransactionMonth')['TotalClaims'].mean()
    
    axes[1].plot(severity.index, severity.values, marker='s', color=PALETTE['accent'], linewidth=2)
    axes[1].set_title('Average Claim Severity Over Time', fontweight='bold')
    axes[1].set_ylabel('Average Claim Amount ($)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_vehicle_claims(df: pd.DataFrame):
    """Plots Top 10 and Bottom 10 Vehicle Makes by Average Claim Severity."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Filter for actual claims, then group by make
    actual_claims = df[df['TotalClaims'] > 0]
    
    # Filter out makes with very few claims to avoid noisy data (e.g., 1 claim for $100k)
    claim_counts = actual_claims['make'].value_counts()
    valid_makes = claim_counts[claim_counts >= 10].index
    filtered_claims = actual_claims[actual_claims['make'].isin(valid_makes)]
    
    make_severity = filtered_claims.groupby('make')['TotalClaims'].mean().sort_values(ascending=False)
    
    # 1. Top 10 Highest Severity
    sns.barplot(x=make_severity.head(10).values, y=make_severity.head(10).index, palette='Reds_r', ax=axes[0])
    axes[0].set_title('Top 10 Vehicle Makes (Highest Avg Claim)', fontweight='bold')
    axes[0].set_xlabel('Average Claim Amount ($)')
    
    # 2. Bottom 10 Lowest Severity
    sns.barplot(x=make_severity.tail(10).values, y=make_severity.tail(10).index, palette='Greens_r', ax=axes[1])
    axes[1].set_title('Top 10 Vehicle Makes (Lowest Avg Claim)', fontweight='bold')
    axes[1].set_xlabel('Average Claim Amount ($)')
    
    plt.tight_layout()
    plt.show()