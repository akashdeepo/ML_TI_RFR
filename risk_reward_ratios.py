# risk_reward_ratios.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Function to calculate per-minute risk-free rate from annual yield
def calculate_per_minute_risk_free_rate(annual_rate):
    return (1 + annual_rate)**(1 / (365.25 * 24 * 60)) - 1

# Function to convert Excel serial number to datetime
def convert_excel_serial_date(serial_date):
    if isinstance(serial_date, (float, int)):
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(serial_date, 'D')
    return serial_date

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(excess_returns):
    mean_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns)
    if std_dev_excess_return == 0:
        return np.nan  # Avoid division by zero
    return mean_excess_return / std_dev_excess_return

# Function to calculate Sortino Ratio based on downside risk
def calculate_sortino_ratio(excess_returns):
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return np.nan  # Avoid division by zero
    return np.mean(excess_returns) / downside_std

# Adjusted Rachev Ratio with distinct quantiles
def calculate_rachev_ratio(excess_returns, beta=0.1, gamma=0.9):  # Adjusted quantiles
    var_beta = np.percentile(excess_returns, 100 * beta)
    avar_beta = np.mean(excess_returns[excess_returns <= var_beta])  # AVaR_beta
    var_gamma = np.percentile(excess_returns, 100 * gamma)
    avar_gamma = np.mean(excess_returns[excess_returns >= var_gamma])  # AVaR_gamma
    if avar_beta == 0:
        return np.nan  # Avoid division by zero
    return avar_gamma / abs(-avar_beta)

# Function to calculate Modified Rachev Ratio
def calculate_modified_rachev_ratio(excess_returns, beta=0.05, gamma=0.95, delta=0.05, epsilon=0.95):
    var_beta = np.percentile(excess_returns, 100 * beta)
    avar_beta_gamma = np.mean(excess_returns[excess_returns <= var_beta]) / gamma
    var_delta = np.percentile(excess_returns, 100 * delta)
    avar_delta_epsilon = np.mean(excess_returns[excess_returns >= var_delta]) / epsilon
    if avar_beta_gamma == 0:
        return np.nan  # Avoid division by zero
    return avar_delta_epsilon / abs(avar_beta_gamma)

# Adjusted Distortion RRR with symmetric quantile calculation
def calculate_distortion_rrr(excess_returns, beta=0.05):
    var_beta = np.percentile(excess_returns, 100 * beta)
    var_one_minus_beta = np.percentile(excess_returns, 100 * (1 - beta))
    distorted_returns_positive = np.mean(excess_returns[excess_returns >= var_one_minus_beta])
    distorted_returns_negative = np.mean(excess_returns[excess_returns <= var_beta])
    if distorted_returns_negative == 0:
        return np.nan  # Avoid division by zero
    return distorted_returns_positive / abs(distorted_returns_negative)

# Function to calculate Gains-Loss Ratio
def calculate_gains_loss_ratio(excess_returns):
    positive_returns = excess_returns[excess_returns > 0]
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return np.nan  # Avoid division by zero
    
    ratio = np.mean(positive_returns) / abs(np.mean(negative_returns))
    
    # Scale between -1 and 1
    return (2 / (1 + np.exp(-ratio))) - 1  # Sigmoid-like scaling

# Function to calculate STAR Ratio
def calculate_star_ratio(excess_returns, cvar_level=0.05):
    var_level = np.percentile(excess_returns, 100 * cvar_level)
    avar = np.mean(excess_returns[excess_returns <= var_level])
    if avar == 0:
        return np.nan  # Avoid division by zero
    return np.mean(excess_returns) / abs(avar)

# Function to calculate MiniMax Ratio
def calculate_minimax_ratio(excess_returns):
    """
    Calculate MiniMax Ratio with improved formula.
    
    MiniMax Ratio = Expected Return / Maximum Drawdown
    where Maximum Drawdown is expressed as a positive number
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + excess_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(drawdowns.min())  # Take absolute value
    
    # Calculate expected return (annualized)
    expected_return = excess_returns.mean() * 252  # Annualize assuming daily data
    
    if max_drawdown == 0:
        return np.nan
    
    return expected_return / max_drawdown

# Function to calculate Gini Ratio
def calculate_gini_ratio(returns):
    # Normalize returns to avoid extreme values
    norm_returns = (returns - np.min(returns)) / (np.max(returns) - np.min(returns))
    
    # Sort the normalized returns
    sorted_returns = np.sort(norm_returns)
    n = len(sorted_returns)
    index = np.arange(1, n + 1)
    
    # Gini calculation
    gini_numerator = np.sum((2 * index - n - 1) * sorted_returns)
    gini_denominator = n * np.sum(sorted_returns)
    
    if gini_denominator == 0:
        return np.nan  # Avoid division by zero
    
    return gini_numerator / gini_denominator

# Function to calculate all reward-risk ratios
def calculate_all_ratios(excess_returns):
    return {
        "Sharpe Ratio": calculate_sharpe_ratio(excess_returns),
        "Sortino Ratio": calculate_sortino_ratio(excess_returns),
        "Rachev Ratio": calculate_rachev_ratio(excess_returns),  # Uses updated beta and gamma values
        "Modified Rachev Ratio": calculate_modified_rachev_ratio(excess_returns),
        "Distortion RRR": calculate_distortion_rrr(excess_returns),  # Uses standard beta for symmetry
        "Gains-Loss Ratio": calculate_gains_loss_ratio(excess_returns),
        "STAR Ratio": calculate_star_ratio(excess_returns),
        "MiniMax Ratio": calculate_minimax_ratio(excess_returns),
        "Gini Ratio": calculate_gini_ratio(excess_returns)
    }



# Function to calculate ratios for a single portfolio file and return the results as a dictionary
def calculate_ratios_from_csv(portfolio_csv, treasury_csv):
    """
    Calculate ratios from portfolio and treasury data.
    
    Parameters:
    -----------
    portfolio_csv : str
        Path to portfolio CSV containing timestamp, value, etc.
    treasury_csv : str
        Path to treasury CSV containing Dates and Close columns
    """
    try:
        # Load portfolio CSV
        portfolio_df = pd.read_csv(portfolio_csv, parse_dates=['timestamp'], index_col='timestamp')
        
        # Calculate returns if not present
        if 'returns' not in portfolio_df.columns:
            portfolio_df['returns'] = portfolio_df['value'].pct_change()
        
        # Load treasury CSV
        treasury_df = pd.read_csv(treasury_csv)
        treasury_df['Dates'] = pd.to_datetime(treasury_df['Dates'], errors='coerce')
        treasury_df.dropna(subset=['Dates'], inplace=True)
        treasury_df.sort_values(by='Dates', inplace=True)

        # Calculate per-minute risk-free rate
        treasury_df['per_minute_risk_free_rate'] = treasury_df['Close'] / 100
        treasury_df['per_minute_risk_free_rate'] = treasury_df['per_minute_risk_free_rate'].apply(
            calculate_per_minute_risk_free_rate
        )
        
        # Merge portfolio and treasury data
        merged_df = pd.merge_asof(
            portfolio_df.reset_index(),
            treasury_df[['Dates', 'per_minute_risk_free_rate']],
            left_on='timestamp',
            right_on='Dates',
            direction='backward'
        )
        
        # Calculate excess returns
        merged_df['excess_returns'] = merged_df['returns'] - merged_df['per_minute_risk_free_rate']

        # Calculate ratios
        return calculate_all_ratios(merged_df['excess_returns'].dropna())
        
    except Exception as e:
        print(f"Error processing {os.path.basename(portfolio_csv)}: {str(e)}")
        print(f"Columns in portfolio data: {portfolio_df.columns.tolist()}")
        return None

# Function to generate a single heatmap for all portfolios
def plot_heatmap(ratios_df, output_pdf):
    """
    Enhanced heatmap with better formatting for different ratio scales.
    """
    # Clean model names
    ratios_df.index = ratios_df.index.str.replace('trade_log_', '')
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Format numbers based on ratio type
    def format_value(val):
        if abs(val) < 0.01:
            return f'{val:.2e}'  # Scientific notation for very small numbers
        else:
            return f'{val:.3f}'  # 3 decimal places for others
    
    # Create heatmap with custom formatting
    sns.heatmap(
        ratios_df,
        annot=True,
        cmap='RdYlBu_r',
        linewidths=0.5,
        fmt='.3g',  # General format
        annot_kws={
            "size": 8,
            "weight": "bold"
        },
        cbar_kws={
            'label': 'Value',
            'format': '%.3f'
        },
        center=0
    )
    
    # Improve formatting
    plt.title('Risk-Reward Ratios Comparison Across Models', fontsize=16, pad=20)
    plt.xlabel('Performance Metrics', fontsize=12)
    plt.ylabel('Trading Models', fontsize=12)
    
    # Add metric descriptions
    metric_descriptions = {
        'Sharpe Ratio': 'Risk-adjusted return',
        'Sortino Ratio': 'Downside risk-adjusted return',
        'Rachev Ratio': 'Tail risk measure',
        'MiniMax Ratio': 'Return/Max drawdown',
        'Gini Ratio': 'Return dispersion'
    }
    
    # Add annotations for metric interpretations
    for i, metric in enumerate(ratios_df.columns):
        if metric in metric_descriptions:
            plt.annotate(
                metric_descriptions[metric],
                xy=(i, -0.5),
                xytext=(0, -20),
                ha='center',
                va='top',
                fontsize=8,
                rotation=45
            )
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

# radar chart
def plot_radar_chart(ratios_df, output_pdf):
    """
    Generate publication-quality radar chart of risk-reward ratios.
    """
    # Clean model names
    ratios_df.index = ratios_df.index.str.replace('', '')
    
    # Select top performing models based on Sharpe ratio
    top_models = ratios_df.nlargest(5, 'Sharpe Ratio')
    
    # Prepare data
    labels = ratios_df.columns
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Normalize data for better visualization
    normalized_ratios = (ratios_df - ratios_df.min()) / (ratios_df.max() - ratios_df.min())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
    
    # Use a color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_models)))
    
    # Plot each model
    for idx, (model, row) in enumerate(top_models.iterrows()):
        values = normalized_ratios.loc[model].tolist()
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, label=model, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Customize the chart
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Clockwise
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    
    # Add legend with better placement
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.2, 0.5),
        fontsize=10,
        title='Top 5 Models\n(by Sharpe Ratio)'
    )
    
    # Add title
    plt.title('Risk-Reward Profile Comparison\nTop 5 Models', fontsize=16, y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Enhanced radar chart saved to {output_pdf}")

# Main function to calculate and save ratios for all portfolios, and generate combined plots
def calculate_and_save_combined_ratios(portfolio_folder, treasury_csv, output_folder):
    """Process all portfolios and create visualizations."""
    all_ratios = {}

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Process each portfolio CSV
    for filename in os.listdir(portfolio_folder):
        if filename.endswith('.csv'):
            portfolio_csv = os.path.join(portfolio_folder, filename)
            model_name = filename.replace('trade_log_', '').replace('.csv', '')

            # Calculate ratios
            ratios = calculate_ratios_from_csv(portfolio_csv, treasury_csv)
            if ratios is not None:
                all_ratios[model_name] = ratios

    # Convert to DataFrame
    if all_ratios:
        ratios_df = pd.DataFrame(all_ratios).T
        
        # Save combined ratios
        combined_csv_path = os.path.join(output_folder, 'combined_risk_reward_ratios.csv')
        ratios_df.to_csv(combined_csv_path)
        print(f"Combined risk-reward ratios saved to {combined_csv_path}")

        # Generate visualizations
        plot_heatmap(ratios_df, os.path.join(output_folder, 'combined_risk_reward_heatmap.pdf'))
        plot_radar_chart(ratios_df, os.path.join(output_folder, 'combined_risk_reward_radar_chart.pdf'))
        
        return ratios_df
    else:
        print("No valid ratios were calculated")
        return pd.DataFrame()
    
    

if __name__ == "__main__":
    # Use proper path joining
    portfolio_folder = 'trade_logs'
    treasury_csv = os.path.join('src', 'data', 'US_treasury(April-Sept).csv')
    output_folder = os.path.join('results', 'RRR')

    print(f"\nUsing paths:")
    print(f"Portfolio folder: {portfolio_folder}")
    print(f"Treasury data: {treasury_csv}")
    print(f"Output folder: {output_folder}")

    try:
        if not os.path.exists(treasury_csv):
            raise FileNotFoundError(f"Treasury file not found: {treasury_csv}")
            
        ratios_df = calculate_and_save_combined_ratios(
            portfolio_folder,
            treasury_csv,
            output_folder
        )
        
        if not ratios_df.empty:
            print("\nCalculated ratios for models:")
            print(ratios_df.index.tolist())
            
    except Exception as e:
        print(f"\nError: {str(e)}")