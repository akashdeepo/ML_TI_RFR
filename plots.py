import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import logging


def plot_series(Series, args, filename=None, show=False):
    """
    Plot time series with proper timestamp handling and validation.
   
    Parameters:
    Series : list of pandas.Series objects with identical timestamps
    args : list of dictionaries containing plt.plot arguments for each series
    filename : output file path (optional)
    show : boolean to control display
    """
    if (not show) and (filename is None):
        print("plot_series: nothing to do - 'show' is False, and filename is None.")
        return

    # Find common timestamp range and align all series
    common_index = Series[0].index
    for s in Series[1:]:
        common_index = common_index.intersection(s.index)

    if len(common_index) == 0:
        raise ValueError("No common timestamps found between series")

    # Align all series to common timestamps
    aligned_series = [s.loc[common_index] for s in Series]
    
    # Set style
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot series
    x = range(len(common_index))
    for j in range(len(aligned_series)):
        ax.plot(x, np.array(aligned_series[j]), **args[j])
    
    # Place one tick per day at the beginning of each day
    tick_locations = [i for i in range(1, len(common_index)) 
                     if common_index[i].day != common_index[i-1].day]
    
    # Label ticks: at most 8 labels, skipping if necessary
    n = len(tick_locations)
    lskip = 1 if n <= 8 else n // 8
    tick_labels = [str(common_index[i].date()) if j % lskip == 0 else None
                  for j, i in enumerate(tick_locations)]
    
    ax.set(xticks=tick_locations, xticklabels=tick_labels)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='lower right', bbox_to_anchor=(0.9, 0.9), frameon=True)
    plt.subplots_adjust(bottom=0.15)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

#########################################################################################

def generate_composite_plot(
    portfolios: dict,
    baseline: pd.DataFrame,
    upper: pd.DataFrame = None,
    lower: pd.DataFrame = None,
    rand_level: float = None,
    show: bool = True,
    filename: str = "portfolio_strategy_comparison.pdf"
) -> None:
    """
    Generate publication-quality plot of trading strategy performance.
    
    Parameters:
    -----------
    portfolios : dict
        Dictionary of portfolio DataFrames
    baseline : DataFrame
        Buy-and-hold baseline results
    upper : DataFrame, optional
        Upper confidence bound
    lower : DataFrame, optional
        Lower confidence bound
    rand_level : float, optional
        Confidence level for bounds
    show : bool, optional
        Whether to display the plot
    filename : str, optional
        Output filepath
    """
    # Set up the figure with better proportions
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
    
    # Main performance plot
    ax1 = fig.add_subplot(gs[0])
    
    # Ensure baseline is properly formatted
    if not isinstance(baseline, pd.DataFrame):
        baseline = pd.DataFrame(baseline)
    if 'value' not in baseline.columns:
        baseline = pd.DataFrame({'value': baseline})
        
    # Colors for different strategy categories
    n_strategies = len(portfolios)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_strategies))
    
    # Create continuous trade point indices
    trade_points = np.arange(len(baseline.index))
    
    # Plot baseline
    baseline_end = float(baseline['value'].iloc[-1])
    ax1.plot(trade_points, baseline['value'], 
            color='black', linestyle='--', linewidth=2,
            label=f'Buy-and-Hold ({baseline_end:.0f})',
            zorder=10)
    
    # Sort strategies by final value
    portfolio_ends = {
        name: df['value'].iloc[-1] 
        for name, df in portfolios.items()
    }
    sorted_portfolios = dict(sorted(
        portfolio_ends.items(),
        key=lambda x: x[1],
        reverse=True
    ))
    
    # Plot active strategies
    for i, (name, portfolio) in enumerate(sorted_portfolios.items()):
        portfolio_data = portfolios[name]
        if not isinstance(portfolio_data, pd.DataFrame):
            portfolio_data = pd.DataFrame(portfolio_data)
        if 'value' not in portfolio_data.columns:
            portfolio_data = pd.DataFrame({'value': portfolio_data})
        
        # Create trade points for this portfolio
        portfolio_points = np.arange(len(portfolio_data.index))
        
        end_value = float(portfolio_data['value'].iloc[-1])
        relative_perf = ((end_value / baseline_end) - 1) * 100
        label = f'{name} ({end_value:.0f}, {relative_perf:+.1f}%)'
        
        ax1.plot(portfolio_points, portfolio_data['value'],
                color=colors[i], linewidth=1.5, label=label,
                alpha=0.8)
    
    # Confidence bounds if provided
    if upper is not None and lower is not None and rand_level is not None:
        bound_points = np.arange(len(upper.index))
        ax1.fill_between(bound_points, lower['value'], upper['value'],
                        color='gray', alpha=0.2,
                        label=f'{int(100*rand_level)}% Confidence Interval')
    
    # Enhanced formatting
    ax1.set_title('Trading Strategy Performance Comparison', 
                 fontsize=14, pad=20)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Format y-axis with comma separator
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Format x-axis to show selected dates
    n_points = len(baseline.index)
    n_labels = 8  # Number of date labels to show
    label_positions = np.linspace(0, n_points-1, n_labels, dtype=int)
    ax1.set_xticks(label_positions)
    ax1.set_xticklabels([baseline.index[i].strftime('%Y-%m-%d') 
                         for i in label_positions], 
                         rotation=45, ha='right')
    
    # Legend with scrollable box if too many items
    if len(sorted_portfolios) > 10:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  fontsize=10, frameon=True, framealpha=0.9,
                  ncol=1)
    else:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  fontsize=10, frameon=True, framealpha=0.9,
                  ncol=1)
    
    # Add drawdown subplot
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate and plot drawdowns for all strategies
    for i, (name, portfolio) in enumerate(sorted_portfolios.items()):
        portfolio_data = portfolios[name]
        if not isinstance(portfolio_data, pd.DataFrame):
            portfolio_data = pd.DataFrame(portfolio_data)
        if 'value' not in portfolio_data.columns:
            portfolio_data = pd.DataFrame({'value': portfolio_data})
        
        portfolio_points = np.arange(len(portfolio_data.index))
        rolling_max = portfolio_data['value'].expanding().max()
        drawdowns = (portfolio_data['value'] - rolling_max) / rolling_max * 100
        ax2.plot(portfolio_points, drawdowns, 
                color=colors[i], linewidth=1, alpha=0.8)
    
    # Baseline drawdown
    rolling_max = baseline['value'].expanding().max()
    drawdowns = (baseline['value'] - rolling_max) / rolling_max * 100
    ax2.plot(trade_points, drawdowns,
            color='black', linestyle='--', linewidth=2)
    
    # Format drawdown x-axis
    ax2.set_xticks(label_positions)
    ax2.set_xticklabels([baseline.index[i].strftime('%Y-%m-%d') 
                         for i in label_positions], 
                         rotation=45, ha='right')
    
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout with more space for legend
    plt.subplots_adjust(right=0.85)
    
    # Save with high DPI
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
        

#####################################################################################    

def plot_model_results(SD, Model, filename=None, last_n=70, show=False):
    """
    Plot comprehensive model analysis results with enhanced visualization.
   
    Parameters:
    SD : StockData object
    Model : Predictor model object
    filename : output file path (optional)
    last_n : number of last points to show in time series plot
    show : boolean to control display
    """
    x_train, y_train = SD.get_train_set()
    x_test, y_test = SD.get_test_set()
    y_train_pred = Model.apply(x_train)
    y_test_pred = Model.apply(x_test)
    feature_names = SD.feature_column_names()
    
    feature_importances = getattr(Model, 'feature_importances', lambda: None)()

    # Set style
    plt.style.use('seaborn-whitegrid')
    
    # Create subplots with better layout
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Training scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    plot_scatter(y_train, y_train_pred, 'Training Data: Actual vs Predicted', ax1)
    
    # Training residuals
    ax2 = fig.add_subplot(gs[0, 1])
    plot_residuals(y_train_pred, y_train, 'Training Data: Residuals', ax2)
    
    # Feature importance if available
    ax3 = fig.add_subplot(gs[0, 2])
    if feature_importances is not None:
        plot_feature_importance(feature_importances, feature_names, ax3)
    
    # Testing scatter plot
    ax4 = fig.add_subplot(gs[1, 0])
    plot_scatter(y_test, y_test_pred, 'Testing Data: Actual vs Predicted', ax4)
    
    # Testing residuals
    ax5 = fig.add_subplot(gs[1, 1])
    plot_residuals(y_test_pred, y_test, 'Testing Data: Residuals', ax5)
    
    # Time series comparison
    ax6 = fig.add_subplot(gs[1, 2])
    plot_time_series(y_train, y_train_pred, y_test, y_test_pred, last_n, ax6)
    
    # Add super title
    plt.suptitle(f'Model Analysis: {Model.get_name()}', fontsize=14, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_scatter(y_true, y_pred, title, ax):
    """Enhanced scatter plot with regression line and statistics."""
    ax.scatter(y_true, y_pred, alpha=0.5, color='blue')
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), "r--", alpha=0.8)
    
    # Add perfect prediction line
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'k--', alpha=0.5, label='Perfect Prediction')
    
    # Calculate R-squared
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{title}\n$R^2$ = {r_squared:.3f}')
    ax.grid(True, alpha=0.3)

def plot_residuals(y_pred, y_true, title, ax):
    """Enhanced residuals plot with statistical annotations."""
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, color='blue')
    ax.axhline(0, color='red', linestyle='--', alpha=0.8)
    
    # Add mean and std lines
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax.axhline(mean_residual, color='green', linestyle='--', alpha=0.5, 
               label=f'Mean: {mean_residual:.3f}')
    ax.axhline(mean_residual + 2*std_residual, color='orange', linestyle=':', alpha=0.5,
               label='+2σ')
    ax.axhline(mean_residual - 2*std_residual, color='orange', linestyle=':', alpha=0.5,
               label='-2σ')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{title}\nσ = {std_residual:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_feature_importance(feature_importances, feature_names, ax):
    """Enhanced feature importance visualization."""
    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1]
    
    # Plot top 15 features if there are more than 15
    if len(indices) > 15:
        indices = indices[:15]
        ax.set_title('Top 15 Feature Importances')
    else:
        ax.set_title('Feature Importances')
    
    # Create horizontal bar plot
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, feature_importances[indices], align='center', alpha=0.8,
            color=plt.cm.viridis(np.linspace(0, 0.8, len(indices))))
    
    # Add feature names and importance values
    for i, v in enumerate(feature_importances[indices]):
        ax.text(v, i, f'{v:.3f}', va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.grid(True, alpha=0.3)

def plot_time_series(y_train, y_train_pred, y_test, y_test_pred, last_n, ax):
    """
    Enhanced time series visualization with error metrics and formatting.
    
    Parameters:
    -----------
    y_train : Series
        Actual training values
    y_train_pred : Series
        Predicted training values
    y_test : Series
        Actual test values
    y_test_pred : Series
        Predicted test values
    last_n : int
        Number of last points to show
    ax : matplotlib.axes.Axes
        Axes object to plot on
    """
    y_actual = pd.concat([y_train, y_test]).sort_index()
    y_predicted = pd.concat([y_train_pred, y_test_pred]).sort_index()

    # Get last n points
    y_actual_last_n = y_actual[-last_n:]
    y_predicted_last_n = y_predicted[-last_n:]

    # Plot with different styles for actual and predicted
    ax.plot(y_actual_last_n.index, y_actual_last_n,
            label='Actual', color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.plot(y_predicted_last_n.index, y_predicted_last_n,
            label='Predicted', color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Calculate error metrics
    mse = np.mean((y_actual_last_n - y_predicted_last_n) ** 2)
    mae = np.mean(np.abs(y_actual_last_n - y_predicted_last_n))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_actual_last_n - y_predicted_last_n) ** 2) / 
              np.sum((y_actual_last_n - np.mean(y_actual_last_n)) ** 2))

    # Add metrics annotation
    metrics_text = (f'RMSE: {rmse:.4f}\n'
                   f'MAE: {mae:.4f}\n'
                   f'R²: {r2:.4f}')
    
    # Position the metrics box
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=box_props,
            fontsize=9)

    # Enhance formatting
    ax.set_title('Time Series Prediction Comparison', fontsize=12)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    
    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    # Adjust y-axis limits to show some padding
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    ax.set_ylim(ymin - 0.1*y_range, ymax + 0.1*y_range)

    # Add vertical line separating train/test if visible in the window
    test_start = y_test.index[0]
    if test_start in y_actual_last_n.index:
        ax.axvline(x=test_start, color='green', linestyle=':', alpha=0.5,
                  label='Train/Test Split')
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    return ax