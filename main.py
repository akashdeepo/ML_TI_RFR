import numpy as np
import pandas as pd
from stockdata import StockData
from pred_rfr import RFR_Predictor
import simulate_trading as sim
import metrics
import plots
import os
from datetime import datetime
import warnings
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_simulation.log'),
        logging.StreamHandler()
    ]
)

# Suppress specific pandas warnings - updated for current pandas version
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', message='.*Downcasting object dtype arrays.*')
pd.options.mode.chained_assignment = None

def setup_directories():
    """Create necessary directories for outputs."""
    directories = ["results", "trade_logs", "plots"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created/verified directory: {directory}")

def initialize_configuration():
    """Initialize all configuration parameters."""
    config = {
        # Dataset configuration
        "dataset_filename": 'src/data/SPY(3.25.2024-10.04.2024).csv',
        "start_timeofday": '09:00',  # CST times
        "end_timeofday": '14:30',    # CST times
        
        # Trading parameters
        "turnover": 0.004,
        "num_random_sims": 0,
        "random_level": 0.05,
        "test_frac": 0.2,
        
        # Model parameters
        "sell_quantile": 1/3,
        "buy_quantile": 2/3,
        
        # List of technical indicators to test
        "indicators": [
            tuple(),  # No technical indicators (baseline case)
            ("sma",),
            ("ema",),
            ("macd",),
            ("rsi",),
            ("boll",),
            ("so",),
            ("fib",),
            ("adx",),
            ("wrobv",),
            ("cci",),
            ("ichi",)
        ]
    }
    return config

def run_baseline_simulation(config):
    """
    Generate buy-and-hold baseline simulation.
    
    Returns:
    --------
    tuple: (portfolio, StockData)
        portfolio: DataFrame with baseline simulation results
        StockData: StockData instance used for the simulation
    """
    logging.info("Generating buy-and-hold baseline...")
    try:
        baseline_SD = StockData(
            config["dataset_filename"],
            test_frac=config["test_frac"],
            technical_indicator_list=[],
            tod_start=config["start_timeofday"],
            tod_end=config["end_timeofday"]
        )
        baseline_portfolio = sim.simulate_baseline(
            baseline_SD, 
            turnover_constraint=config["turnover"]
        )
        logging.info("Baseline simulation completed successfully")
        return baseline_portfolio, baseline_SD
    except Exception as e:
        logging.error(f"Error in baseline simulation: {str(e)}")
        raise

def run_indicator_simulation(indicators, config):
    """Run simulation for a specific set of indicators."""
    # Updated model naming logic
    if len(indicators) > 0:
        model_name = "rfr" + ''.join('_' + i for i in indicators)
    else:
        model_name = "RFR (no indicators)"
    
    logging.info(f"Running simulations for {model_name}...")
    
    try:
        # Initialize model and data
        model = RFR_Predictor(
            model_name,
            sell_quantile=config["sell_quantile"],
            buy_quantile=config["buy_quantile"]
        )
        sd = StockData(
            config["dataset_filename"],
            test_frac=config["test_frac"],
            technical_indicator_list=indicators,
            tod_start=config["start_timeofday"],
            tod_end=config["end_timeofday"]
        )

        # Train model
        trainX, trainY = sd.get_train_set()
        model.train(trainX, trainY)

        # Run trading simulation
        portfolio_value = sim.simulate_trading(
            sd,
            model,
            turnover_constraint=config["turnover"]
        )

        # Save trade log
        trade_log_filename = f"trade_logs/trade_log_{model.get_name().replace(' ', '_')}.csv"
        portfolio_value.to_csv(trade_log_filename)
        logging.info(f"Saved trade log to {trade_log_filename}")

        # Generate model analysis plot
        plots.plot_model_results(
            sd,
            model,
            f'plots/model_{model.get_name().replace(" ", "_")}.pdf',
            show=False
        )

        # Calculate metrics
        model_metrics = metrics.calculate_metrics(sd, model)
        
        return model.get_name(), portfolio_value, model_metrics
        
    except Exception as e:
        logging.error(f"Error in {model_name} simulation: {str(e)}")
        raise
    
    

def run_hybrid_strategy(top_indicators, config):
    """Run hybrid strategy using top performing indicators."""
    if not top_indicators:
        logging.info("No indicators for hybrid strategy")
        return None, None, None
        
    logging.info(f"Running hybrid model with indicators: {top_indicators}")
    hybrid_model_name = "rfr_hybrid_" + "_".join(top_indicators)
    
    try:
        # Initialize and train hybrid model
        model = RFR_Predictor(
            hybrid_model_name,
            sell_quantile=config["sell_quantile"],
            buy_quantile=config["buy_quantile"]
        )
        sd = StockData(
            config["dataset_filename"],
            test_frac=config["test_frac"],
            technical_indicator_list=top_indicators,
            tod_start=config["start_timeofday"],
            tod_end=config["end_timeofday"]
        )

        trainX, trainY = sd.get_train_set()
        model.train(trainX, trainY)

        # Run simulation
        portfolio_value = sim.simulate_trading(
            sd,
            model,
            turnover_constraint=config["turnover"]
        )

        # Save trade log
        trade_log_filename = f"trade_logs/trade_log_{model.get_name().replace(' ', '_')}.csv"
        portfolio_value.to_csv(trade_log_filename)
        logging.info(f"Saved hybrid trade log to {trade_log_filename}")

        # Calculate metrics
        model_metrics = metrics.calculate_metrics(sd, model)
        
        return model.get_name(), portfolio_value, model_metrics
        
    except Exception as e:
        logging.error(f"Error in hybrid strategy: {str(e)}")
        raise

def save_results(portfolios, metrics_results):
    """Save simulation results and metrics to CSV files."""
    try:
        # Save simulation returns
        simulation_returns_df = pd.DataFrame({
            "Model": list(portfolios.keys()),
            "Final Return": [portfolios[model]["value"].iloc[-1] for model in portfolios]
        })
        simulation_returns_df.to_csv("results/simulation_returns.csv", index=False)
        logging.info("Saved simulation returns to results/simulation_returns.csv")

        # Save metrics
        metrics_df = pd.DataFrame.from_dict(metrics_results, orient='index')
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        metrics_df.to_csv("results/performance_metrics.csv", index=False)
        logging.info("Saved performance metrics to results/performance_metrics.csv")
        
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise

def main():
    start_time = datetime.now()
    logging.info("Starting trading simulation pipeline")
    
    try:
        # Setup and initialization
        setup_directories()
        config = initialize_configuration()
        
        # Results storage
        portfolios = {}
        metrics_results = {}
        indicator_returns = {}
        
        # Generate baseline and store StockData instance
        baseline_portfolio, baseline_SD = run_baseline_simulation(config)
        
        # Run individual indicator simulations
        for indicators in config["indicators"]:
            model_name, portfolio_value, model_metrics = run_indicator_simulation(indicators, config)
            portfolios[model_name] = portfolio_value
            metrics_results[model_name] = model_metrics
            indicator_returns[model_name] = portfolio_value['value'].iloc[-1]
        
        # Identify top performers
        sorted_indicators = sorted(
            [(k, v) for k, v in indicator_returns.items() if k != "RFR (no indicators)"],  # Updated name
            key=lambda x: x[1],
            reverse=True
        )
        top_3_model_names = sorted_indicators[:3]
        logging.info(f"Top 3 performing models: {[name for name, _ in top_3_model_names]}")
        
        # Run hybrid strategy
        top_3_indicators = [name[4:] for name, _ in top_3_model_names if name.startswith("rfr_")]
        hybrid_name, hybrid_portfolio, hybrid_metrics = run_hybrid_strategy(top_3_indicators, config)
        
        if hybrid_name:
            portfolios[hybrid_name] = hybrid_portfolio
            metrics_results[hybrid_name] = hybrid_metrics
        
        # Generate random trading bounds if requested
        upper_bound = None
        lower_bound = None
        if config["num_random_sims"] > 0:
            logging.info(f"Generating random trading bounds with {config['num_random_sims']} simulations...")
            upper_bound, lower_bound = sim.simulate_random_trading(
                baseline_SD,
                alpha=config["random_level"],
                num_sims=config["num_random_sims"],
                turnover_constraint=config["turnover"]
            )
        
        # Generate final composite plot
        logging.info("Generating composite performance plot...")
        plots.generate_composite_plot(
            portfolios,
            baseline=baseline_portfolio,
            upper=upper_bound,
            lower=lower_bound,
            rand_level=config["random_level"],
            show=True,
            filename='plots/portfolio_strategy_comparison.pdf'
        )
        
        # Save final results
        save_results(portfolios, metrics_results)
        
        # Log completion time
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Trading simulation pipeline completed. Duration: {duration}")
        
    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
        raise
    
    
if __name__ == "__main__":
    main()