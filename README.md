# Stock Market Technical Analysis Framework

## Overview
This repository contains the implementation of a comprehensive stock market technical analysis framework, designed for academic research in financial time series prediction. The framework provides robust functionality for processing and analyzing high-frequency stock market data, with a particular focus on computing various technical indicators and preparing data for machine learning applications.

## Key Features

- **Flexible Data Processing**: Handles minute-level stock data with customizable time windows and trading hours
- **Dividend Adjustment**: Automatic price adjustment for dividend events to ensure data continuity
- **Technical Indicators**: Implementation of multiple technical analysis indicators including:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Moving Average Convergence Divergence (MACD)
  - Relative Strength Index (RSI)
  - Bollinger Bands
  - Stochastic Oscillator
  - Fibonacci Retracement
  - Average Directional Index (ADX)
  - On-Balance Volume (OBV)
  - Windowed Relative On-Balance Volume (WROBV)
  - Commodity Channel Index (CCI)
  - Ichimoku Cloud Indicators

## Requirements

- Python 3.x
- pandas
- numpy

## Installation

Clone the repository:
```bash
git clone [https://github.com/akashdeepo/ML_TI_RFR.git]
cd ML_TI_RFR
```

Install required packages:
```bash
pip install pandas numpy
```

## Usage

The main class `StockData` can be initialized with various parameters to customize the analysis:

```python
from stockdata import StockData

# Initialize with basic parameters
stock_data = StockData(
    filepath='path/to/your/data.csv',
    technical_indicator_list=['sma', 'ema', 'macd'],
    test_frac=0.2,
    tod_start='9:00',
    tod_end='14:30',
    delta_t=1
)

# Get training and test sets
trainX, trainY = stock_data.get_train_set()
testX, testY = stock_data.get_test_set()
```

### Key Parameters

- `filepath`: Path to CSV file containing stock data
- `technical_indicator_list`: List of technical indicators to compute
- `test_frac`: Fraction of data to use for testing (default: 0.2)
- `test_size`: Override test_frac with specific size
- `test_start_date`: Specific start date for test set
- `tod_start`: Trading day start time
- `tod_end`: Trading day end time
- `delta_t`: Future time steps for prediction

## Data Format

The input CSV file should contain the following columns:
- timestamp
- open
- high
- low
- close
- volume

## Features

### Data Preprocessing
- Automatic handling of missing values
- Log-return calculations
- Volume normalization
- Trading hours filtering
- Dividend adjustment

### Technical Indicators
The framework supports multiple technical indicators with customizable parameters. Each indicator is implemented with careful consideration of financial theory and practical applications.

### Train-Test Split
Multiple options for creating train-test splits:
- Percentage-based splitting
- Fixed-size test sets
- Date-range-based splitting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
