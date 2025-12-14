# Multimodal Equity Sizing

This project implements a multimodal trading system that predicts next-day open-to-close excess returns for U.S. equities. It integrates quantitative price data with qualitative news sentiment using two distinct neural network towers (Leg 1: Price, Leg 2: News). The calibrated forecasts are aggregated and passed to an offline Conservative Q-Learning (CQL) agent to determine optimal discrete position sizes.

---

### Prologue: Running Instructions (`Kaggle`)

To reproduce results, run `01_prepare_data.ipynb` to download and preprocess price/volume data using `yfinance` and news headlines from the FNSPID dataset, engineering return, volatility, and sentiment features.  
Next, train the price and news models in `02_train_leg1_price_model.ipynb` and `02_train_leg2_news_model.ipynb` using walk-forward validation to generate out-of-fold predictions.  
Calibrate and aggregate predictions in `03_calibrate_and_stack.ipynb` using isotonic regression and disagreement-based shrinkage, then risk-normalize the combined signal.  
Train the offline Conservative Q-Learning agent in `04_train_cql.ipynb` to map normalized forecasts to discrete buy/sell/hold actions.  
Finally, run `05_backtest_and_reports.ipynb` to backtest the strategy, apply portfolio guards, and generate performance reports.

---

### 1. Data Pipeline (`src/data`)

The data module handles ingestion and preprocessing of both financial time-series and textual news data.

* **`loaders.py`**
  * **Financial Data**: Computes target excess returns (Open-to-Close vs. SPY), Corwin–Schultz spread estimates, and market regime indicators (VIX).
  * **News Processing**: Implements the `NewsDataset` class to process headlines and compute auxiliary features such as **velocity** and **novelty**, with a custom `han_collate_fn` for batching variable-length inputs.

---

### 2. Models (`src/models`)

Two supervised models independently generate return forecasts.

* **Leg 1: Price Tower (`price_mlp.py`)**
  * **Architecture**: Hybrid **LSTM–GRU** network processing sliding windows of technical features.
  * **Objective**: Predict next-day excess returns from price and volume dynamics.

* **Leg 2: News Tower (`HAN_l2.py`)**
  * **Architecture**: **FinBERT-based Hierarchical Attention Network (HAN)** with time-aware attention and auxiliary feature fusion.
  * **Objective**: Predict excess returns from headline sentiment and information flow.

* **Aggregator (`aggregator.py`)**
  * Combines calibrated outputs from both towers using equal weighting and applies disagreement-based shrinkage to reduce conviction under conflicting signals.

---

### 3. Reinforcement Learning (`src/rl`)

* **CQL Agent (`cql_agent.py`)**
  * Implements **Conservative Q-Learning (CQL)** for offline, cost-aware decision-making.
  * Inputs include the risk-normalized forecast, market regime indicators, and trading cost proxies.
  * Outputs discrete actions: Buy, Hold, or Sell.

---

### 4. Backtesting (`src/backtest`)

* **Simulator (`simulator.py`)**
  * Executes policy backtests with portfolio guards (turnover, exposure limits).
  * Computes NAV trajectories while accounting for transaction costs and slippage.
