# Multimodal Equity Sizing

This project implements a multimodal trading system that predicts next-day open-to-close excess returns for U.S. equities. It integrates quantitative price data with qualitative news sentiment using two distinct neural network "towers" (Leg 1 and Leg 2). The forecasts are aggregated and used by an offline Conservative Q-Learning (CQL) agent to determine optimal discrete position sizes.


### 0. Running Instruction (`Kaggle`)
1. To reproduce the results, first run `01_prepare_data.ipynb` to download and process price/volume data for U.S stocks using `yfinance` and news headlines from the Kaggle dataset, engineering return, volatility, and sentiment features.  
2., train the price model in `02_train_leg1_price_model.ipynb` and the news model in `02_train_leg2_news_model.ipynb` using walk-forward validation to generate out-of-fold predictions.  
3. Calibrate and aggregate these predictions in `03_calibrate_and_stack.ipynb` using isotonic regression and disagreement-based shrinkage, then risk-normalize the combined signal.  
4. Train the offline Conservative Q-Learning (CQL) agent in `04_train_cql.ipynb` to map normalized forecasts to discrete buy/sell/hold actions.  
Finally, run `05_backtest_and_reports.ipynb` to perform backtesting, apply portfolio guards, and generate performance reports.



### 1. Data Pipeline (`src/data`)
The data module handles the ingestion and preprocessing of both financial time-series and textual news data.

* **`loaders.py`**:
    * **Financial Data**: Contains helpers like `get_excess_return` to compute the target variable (Open-to-Close return vs SPY) and `get_spread_z` to calculate Corwin-Schultz spread estimates. It also computes market regime signals using `get_vix_data`.
    * **News Processing**: The `NewsDataset` class processes headlines, calculating auxiliary features such as **velocity** (news volume) and **novelty** (new entity mentions). It employs a custom `han_collate_fn` to batch variable-length document sequences for the Hierarchical Attention Network.

### 2. Models (`src/models`)
Two separate supervised learning models generate return forecasts, which are then combined.

* **Leg 1: Price Tower (`price_mlp.py`)**
    * **Architecture**: A Hybrid **LSTM-GRU** model. It processes sliding windows of technical features (returns, volatility, etc.) through a GRU layer followed by an LSTM layer to capture temporal dependencies.
    * **Goal**: Predict next-day excess returns based purely on price and volume dynamics.

* **Leg 2: News Tower (`HAN_l2.py`)**
    * **Architecture**: A **FinbertHAN** (Hierarchical Attention Network).
        1.  **Embedding**: Uses a frozen `DistilRoBERTa` (fine-tuned on financial text) to embed headlines.
        2.  **Document Encoding**: A Bi-GRU encodes the narrative flow of headlines for a given day.
        3.  **Attention**: A time-aware attention mechanism applies decay to older headlines, prioritizing recent news.
        4.  **Fusion**: The document summary is concatenated with auxiliary features (novelty, velocity, flags) before the final regression head.

* **Aggregator (`aggregator.py`)**
    * **Logic**: Combines the calibrated outputs of the Price and News towers using a weighted average.
    * **Disagreement Shrink**: Applies a penalty factor to the combined forecast when the two models disagree significantly, reducing conviction in conflicting signals.

### 3. Reinforcement Learning (`src/rl`)
The final decision-making component converts forecasts into actionable trading decisions.

* **CQL Agent (`cql_agent.py`)**
    * **Algorithm**: Implements **Conservative Q-Learning (CQL)**, an offline RL algorithm designed to avoid overestimating Q-values for actions not seen in the training data.
    * **State Space**: Inputs include the risk-normalized aggregated forecast ($z$), market regime (VIX z-score), and trading costs.
    * **Q-Network**: A simple MLP that maps the state to Q-values for discrete actions (Buy, Hold, Sell).
