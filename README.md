# SMA Monte Carlo Backtester

Python project that backtests a baseline **20/50-day SMA crossover** on **SPY** with a **$10,000** portfolio, and runs **1,000+ one-year Monte Carlo price paths** with trade logging and standard risk metrics.

## What it does

- Generates **1,000+** simulated one-year SPY paths from historical return statistics
- Runs a **20/50 SMA** rule on each path and on a historical QuantConnect backtest
- Logs trades (entry, exit, return) and summarizes wins / losses
- Reports **Sharpe**, **Sortino**, and **max drawdown**, etc.

## Run in QuantConnect (LEAN)

1. Create a new Algorithm Lab project
2. Replace `main.py` with this repo’s `main.py`
3. Backtest (daily resolution, SPY, 2012–2022 by default)

`research.py` is for the QuantConnect Research environment (Bollinger Band exploration on SPY).

## Run with Alpaca (paper)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ALPACA_API_KEY="your_paper_key"
export ALPACA_SECRET_KEY="your_paper_secret"
python alpaca_sma.py
```

Uses Alpaca market data for a local SPY 20/50 backtest with the same metrics, then optionally submits a paper order. Set `ALPACA_MODE=loop` to rebalance hourly.

## Repository contents

| File | Purpose |
|------|---------|
| `main.py` | QuantConnect LEAN algorithm (Monte Carlo + historical SMA) |
| `alpaca_sma.py` | Local metrics + Alpaca paper trading helper |
| `research.py` | QuantBook research snippet |
| `requirements.txt` | Local / Alpaca Python deps |
| `Results.json` / `*.png` | Saved QuantConnect backtest artifacts from an earlier run |

## Notes

- Re-run the LEAN backtest to refresh `Results.json` and images if you want artifacts to match the updated algorithm.
