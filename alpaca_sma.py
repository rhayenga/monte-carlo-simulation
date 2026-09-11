# Paper-trade the same 20/50 SPY SMA using Alpaca.
# Requires: pip install -r requirements.txt
# Set ALPACA_API_KEY and ALPACA_SECRET_KEY (paper keys recommended).

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
except ImportError as exc:
    raise SystemExit(
        "Missing Alpaca packages. Run: pip install -r requirements.txt"
    ) from exc


SYMBOL = "SPY"
FAST = 20
SLOW = 50
STARTING_CASH = 10000.0


def load_daily_closes(api_key: str, secret: str, symbol: str, lookback_days: int = 400) -> pd.Series:
    client = StockHistoricalDataClient(api_key, secret)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise RuntimeError(f"No bars returned for {symbol}")
    if isinstance(bars.index, pd.MultiIndex):
        closes = bars.xs(symbol)["close"]
    else:
        closes = bars["close"]
    return closes.astype(float)


def sma_signal(closes: pd.Series, fast: int = FAST, slow: int = SLOW) -> str:
    if len(closes) < slow:
        return "hold"
    f = float(closes.iloc[-fast:].mean())
    s = float(closes.iloc[-slow:].mean())
    if f > s:
        return "long"
    return "flat"


def sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if returns.size < 2 or np.std(returns, ddof=1) == 0:
        return float("nan")
    return float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def sortino(returns: np.ndarray, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    if returns.size < 2 or downside.size == 0 or np.std(downside, ddof=1) == 0:
        return float("nan")
    return float(np.mean(returns) / np.std(downside, ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def backtest_local(closes: pd.Series, cash: float = STARTING_CASH):
    """Offline SMA path with trade logging (no brokerage calls)."""
    prices = closes.to_numpy(dtype=float)
    n = len(prices)
    equity = np.empty(n, dtype=float)
    cash_bal = cash
    shares = 0.0
    trades = []
    entry_price = None

    for i in range(n):
        if i >= SLOW - 1:
            f = float(np.mean(prices[i - FAST + 1 : i + 1]))
            s = float(np.mean(prices[i - SLOW + 1 : i + 1]))
            px = float(prices[i])
            if f > s and shares <= 0:
                shares = cash_bal / px
                cash_bal = 0.0
                entry_price = px
            elif f <= s and shares > 0:
                cash_bal = shares * px
                if entry_price and entry_price > 0:
                    trades.append(
                        {
                            "entry_price": entry_price,
                            "exit_price": px,
                            "return": (px - entry_price) / entry_price,
                        }
                    )
                shares = 0.0
                entry_price = None
        equity[i] = cash_bal + shares * float(prices[i])

    rets = np.diff(equity) / equity[:-1]
    metrics = {
        "trades": len(trades),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "max_drawdown": max_drawdown(equity),
        "avg_win": float(np.mean([t["return"] for t in trades if t["return"] > 0]) or 0),
        "avg_loss": float(np.mean([t["return"] for t in trades if t["return"] <= 0]) or 0),
    }
    return trades, equity, metrics


def paper_rebalance(api_key: str, secret: str, signal: str) -> None:
    trading = TradingClient(api_key, secret, paper=True)
    positions = {p.symbol: p for p in trading.get_all_positions()}
    invested = SYMBOL in positions

    if signal == "long" and not invested:
        order = MarketOrderRequest(
            symbol=SYMBOL,
            notional=STARTING_CASH,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trading.submit_order(order)
        print(f"{datetime.now()}: submitted BUY notional ${STARTING_CASH:.0f} {SYMBOL}")
    elif signal == "flat" and invested:
        trading.close_position(SYMBOL)
        print(f"{datetime.now()}: closed {SYMBOL} position")
    else:
        print(f"{datetime.now()}: hold ({signal}, invested={invested})")


def main() -> None:
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not secret:
        raise SystemExit(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment (paper keys)."
        )

    closes = load_daily_closes(api_key, secret, SYMBOL)
    trades, equity, metrics = backtest_local(closes)
    print(
        f"Local SPY 20/50 backtest on Alpaca history | trades={metrics['trades']} | "
        f"Sharpe={metrics['sharpe']:.3f} | Sortino={metrics['sortino']:.3f} | "
        f"max drawdown={metrics['max_drawdown']:.2%} | "
        f"avg win={metrics['avg_win']:.2%} | avg loss={metrics['avg_loss']:.2%}"
    )

    signal = sma_signal(closes)
    mode = os.environ.get("ALPACA_MODE", "once").lower()
    if mode == "loop":
        while True:
            closes = load_daily_closes(api_key, secret, SYMBOL)
            signal = sma_signal(closes)
            paper_rebalance(api_key, secret, signal)
            time.sleep(60 * 60)
    else:
        paper_rebalance(api_key, secret, signal)


if __name__ == "__main__":
    main()
