# QuantConnect LEAN algorithm: SMA Monte Carlo backtester.
# Paste this as main.py in a QuantConnect project, or run with the LEAN CLI.

from AlgorithmImports import *
import numpy as np


class SmaMonteCarloBacktester(QCAlgorithm):
    """Historical SPY 20/50 SMA backtest plus 1,000+ one-year Monte Carlo path sims."""

    def Initialize(self):
        self.SetStartDate(2012, 1, 1)
        self.SetEndDate(2022, 1, 1)
        self.SetCash(10000)

        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.fast = self.SMA(self.symbol, 20, Resolution.Daily)
        self.slow = self.SMA(self.symbol, 50, Resolution.Daily)
        self.SetWarmUp(50)

        # Trade log for the historical LEAN backtest (entry / exit / return).
        self.trades = []
        self._open_entry = None

        # Equity samples for historical risk metrics.
        self._equity_curve = []

        self.Schedule.On(
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.AfterMarketOpen(self.symbol, 1),
            self._record_equity,
        )

        self._run_monte_carlo()

    def _record_equity(self):
        if self.IsWarmingUp:
            return
        self._equity_curve.append(float(self.Portfolio.TotalPortfolioValue))

    def OnData(self, data: Slice):
        if self.IsWarmingUp or not data.Bars.ContainsKey(self.symbol):
            return
        if not self.fast.IsReady or not self.slow.IsReady:
            return

        price = float(data.Bars[self.symbol].Close)
        invested = self.Portfolio[self.symbol].Invested

        if self.fast.Current.Value > self.slow.Current.Value:
            if not invested:
                self.SetHoldings(self.symbol, 1.0)
                self._open_entry = {"time": self.Time, "price": price}
        else:
            if invested:
                self.Liquidate(self.symbol)
                if self._open_entry is not None and self._open_entry["price"] > 0:
                    entry = self._open_entry["price"]
                    ret = (price - entry) / entry
                    self.trades.append(
                        {
                            "entry_time": self._open_entry["time"],
                            "exit_time": self.Time,
                            "entry_price": entry,
                            "exit_price": price,
                            "return": ret,
                        }
                    )
                self._open_entry = None

    def OnEndOfAlgorithm(self):
        self._log_trade_summary(self.trades, label="Historical SPY backtest")
        self._log_risk_metrics(self._equity_curve, label="Historical SPY backtest")

    def _run_monte_carlo(self):
        hist = self.History[TradeBar](self.symbol, 252 * 2, Resolution.Daily)
        closes = [float(bar.Close) for bar in hist]
        if len(closes) < 60:
            self.Debug("Not enough SPY history for Monte Carlo.")
            return

        log_returns = np.diff(np.log(np.asarray(closes, dtype=float)))
        mu = float(np.mean(log_returns))
        sigma = float(np.std(log_returns, ddof=1))

        n_paths = 1000
        horizon = 252
        rng = np.random.default_rng(42)

        all_trades = []
        terminal_equity = []
        sharpes = []
        sortinos = []
        max_dds = []

        for _ in range(n_paths):
            shocks = rng.normal(mu, sigma, horizon)
            path = np.exp(np.cumsum(shocks))
            path = path / path[0] * closes[-1]

            trades, equity = self._simulate_sma_path(path, fast=20, slow=50, cash=10000.0)
            all_trades.extend(trades)
            terminal_equity.append(equity[-1] / equity[0] - 1.0)

            rets = np.diff(equity) / equity[:-1]
            sharpes.append(self._sharpe(rets))
            sortinos.append(self._sortino(rets))
            max_dds.append(self._max_drawdown(equity))

        self.Debug(
            f"Monte Carlo SPY: {n_paths} one-year paths | "
            f"mean terminal return={np.mean(terminal_equity):.2%} | "
            f"5%={np.percentile(terminal_equity, 5):.2%} | "
            f"95%={np.percentile(terminal_equity, 95):.2%}"
        )
        self._log_trade_summary(all_trades, label=f"Monte Carlo ({n_paths} paths)")
        self.Debug(
            f"Monte Carlo risk | Sharpe={np.nanmean(sharpes):.3f} | "
            f"Sortino={np.nanmean(sortinos):.3f} | "
            f"max drawdown={np.nanmean(max_dds):.2%}"
        )

    @staticmethod
    def _simulate_sma_path(prices, fast=20, slow=50, cash=10000.0):
        prices = np.asarray(prices, dtype=float)
        n = len(prices)
        equity = np.empty(n, dtype=float)
        cash_bal = cash
        shares = 0.0
        trades = []
        entry_price = None

        for i in range(n):
            if i >= slow - 1:
                f = float(np.mean(prices[i - fast + 1 : i + 1]))
                s = float(np.mean(prices[i - slow + 1 : i + 1]))
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

        if shares > 0:
            px = float(prices[-1])
            cash_bal = shares * px
            if entry_price and entry_price > 0:
                trades.append(
                    {
                        "entry_price": entry_price,
                        "exit_price": px,
                        "return": (px - entry_price) / entry_price,
                    }
                )
            equity[-1] = cash_bal

        return trades, equity

    def _log_trade_summary(self, trades, label):
        if not trades:
            self.Debug(f"{label}: no completed trades logged.")
            return
        rets = np.array([t["return"] for t in trades], dtype=float)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        self.Debug(
            f"{label}: {len(trades)} trades | "
            f"avg win={wins.mean() if len(wins) else 0:.2%} | "
            f"avg loss={losses.mean() if len(losses) else 0:.2%}"
        )

    @staticmethod
    def _sharpe(returns, periods_per_year=252):
        returns = np.asarray(returns, dtype=float)
        if returns.size < 2 or np.std(returns, ddof=1) == 0:
            return float("nan")
        return float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(periods_per_year))

    @staticmethod
    def _sortino(returns, periods_per_year=252):
        returns = np.asarray(returns, dtype=float)
        downside = returns[returns < 0]
        if returns.size < 2 or downside.size == 0 or np.std(downside, ddof=1) == 0:
            return float("nan")
        return float(np.mean(returns) / np.std(downside, ddof=1) * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown(equity):
        equity = np.asarray(equity, dtype=float)
        if equity.size == 0:
            return float("nan")
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        return float(dd.min())

    def _log_risk_metrics(self, equity_curve, label):
        if len(equity_curve) < 3:
            self.Debug(f"{label}: not enough equity samples for risk metrics.")
            return
        equity = np.asarray(equity_curve, dtype=float)
        rets = np.diff(equity) / equity[:-1]
        self.Debug(
            f"{label} risk | Sharpe={self._sharpe(rets):.3f} | "
            f"Sortino={self._sortino(rets):.3f} | "
            f"max drawdown={self._max_drawdown(equity):.2%}"
        )
