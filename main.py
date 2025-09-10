# This allows me to access QuantConnect's Lean API
from AlgorithmImports import *
import numpy as np
#Start of class because it needs to be in a QCAlrgorithm to work
class SimpleMonteCarlo(QCAlgorithm):

    def Initialize(self):
        # Starting with 10K from 2012-2022 because 2022 is the lastest data on QC
        self.SetStartDate(2012, 1, 1)
        self.SetEndDate(2022, 1, 1)
        self.SetCash(10000)

        #I selected Ralph Lauuren as the equity but any works here. 
        self.symbol = self.AddEquity("RL", Resolution.Daily).Symbol

        # I did about 2 years of history (enough for my desired estimation)
        hist = self.History[TradeBar](self.symbol, 252 * 2, Resolution.Daily)
        closes = [bar.Close for bar in hist]

        if len(closes) >= 3:
            # This is my Monte Carlo estimate alrogirthm 
            log_returns = np.diff(np.log(closes))
            mu, sigma = np.mean(log_returns), np.std(log_returns)

            rng = np.random.default_rng(42)
            final_vals = []
            for _ in range(100):
                sim = rng.normal(mu, sigma, 252)
                path = np.exp(np.cumsum(sim))
                path /= path[0]
                final_vals.append(path[-1])

            # Log only the essential results 
            self.Debug(f"Monte Carlo RL (1y) → mean: {np.mean(final_vals):.2f}, "
                       f"5%: {np.percentile(final_vals, 5):.2f}, "
                       f"95%: {np.percentile(final_vals, 95):.2f}")
        else:
            self.Debug("Not enough history for Monte Carlo.")

        # Simple strategy: Simple moving average crossover (just to plot on equity curve)
        self.fast = self.SMA(self.symbol, 10, Resolution.Daily)
        self.slow = self.SMA(self.symbol, 50, Resolution.Daily)
        self.SetWarmUp(50)

    def OnData(self, data: Slice):
        if self.IsWarmingUp: 
            return

        if self.fast.Current.Value > self.slow.Current.Value:
            if not self.Portfolio[self.symbol].Invested:
                self.SetHoldings(self.symbol, 1.0)
        else:
            if self.Portfolio[self.symbol].Invested:
                self.Liquidate(self.symbol)
print("\nSaved: main.py.csv and research.ipynb")
#RowanHayenga 2025
