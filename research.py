# QuantBook research notebook helper (run inside QuantConnect Research).
# Bollinger Band chart on SPY for exploratory analysis.

qb = QuantBook()
spy = qb.add_equity("SPY")
history = qb.history(qb.securities.keys(), 360, Resolution.DAILY)

bbdf = qb.indicator(BollingerBands(30, 2), spy.symbol, 360, Resolution.DAILY)
bbdf.drop("standarddeviation", axis=1).plot()
