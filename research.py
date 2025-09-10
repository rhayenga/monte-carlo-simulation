# QuantBook Analysis Tool
qb = QuantBook()
spy = qb.add_equity("RL")
history = qb.history(qb.securities.keys(), 360, Resolution.DAILY)

# Indicator Analysis
bbdf = qb.indicator(BollingerBands(30, 2), RL.symbol, 360, Resolution.DAILY)
bbdf.drop('standarddeviation', axis=1).plot()