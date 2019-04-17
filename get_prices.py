from binance.client import Client
import pandas as pd

c = Client('','')

par = ['BTCUSDT']

#sma 600 retira:
#	5m	= 50h	= 2d + 1h
# 	15m	= 150h	= 6d + 6h
#	1h	= 25d
#	4h	= 100d	= 3M + 9d

src = c.get_historical_klines(par[0], Client.KLINE_INTERVAL_15MINUTE, '2018-06-21', '2018-11-28')

src = [x[0:6] for x in src]

pd.DataFrame(src).to_csv("prices/backtest_15m.csv", sep=',', index=False)