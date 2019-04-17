import pandas as pd
import datetime
import dataset_builder as db
import numpy as np
import tensorflow as tf
import telebot
from binance.client import Client

PAR = 'BTCUSDT'
BOT_API_TOKEN = ''
CHANNEL_ID = 0
API_KEY = ''
API_SECRET = ''
client = Client(API_KEY, API_SECRET)

# bot = telebot.TeleBot(token=BOT_API_TOKEN)

# src = pd.read_csv('prices/backtest_15m.csv').values.tolist()[9500:] # OCT & NOV
# src = pd.read_csv('prices/backtest_15m.csv').values.tolist()[12000:] # NOV
# src = pd.read_csv('prices/backtest_30m.csv').values.tolist()[4550:]  # OCT & NOV
# src = pd.read_csv('prices/backtest_30m.csv').values.tolist()[6000:] # NOV
# src = pd.read_csv('prices/backtest_1h.csv').values.tolist()[2200:] # OCT & NOV
# src = pd.read_csv('prices/backtest_1h.csv').values.tolist()[2950:] # NOV
net = tf.keras.models.load_model('models/64norm_btc_full_1024N_eluA_AdamO_mean_squared_logarithmic_errorL_1024B.model')

src = client.get_historical_klines(PAR, Client.KLINE_INTERVAL_30MINUTE, '2018-11-10')

time = [datetime.datetime.fromtimestamp(int(x[0]) / 1000).isoformat() for x in src]
close = [x[4] for x in src]

usd = 100
btc = 0
in_btc = False
th_buy = 1.1
th_sell = 0.85
buy_value = 3790

for i in range(664, len(src)):
	sub_src = src[i - 664:i]
	x = db.build(sub_src)
	predict = net.predict(x)
	is_top, is_bot = db.last_moment(sub_src)
	t = time[i - 1]
	c = float(close[i - 1])

	# print(str(t) + '\tOk Close: ' + str(c) + ' - Pred Bot: ' + str(predict[0, 1]) + ' - Pred Top: ' + str(
	# 	predict[0, 0]) + ' - Is Bot: ' + str(is_bot) + ' - Is Top: ' + str(is_top))

	# 1. nao comprado
	# 2. ultimo sinal de fundo
	# 3. tendencia de alta > th_buy
	if not in_btc and is_bot and predict[0, 1] >= th_buy:
		btc = usd / c * 0.999
		buy_value = c
		reply = "\t↗ BUY BTC @ USDT " + format(c, '.2f')
		print(str(t) + reply + "\n\t\t\t\t\tCURRENT BALANCE : BTC " + format(btc, '.6f'))
		# bot.send_message(CHANNEL_ID, reply)
		in_btc = True
	# 1. comprado
	# 2. ultimo sinal de topo
	# 3. tendencia de baixa > th_sell
	elif in_btc and is_top and predict[0, 0] >= th_sell:
		usd = btc * c * 0.999
		profit = (c / buy_value - 1) * 100
		reply = "\t↘ SELL BTC @ USDT " + format(c, '.2f') + "\n   PROFIT: " + format(profit, '.2f') + "%"
		print(str(t) + reply + "\n\t\t\t\t\tCURRENT BALANCE : USDT " + format(usd, '.2f'))
		# bot.send_message(CHANNEL_ID, reply)
		in_btc = False

# bot.polling(none_stop=True)
