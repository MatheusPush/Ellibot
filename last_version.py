import time
import datetime
from binance.client import Client
import tensorflow as tf
import telebot
import dataset_builder as db

API_KEY = '-'
API_SECRET = '-'
PAR = 'BTCUSDT'
BOT_API_TOKEN = '-'
CHANNEL_ID = 0

bot = telebot.TeleBot(token=BOT_API_TOKEN)

client = Client(API_KEY, API_SECRET)

net = tf.keras.models.load_model('models/64norm_btc_full_1024N_eluA_AdamO_mean_squared_logarithmic_errorL_1024B.model')

usd = 100
btc = 0
in_btc = True
th_buy = 1.1
th_sell = 0.85
buy_value = 0

print('STARTED!')

while True:

	now = datetime.datetime.today()
	future = datetime.datetime(now.year, now.month, now.day, now.hour, now.minute - (now.minute % 30),
							   second=10) + datetime.timedelta(minutes=30)

	# aguardar proximo candle de 30min
	time.sleep((future - now).seconds)

	src = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_30MINUTE, '15 days ago')
	src.pop()
	sub_src = src[-664:]

	x = db.build(sub_src)
	predict = net.predict(x)
	is_top, is_bot = db.last_moment(sub_src)

	now = sub_src.pop()
	t = datetime.datetime.fromtimestamp(int(now[0]) / 1000).isoformat()
	c = float(now[4])

	print(str(t) + '\tOk Close: ' + str(c) + ' - Pred Bot: ' + str(predict[0, 1]) + ' - Pred Top: ' + str(
		predict[0, 0]) + ' - Is Bot: ' + str(is_bot) + ' - Is Top: ' + str(is_top))

	# 1. nao comprado
	# 2. ultimo sinal de fundo
	# 3. tendencia de alta > th_buy
	if not in_btc and is_bot and predict[0, 1] >= th_buy:
		btc = usd / c * 0.999
		buy_value = c
		reply = "\t↗ BUY BTC @ USDT " + format(c, '.2f')
		print(str(t) + reply + "\n\t\t\t\t\tCURRENT BALANCE : BTC " + format(btc, '.6f'))
		bot.send_message(CHANNEL_ID, reply)
		in_btc = True
	# 1. comprado
	# 2. ultimo sinal de topo
	# 3. tendencia de baixa > th_sell
	elif in_btc and is_top and predict[0, 0] >= th_sell:
		usd = btc * c * 0.999
		profit = (c / buy_value - 1) * 100
		reply = "\t↘ SELL BTC @ USDT " + format(c, '.2f') + "\n   PROFIT: " + format(profit, '.2f') + "%"
		print(str(t) + reply + "\n\t\t\t\t\tCURRENT BALANCE : USDT " + format(usd, '.2f'))
		bot.send_message(CHANNEL_ID, reply)
		in_btc = False

bot.polling(none_stop=True)
