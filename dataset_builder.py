from binance.client import Client
import numpy as np
import datetime
import pandas as pd
import sys


def build_sma(src, periods=[20]):
	smas = []
	for p in periods:
		sma = ([0] * (p - 1)) + movingaverage(src, p)
		sma = sma[start:]
		smas.append(sma)
	return smas


def build_rsi(close, periods=[14]):
	rsis = []
	for p in periods:
		r = rsi(close, p)
		r = r[start:]
		rsis.append(r)
	return rsis


def movingaverage(values, window):
	weigths = np.repeat(1.0, window) / window
	smas = np.convolve(values, weigths, 'valid')
	return smas.tolist()  # as a numpy array


def ema(values, window):
	df = pd.DataFrame({'close': values})

	e = df.ewm(span=window, adjust=False).mean()

	return e['close'].tolist()


def rsi(prices, n):
	deltas = np.diff(prices)
	seed = deltas[:n + 1]
	up = seed[seed >= 0].sum() / n
	down = -seed[seed < 0].sum() / n
	rs = up / down
	rsi = np.zeros_like(prices)
	rsi[:n] = 100. - 100. / (1. + rs)

	for i in range(n, len(prices)):
		delta = deltas[i - 1]  # cause the diff is 1 shorter

		if delta > 0:
			upval = delta
			downval = 0.
		else:
			upval = 0.
			downval = -delta

		up = (up * (n - 1) + upval) / n
		down = (down * (n - 1) + downval) / n

		rs = up / down
		rsi[i] = 100. - 100. / (1. + rs)

	return rsi.tolist()


def last_moment(src):
	close = [float(x[4]) for x in src]
	top, bottom = top_bottoms(close, False)
	is_top = False
	is_bot = False
	for i in range(len(top) - 1, 0, -1):
		if top[i] > 0:
			is_top = True
		if bottom[i] > 0:
			is_bot = True
		if is_top or is_bot:
			break
	return is_top, is_bot


def top_bottoms(close, zero=True):
	top = [0] * len(close)
	bottom = [0] * len(close)
	moments = [0] * len(close)

	for i in range(len(close), 0, -1):
		init = i - window if i > window else 0
		high_id = np.argmax(close[init:i]) + init
		low_id = np.argmin(close[init:i]) + init
		top[high_id] += 1
		bottom[low_id] += 1

	if not zero:
		return np.asarray(top), np.asarray(bottom)

	dw = window ** 2

	for i in range(len(close)):
		# Gerar moments de -1 a 1
		moments[i] = (top[i] ** 2 / dw) - (bottom[i] ** 2 / dw)
		if moments[i] != 1 and moments[i] != -1:
			moments[i] = 0

	# fix

	last_index = 0
	last_moment = moments[0]
	for i in range(len(close)):
		if moments[i] != 0:
			if moments[i] == last_moment:
				# fix double moments
				if last_moment == 1:  # top
					new_min = np.argmin(close[last_index:i + 1])
					moments[new_min + last_index] = -1
				elif last_moment == -1:  # bot
					new_max = np.argmax(close[last_index:i + 1])
					moments[new_max + last_index] = 1

			last_index = i
			last_moment = moments[i]

	return moments


def build_trends(close):
	tb = top_bottoms(close)
	trends = [0] * len(close)
	trend = 0

	for i in range(window):
		trends[i] = -1
		if tb[i] != 0:
			trend = 0 if tb[i] == 1 else 1

	for i in range(window, len(close) - window):
		trends[i] = trend
		if tb[i] != 0:
			trend = 0 if tb[i] == 1 else 1

	for i in range(len(close) - window, len(close)):
		trends[i] = -1

	return trends


# client = Client("", "")
window = 32  # Used to find tops & bottons
periods = 64  # No. of candles to be calculated and used
max_sma = 600  # Last SMA calculated in net
start = max_sma


def build(src):
	# Slice source
	open = [float(x[1]) for x in src]
	high = [float(x[2]) for x in src]
	low = [float(x[3]) for x in src]
	close = [float(x[4]) for x in src]
	volume = [float(x[5]) for x in src]

	# Calc datas
	volume20 = build_sma(volume)[0]
	smas = build_sma(close, [20, 60, 200, 600])
	rsis = build_rsi(close, [2, 3, 14])

	# Slice source datas
	open = open[start:]
	low = low[start:]
	high = high[start:]
	close = close[start:]
	volume = volume[start:]

	del src

	# building the open pack
	for i in range(len(open) - periods + 1):
		_aux = []
		for j in range(i, i + periods):
			_aux.append(open[j])
		open[i] = _aux
	open = open[:len(open) - periods + 1]

	# building the high pack
	for i in range(len(high) - periods + 1):
		_aux = []
		for j in range(i, i + periods):
			_aux.append(high[j])
		high[i] = _aux
	high = high[:len(high) - periods + 1]

	# building the low pack
	for i in range(len(low) - periods + 1):
		_aux = []
		for j in range(i, i + periods):
			_aux.append(low[j])
		low[i] = _aux
	low = low[:len(low) - periods + 1]

	# building the close pack
	for i in range(len(close) - periods + 1):
		_aux = []
		for j in range(i, i + periods):
			_aux.append(close[j])
		close[i] = _aux
	close = close[:len(close) - periods + 1]

	# building the volume pack
	for i in range(len(volume) - periods + 1):
		_aux = []
		for j in range(i, i + periods):
			_aux.append(volume[j])
		volume[i] = _aux
	volume = volume[:len(volume) - periods + 1]

	# building the volume20 pack
	for i in range(len(volume20) - periods + 1):
		_aux = []
		for j in range(i, i + periods):
			_aux.append(volume20[j])
		volume20[i] = _aux
	volume20 = volume20[:len(volume20) - periods + 1]

	# building the sma pack
	for k in range(len(smas)):
		for i in range(len(smas[k]) - periods + 1):
			_aux = []
			for j in range(i, i + periods):
				_aux.append(smas[k][j])
			smas[k][i] = _aux
		smas[k] = smas[k][:len(smas[k]) - periods + 1]

	# building the rsi pack
	for k in range(len(rsis)):
		for i in range(len(rsis[k]) - periods + 1):
			_aux = []
			for j in range(i, i + periods):
				_aux.append(rsis[k][j])
				rsis[k][i] = _aux
		rsis[k] = rsis[k][:len(rsis[k]) - periods + 1]

	del _aux

	# join packs

	for i in range(len(open)):
		open[i] += high[i]
		open[i] += low[i]
		open[i] += close[i]
		for sma in smas:
			open[i] += sma[i]
		open[i] += volume[i]
		open[i] += volume20[i]
		for rsi in rsis:
			open[i] += rsi[i]

	del high
	del low
	del close
	del sma
	del smas
	del volume
	del volume20
	del rsi
	del rsis

	price_end_index = 8 * periods  # X dados do campo preço
	volume_end_index = price_end_index + 2 * periods  # X dados do campo volume

	# Normalize
	for i in range(len(open)):

		# prices
		_min = min(open[i][:price_end_index])
		_max = max(open[i][:price_end_index])
		for j in range(0, price_end_index):
			open[i][j] = (open[i][j] - _min) / (_max - _min)

		# volumes
		_min = min(open[i][price_end_index:volume_end_index])
		_max = max(open[i][price_end_index:volume_end_index])
		for j in range(price_end_index, volume_end_index):
			open[i][j] = (open[i][j] - _min) / (_max - _min)

		# rsi
		for j in range(volume_end_index, len(open[i])):
			open[i][j] = open[i][j] / 100

	open = np.asarray(open)  # .pop())

	return open
