import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import numpy as np
import pandas as pd

datas = ['32norm_btc_5m', '64norm_btc_5m', '96norm_btc_5m']

for d in datas:

	source = pd.read_csv('datasets3/' + d + '.csv').values.tolist()

	np.random.seed(333)
	np.random.shuffle(source)

	x = np.array([x[:-1] for x in source])
	y = np.array([x[-1:] for x in source])
	y = y.reshape((1, len(y)))[0]

	y2 = tf.keras.utils.to_categorical(y, num_classes=2)

	del source

	best_acc = 0
	best = ''
	o = 'Adam'
	i = 512

	for a in ['selu', 'elu', 'softmax']:
		for l in ['mean_squared_logarithmic_error', 'mean_squared_error', 'cosine_proximity']:

			title = d + '_' + str(i) + 'N_' + a + 'A_' + o + 'O_' + l + 'L_' + '1024B'

			print('\n\n' + title)

			model = tf.keras.models.Sequential()  # a basic feed-forward model
			model.add(tf.keras.layers.Dense(i, activation=tf.nn.relu, input_dim=len(x[0])))
			model.add(tf.keras.layers.Dense(i, activation=tf.nn.relu))
			model.add(tf.keras.layers.Dense(i, activation=tf.nn.relu))
			model.add(tf.keras.layers.Dense(2, activation=a))

			model.compile(
				loss=l,
				optimizer=o,
				metrics=['accuracy']
			)

			try:

				model.fit(x, y2, epochs=95, validation_split=0.2, verbose=0, batch_size=1024)

				model.save('models/' + title + '.model')

				val_loss, val_acc = model.evaluate(x, y2, verbose=0, batch_size=1024)
				print("Total LOSS:\t\t" + str(val_loss))  # model's loss (error)
				print("Total ACC:\t\t" + str(val_acc))  # model's accuracy

				if val_acc > best_acc:
					best_acc = val_acc
					best = title

			except:

				model.fit(x, y, epochs=95, validation_split=0.2, verbose=0, batch_size=1024)

				model.save('models/' + title + '.model')

				val_loss, val_acc = model.evaluate(x, y, verbose=0, batch_size=1024)
				print("Total LOSS:\t\t" + str(val_loss))  # model's loss (error)
				print("Total ACC:\t\t" + str(val_acc))  # model's accuracy

				if val_acc > best_acc:
					best_acc = val_acc
					best = title

			print('\n\t\t\t\tThe best is: ' + best + '\n')
