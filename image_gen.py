import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gen_graph(history, title):
	title = title

	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Accuracy ' + title)
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig('imgs/' + title + ' acc.png')
	plt.close()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Loss ' + title)
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig('imgs/' + title + ' loss.png')
	plt.close()


source = pd.read_csv('datasets3/32norm_btc_5m.csv').values.tolist()[80000:]

np.random.seed(333)
np.random.shuffle(source)

x = np.array([x[:-1] for x in source])
y = np.array([x[-1:] for x in source])
y = y.reshape((1, len(y)))[0]

y2 = tf.keras.utils.to_categorical(y, num_classes=2)

del source

best_acc = 0
best = ''
i = 512

for a in ['softmax',
		  'elu',
		  'selu',
		  'softplus',
		  'sigmoid',
		  'hard_sigmoid']:
	for o in ['Adam']:
		for l in ['mean_squared_error']:  # sparse_categorical_crossentropy

			title = a + ' - ' + o + ' - ' + l

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

				history_rmsprop = model.fit(x, y2, epochs=100, validation_split=0.2, verbose=0, batch_size=1024)

				val_loss, val_acc = model.evaluate(x, y2, verbose=0, batch_size=1024)
				print("Total LOSS:\t\t" + str(val_loss))  # model's loss (error)
				print("Total ACC:\t\t" + str(val_acc))  # model's accuracy

				if val_acc > best_acc:
					best_acc = val_acc
					best = title

			except:

				history_rmsprop = model.fit(x, y, epochs=100, validation_split=0.2, verbose=0, batch_size=1024)

				val_loss, val_acc = model.evaluate(x, y, verbose=0, batch_size=1024)
				print("Total LOSS:\t\t" + str(val_loss))  # model's loss (error)
				print("Total ACC:\t\t" + str(val_acc))  # model's accuracy

				if val_acc > best_acc:
					best_acc = val_acc
					best = title

			print('\n\t\t\t\tThe best is: ' + best + '\n')

			# plot the accuracy
			gen_graph(history_rmsprop, title)
