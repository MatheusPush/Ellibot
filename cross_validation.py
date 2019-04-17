import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold

source = pd.read_csv('datasets3/64norm_btc_5m.csv').values.tolist()

# Shuffle data
np.random.seed(333)
np.random.shuffle(source)

x = np.array([x[:-1] for x in source])
y = np.array([x[-1:] for x in source])
y = y.reshape((1, len(y)))[0]

# Preproccess
y = tf.keras.utils.to_categorical(y, num_classes=2)

del source

# Cross-Validation
kf = KFold(5)

batch = 128

while batch <= 32768:

	neurons = 256

	while neurons <= 2048:

		title = "b" + str(batch) + "_n" + str(neurons)

		print(title + "\n")

		oos_y = []
		oos_pred = []
		fold = 0

		for train, test in kf.split(x):
			fold += 1
			print('Fold: ' + str(fold))

			x_train = x[train]
			x_test = x[test]
			y_train = y[train]
			y_test = y[test]

			model = tf.keras.models.Sequential()  # a basic feed-forward model
			model.add(tf.keras.layers.Dense(neurons, activation=tf.nn.relu, input_dim=len(x[0])))
			model.add(tf.keras.layers.Dense(neurons, activation=tf.nn.relu))
			model.add(tf.keras.layers.Dense(neurons, activation=tf.nn.relu))
			model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

			model.compile(
				loss='sparse_categorical_crossentropy',
				optimizer='Adam',
				metrics=['accuracy']
			)

			monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=50, verbose=1,
													   mode='auto')

			model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor], epochs=100, verbose=0,
					  batch_size=batch)

			pred = model.predict(x_test)

			oos_y.append(y_test)
			oos_pred.append(np.argmax(pred, axis=1))

			score = metrics.accuracy_score(y_test, np.argmax(pred, axis=1))
			print('Fold score (acc): ' + format(score, '.2f'))

		# All dataset
		oos_y = np.concatenate(oos_y)
		oos_pred = np.concatenate(oos_pred)

		score = metrics.accuracy_score(oos_y, oos_pred)
		print('All data score (acc): ' + format(score, '.2f'))

		neurons += neurons

	batch += batch
