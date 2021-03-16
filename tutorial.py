import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

# arr = [[3.2493553e-08, 3.0776000e-09, 6.1392197e-06, 3.6275067e-04,
#         1.3438174e-12, 1.2900493e-07, 3.3465957e-13, 9.9962962e-01,
#         1.3711252e-07, 1.2309121e-06],
#        [4.6379187e-10, 7.2301715e-05, 9.9991941e-01, 7.7967316e-06,
#         3.6743015e-14, 3.8650910e-08, 2.1432231e-07, 1.0403410e-11,
#         2.6735415e-07, 1.5825274e-13],
#        [3.1201697e-08, 9.9847680e-01, 6.6532615e-05, 1.1636662e-06,
#         1.8259542e-04, 2.2055532e-05, 1.1267507e-05, 1.1076875e-03,
#         1.3161416e-04, 2.6729350e-07],
#        [9.9982601e-01, 3.7892041e-09, 5.3324198e-05, 1.5917327e-06,
#         1.3992540e-07, 1.1159487e-05, 3.0220517e-05, 5.2493528e-05,
#         8.5172459e-08, 2.5104226e-05],
#        [2.0598497e-06, 1.5107783e-08, 2.3527196e-05, 6.1602996e-08,
#         9.9799818e-01, 1.3464137e-07, 1.5736434e-06, 2.4599841e-04,
#         2.5011952e-06, 1.7259006e-03]]
#
# print("<tf.Tensor: shape=(5, 10), dtype=float32, numpy=array({}, dtype=float32)>", arr)
