
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
new_model = tf.keras.models.load_model('SavedModel.h5')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

new_predictions = new_model.predict(x_test)

for i in range(5):
  n = np.random.randint(low=1, high= 10000)  
  plt.imshow(x_test[n], cmap='Greys')
  predict = new_predictions[n].tolist().index(max(new_predictions[n].tolist()))
  print("El resultado es ", predict)
  plt.show()

