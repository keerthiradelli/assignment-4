Python 3.6.7 (v3.6.7:6ec5cf24b7, Oct 20 2018, 13:35:33) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> Python 3.6.7 (v3.6.7:6ec5cf24b7, Oct 20 2018, 13:35:33) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy
>>> from keras.datasets import mnist
Using TensorFlow backend.
>>> from keras.models import Sequential
>>> from keras.layers import Dense
>>> from keras.layers import Dropout
>>> from keras.layers import Flatten
>>> from keras.layers.convolutional import Conv2D
>>> from keras.layers.convolutional import MaxPooling2D
>>> from keras.utils import np_utils
>>> from keras import backend as K
>>> K.set_image_dim_ordering('th')
>>> # fix random seed for reproducibility
... seed = 7
>>> numpy.random.seed(seed)
>>> # load data
... (X_train, y_train), (X_test, y_test) = mnist.load_data()
>>> # reshape to be [samples][pixels][width][height]
... X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
>>> X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
>>> # normalize inputs from 0-255 to 0-1
... X_train = X_train / 255
>>> X_test = X_test / 255
>>> # one hot encode outputs
... y_train = np_utils.to_categorical(y_train)
>>> y_test = np_utils.to_categorical(y_test)
>>> num_classes = y_test.shape[1]
>>> def baseline_model():
...   # create model
...   model = Sequential()
...   model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
...   model.add(MaxPooling2D(pool_size=(2, 2)))
...   model.add(Dropout(0.2))
...   model.add(Flatten())
...   model.add(Dense(128, activation='relu'))
...   model.add(Dense(128, activation='relu'))
...   model.add(Dense(num_classes, activation='softmax'))
...   # Compile model
...   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
...   return model
...
>>> # build the model
... model = baseline_model()
WARNING:tensorflow:From C:\Users\Keerthi\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-06-17 23:04:55.217456: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
WARNING:tensorflow:From C:\Users\Keerthi\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
>>> # Fit the model
... model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
WARNING:tensorflow:From C:\Users\Keerthi\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
 - 52s - loss: 0.2448 - acc: 0.9289 - val_loss: 0.0799 - val_acc: 0.9754
Epoch 2/10
 - 51s - loss: 0.0696 - acc: 0.9789 - val_loss: 0.0540 - val_acc: 0.9829
Epoch 3/10
 - 51s - loss: 0.0485 - acc: 0.9848 - val_loss: 0.0446 - val_acc: 0.9857
Epoch 4/10
 - 51s - loss: 0.0367 - acc: 0.9887 - val_loss: 0.0388 - val_acc: 0.9874
Epoch 5/10
 - 58s - loss: 0.0297 - acc: 0.9908 - val_loss: 0.0384 - val_acc: 0.9870
Epoch 6/10
 - 55s - loss: 0.0249 - acc: 0.9918 - val_loss: 0.0358 - val_acc: 0.9883
Epoch 7/10
 - 56s - loss: 0.0209 - acc: 0.9933 - val_loss: 0.0283 - val_acc: 0.9905
Epoch 8/10
 - 55s - loss: 0.0164 - acc: 0.9945 - val_loss: 0.0323 - val_acc: 0.9900
Epoch 9/10
 - 57s - loss: 0.0153 - acc: 0.9948 - val_loss: 0.0333 - val_acc: 0.9894
Epoch 10/10
 - 54s - loss: 0.0129 - acc: 0.9955 - val_loss: 0.0340 - val_acc: 0.9887
<keras.callbacks.History object at 0x0000021F3FA1D780>
>>> # Final evaluation of the model
... scores = model.evaluate(X_test, y_test, verbose=0)
>>> print("CNN Error: %.2f%%" % (100-scores[1]*100))
CNN Error: 1.13%
>>>
