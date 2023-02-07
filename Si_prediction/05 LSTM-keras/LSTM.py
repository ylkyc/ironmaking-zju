import pandas as pd
from utils import *
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from metrics import *
from keras import optimizers
import keras.backend as K
from keras.callbacks import LearningRateScheduler, EarlyStopping


def scheduler(epoch):
    if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.7)
        print("lr changed to {}".format(lr * 0.7))
    return K.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)

dataset_file = 'dataset.csv'
keep_prob = 0.2
layers = [128, 128, 64]
batchsize = 32
epochs = 500
train_size = 2632
valid_size = 877
test_size = 877
seq_length = 10
learning_rate = 0.0001

dataset = pd.read_csv(dataset_file)
x_train, x_valid, x_test, \
y_train, y_valid, y_test, data_min, data_max = \
    load_timeseries(dataset,seq_length, train_size, test_size, valid_size)

model = Sequential()
model.add(LSTM(layers[0],
               activation='tanh',
               input_shape=(x_train.shape[1], x_train.shape[2]),
               return_sequences=True))
model.add(Dropout(keep_prob))
model.add(LSTM(layers[1],
               activation='tanh'))
model.add(Dropout(keep_prob))
model.add(Dense(units=layers[2], activation='tanh'))
model.add(Dense(units=1))
adam = optimizers.Adam(lr=learning_rate, epsilon=None, amsgrad=False)
model.compile(loss='mse', optimizer=adam)
early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='min')
# fit network
LSTM = model.fit(x_train, y_train,
                 epochs=epochs,
                 batch_size=batchsize,
                 validation_data=(x_valid, y_valid),
                 verbose=2,
                 shuffle=False,
                 callbacks=[reduce_lr, early_stop])
# plot history
plt.plot(LSTM.history['loss'], label='train')
plt.plot(LSTM.history['val_loss'], label='valid')
plt.legend()
plt.show()

train_predict = model.predict(x_train)
valid_predict = model.predict(x_valid)
test_predict = model.predict(x_test)

train_predict_re = reverse_normalization(train_predict, data_max[-1], data_min[-1])
valid_predict_re = reverse_normalization(valid_predict, data_max[-1], data_min[-1])
test_predict_re = reverse_normalization(test_predict, data_max[-1], data_min[-1])

y_train_re = reverse_normalization(y_train, data_max[-1], data_min[-1])
y_valid_re = reverse_normalization(y_valid, data_max[-1], data_min[-1])
y_test_re = reverse_normalization(y_test, data_max[-1], data_min[-1])

test_HR_re = HR(test_predict_re, y_test_re)
print('The HR indicator on the test set is: ', test_HR_re)
test_mse_re = mse(test_predict_re, y_test_re)
print('The mse on the test set is: ', test_mse_re)

plt.figure(figsize=(20, 3))
plt.plot(y_train_re, label='ground truth')
plt.plot(train_predict_re, label='predict')
plt.tick_params(labelsize=12)
plt.xlabel('NO.', size=13)
plt.ylabel('train', size=15)
plt.legend()
plt.show()

plt.figure(figsize=(20, 3))
plt.plot(y_valid_re, label='ground truth')
plt.plot(valid_predict_re, label='predict')
plt.tick_params(labelsize=12)
plt.xlabel('NO.', size=13)
plt.ylabel('validation', size=15)
plt.legend()
plt.show()

plt.figure(figsize=(20, 3))
plt.plot(y_test_re, label='ground truth')
plt.plot(test_predict_re, label='predict')
plt.tick_params(labelsize=12)
plt.xlabel('NO.', size=13)
plt.ylabel('test', size=15)
plt.legend()
plt.show()