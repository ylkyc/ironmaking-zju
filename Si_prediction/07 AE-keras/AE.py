import pandas as pd
from utils import *
from keras.layers import Dense, GRU, TimeDistributed, RepeatVector, Input, Concatenate, Lambda
import matplotlib.pyplot as plt
from metrics import *
from keras import optimizers, Model
import keras.backend as K
from keras.callbacks import LearningRateScheduler, EarlyStopping

dataset_file = 'dataset.csv'
keep_prob = 0.2
encoder_units = [128, 64]
decoder_units = [64, 128]
mlp_units = [128, 1]
batchsize = 128
epochs = 500
train_size = 2632
valid_size = 877
test_size = 877
seq_length = 10
learning_rate = 0.0001
loss_weights = [0.1, 0.9]

dataset = pd.read_csv(dataset_file)
x_train, x_valid, x_test, \
y_train, y_valid, y_test, data_min, data_max = \
    load_timeseries(dataset,seq_length, train_size, test_size, valid_size)


def encoder_decoder(inputs, en_units, de_units):
    x = inputs
    x = GRU(en_units[0],
             activation='tanh',
             input_shape=(inputs.shape[1], inputs.shape[2]),
             return_sequences=True)(x)
    x = GRU(en_units[1], activation='tanh')(x)
    encoder_vector = x
    x = RepeatVector(inputs.shape[1])(x)
    x = GRU(de_units[0], activation='tanh', return_sequences=True)(x)
    x = GRU(de_units[1], activation='tanh', return_sequences=True)(x)
    x = TimeDistributed(Dense(x_train.shape[2]))(x)
    return x, encoder_vector


def mlp(inputs, units):
    x = inputs
    x = Dense(units[0],
              kernel_initializer='glorot_normal',
              activation='tanh',
              input_dim=inputs.shape[1])(x)
    x = Dense(units[1],
              kernel_initializer='glorot_normal',
              activation='tanh')(x)
    return x


def ae_model(inputs, e_units, d_units, m_units):
    x = inputs
    reconstruction, encoder_vector = encoder_decoder(x, e_units, d_units)
    last_step_x = Lambda(lambda x: x[:, -1, :])(x)
    x = Concatenate()([last_step_x, encoder_vector])
    x = mlp(x, m_units)
    return x, reconstruction


def scheduler(epoch):
    if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.8)
        print("lr changed to {}".format(lr * 0.8))
    return K.get_value(model.optimizer.lr)


def customize_mse(x_true, x_recon, W):
    def mse(y_true, y_pred):
        loss_recon = K.mean(K.square(x_true - x_recon))
        loss_pre = K.mean(K.square(y_pred - y_true), axis=-1)
        return W[0]*loss_recon + W[1]*loss_pre
    return mse


reduce_lr = LearningRateScheduler(scheduler)
early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='min')
model_inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
model_output, x_reconstruction = ae_model(model_inputs, encoder_units, decoder_units, mlp_units)
model = Model(model_inputs, model_output)

customize_loss = customize_mse(model_inputs, x_reconstruction, loss_weights)
adam = optimizers.Adam(lr=learning_rate, epsilon=None, amsgrad=False)
model.compile(loss=customize_loss, optimizer=adam)
model_history = model.fit(x_train, y_train,
                          epochs=epochs,
                          batch_size=batchsize,
                          validation_data=(x_valid, y_valid),
                          verbose=2,
                          shuffle=False,
                          callbacks=[reduce_lr, early_stop])


# plot history
plt.plot(model_history.history['loss'], label='train')
plt.plot(model_history.history['val_loss'], label='valid')
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