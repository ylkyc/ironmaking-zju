import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd

dataset_file = 'dataset.csv'
train_size = 3509
test_size = 877


def HR(y_real, y_pre):
    y_real = y_real.reshape(-1)
    y_pre = y_pre.reshape(-1)
    err_temp = np.abs(y_real - y_pre)
    err = np.where(err_temp <= 0.1, 1, 0)
    HR = np.mean(err)*100
    return HR


# Load data
dataset = pd.read_csv(dataset_file)
dataset = np.array(dataset)

data_train = dataset[:train_size]
data_test = dataset[train_size:train_size+test_size]

X_train = data_train[:, :-1]
y_train = data_train[:, -1]

X_test = data_test[:, :-1]
y_test = data_test[:, -1]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)

plt.figure(figsize=(20, 3))    # 定义一个图像窗口
plt.plot(np.array(y_test), label='real')
plt.plot(np.array(y_predict), label='predict')
plt.tick_params(labelsize=12)  # 设置刻度的字体大小
plt.xlabel('NO.', size=13)
plt.ylabel('SVR', size=15)
plt.legend(loc="upper right")
plt.show()
mse = np.mean(np.square(y_test-y_predict))
print('MSE on the test set:', mse)
print('HR on the test set', HR(y_test, y_predict))


