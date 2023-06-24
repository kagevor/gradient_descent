from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import datasets
import time

from batch_gd import BGDRegressor
from mini_batch_gd import MBGDRegressor
from stochastic_gd import SGDRegressor

dataset = datasets.load_diabetes()
x = dataset.data
y = dataset.target
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)

batch_gd = BGDRegressor(epochs=1000, learning_rate=0.01)

start = time.time()
batch_gd.fit(X_train, Y_train)
score = time.time() - start
print('The time taken is', score)

y_predict = batch_gd.predict(X_test)
print(r2_score(Y_test, y_predict))

stochastic_gd = SGDRegressor(epochs=1000, learning_rate=0.01)
print('stochastic')

start = time.time()
stochastic_gd.fit(X_train, Y_train)
score = time.time() - start
print('The time taken is', score)

y_predict = stochastic_gd.predict(X_test)
print(r2_score(Y_test, y_predict))

mb_gd = MBGDRegressor(batch_size=10, epochs=1000, learning_rate=0.01)
print('mini batch')

start = time.time()
mb_gd.fit(X_train, Y_train)
score = time.time() - start
print('The time taken is', score)
y_predict = mb_gd.predict(X_test)
print(r2_score(Y_test, y_predict))

