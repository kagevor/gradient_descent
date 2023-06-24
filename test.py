import numpy as np
from matplotlib import pyplot as plt

from derivate import f_der

f = input()
print(f_der(f))




# # Function to minimize
# def f(x):
#     f = eval(input())
#     return f
#
#
# # Derivative of the function
# def df(x):
#     df = f.diff(x)
#     return df


# x = Symbol('x')
# f = eval(input())
# df = f.diff(x)

# print (f, df)

# Gradient descent algorithm
def gradient_descent(x_start, learning_rate, epochs):
    x = x_start
    history = [x]

    for _ in range(epochs):
        x = x - learning_rate * df
        history.append(x)

    return history


# Start value
upper_lim = 5
lower_lim = -5
epochs = 10
x_start = np.random.uniform(lower_lim, upper_lim)
# x_start = np.random.randint(lower_lim, upper_lim)
# x_start = 3


# Different learning rates
learning_rates = [0.01, 1]

fig, ax = plt.subplots(nrows=1, ncols=len(learning_rates), figsize=(20, 5))
x = np.linspace(lower_lim, upper_lim, 100)
ax.plot(x, f_der.func)

for i, learning_rate in enumerate(learning_rates):
    history = gradient_descent(x_start, learning_rate, epochs=epochs)
    ax[i].set_title(f"Epoch: {epochs}")
    ax[i].plot(x, f)
    ax[i].plot(history, [f for x in history], 'o-', label=f'learning_rate = {learning_rate}')
    ax[i].legend()


plt.show()

# fig, ax = plt.subplots(figsize=(16, 6))
# ax.set_title("Learning Curves")
# for learning_rate in learning_rates:
#     history = gradient_descent(x_start, learning_rate, epochs=100)
#     ax.plot([f(x) for x in history], label=f'learning_rate = {learning_rate}')

# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# ax.legend()

# plt.show()
