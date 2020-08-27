import numpy as np
import matplotlib.pyplot as plt
import sdepy
import pandas as pd

# Plot Parameters
###########################################
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['lines.linewidth'] = 1.
###########################################
np.random.seed(1)


@sdepy.integrate
def my_ou(t, x, theta=1., k=1., sigma=1.):
    return {'dt': k * (x-theta), 'dw': sigma}
T = 1
t = np.linspace(0, T, 1000)
x = my_ou(x0= 16.31, k = 0.011928352054776574, theta = 16.31,
          sigma = 1.00597006920309, paths = 2, steps = len(t))(t)
x = np.array(x)
print(x)
name = ['path1', 'path2']
path = pd.DataFrame(columns=name, data=x)
path.to_csv('generateOUpath.csv')
