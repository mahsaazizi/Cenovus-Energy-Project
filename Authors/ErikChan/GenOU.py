import numpy as np
import matplotlib.pyplot as plt
import sdepy

# Plot Parameters
###########################################
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['lines.linewidth'] = 1.
###########################################
np.random.seed(1)


#@sdepy.integrate
#def my_ou(t, x, theta=1., k=1., sigma=1.):
#    return {'dt': k * (x-theta), 'dw': sigma}
#x = my_ou(x0= 16.31, k = 0.011928352054776574, theta = 16.31,
#          sigma = 1.00597006920309, paths = 1, steps = len(t))(t)

T = 1
t = np.linspace(0, T, 655)

#Mean and covariance matrix of the multivariate normal
#dZ_t dW_t
mean = (0, 0, 0, 0)
cov = [[1.0000, 0.8819, 0.8118, 0.5096],
       [0.8819, 1.0000, 0.9744, 0.3065],
       [0.8118, 0.9744, 1.0000, 0.2832],
       [0.5096, 0.3065, 0.2832, 1.0000]]

# Note that if we want to code this ourselves using the Kaiser-Dickman algorithm,
# we should use the Cholesky decomposition on the covariance matrix. But here
# we simply let numpy do the work.
dW = np.random.multivariate_normal(mean, cov, (655))

# Vector of initial values

#X_0 = [5, 4.5, 5.25, 0]

X_0 = [9.87+ 5 + 0.67,
       9.87+ 4.5,
       9.87 +5.25+ 2.1,
       9.87+0]

#Mean reversion rate
k = [0.119, 0.119, 0.119, 0.119]
#k = [2,2,2,2]
#Mean reversion level
mu = [9.87+ 5 + 0.67, 9.87+ 4.5, 9.87 +5.25+ 2.1, 9,87+0]

#Setting X_0 = X_0
X_t = [X_0]

for t in range(len(t)):
    delXt = []

    for i in range(len(X_0)):

        delXt.append(X_t[t][i] + k[i]*(mu[i]-X_t[t][i])+dW[t][i])
    X_t.append(delXt)

X_t = np.array(X_t)


#X_t = dX_t-1 + X_t-1
for i in range(len(X_0)):
    plt.plot(X_t[:,i])
plt.show()

def genOU():
    pass
