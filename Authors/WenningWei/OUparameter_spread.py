import numpy as np
import scipy.optimize as optimize
import pandas as pd
import matplotlib.pyplot as plt


dataframe = pd.read_excel('Cleaned_WTI_WSC.xlsx')
time = dataframe['DateTime']
spread = dataframe['WTI_WCS_diff']
N = len(spread)

fig, ax = plt.subplots()
plt.plot(time, spread)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xticks(rotation=20)
plt.title('WTI/WCS daily spread Jan. 2018-Jul. 2020')
plt.show()

# maximum likelihood
spread = np.array(spread)

# negative loglikelihood function
def loglikelihood(x, h=1):
    alpha = x[0]
    mu = x[1]
    sigma = x[2]
    llh = 0
    for i in range(len(spread)-1):
        llh += alpha * (spread[i+1]-np.exp(-alpha*h)*spread[i]-mu*(1-np.exp(- alpha * h))) ** 2\
               / (sigma**2*(1-np.exp(- 2* alpha * h))) +np.log(sigma)+np.log(np.pi*(1-np.exp(- 2* alpha * h))/alpha) / 2
    return llh


bds = ((.01,15),(10,40),(.5,100))

res = optimize.minimize(loglikelihood, (1,16,3), method='SLSQP', bounds=bds)
x = res.x
print(x)
alpha = x[0]
mu = x[1]
sigma = x[2]

# generator the path of OU process

M = [0,0,0]
COR = [[1,0.8, .7],
       [.8,1, .56],
       [.7,.56,1]]

simu_spread = np.zeros((N,3))
simu_spread[0] = np.mean(spread)*np.ones(3) + np.random.multivariate_normal(M,np.eye(3))
for i in range(1,N):
    simu_spread[i] = np.exp(-alpha)*simu_spread[i-1] +mu*(1-np.exp(-alpha))*np.ones(3)+\
                     sigma * np.sqrt((1-np.exp(-2*alpha))/(2*alpha)) *np.random.multivariate_normal(M,COR)

print('simulate path: ', simu_spread)

'''name = ['path1','path2', 'path3']
path = pd.DataFrame(columns=name, data=simu_spread)
path.to_csv('generateOUpath.csv')
plt.plot(path['path1'])
plt.plot(path['path2'])
plt.plot(path['path3'])
plt.show()'''
