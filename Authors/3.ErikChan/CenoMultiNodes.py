import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sdepy
from GenOU import *
from mip import Model, xsum, minimize, BINARY, Var

# Plot Parameters
###########################################
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['lines.linewidth'] = 1.
###########################################
np.random.seed(1)

###################################################################################################################
# Preliminary Data Cleaning
###################################################################################################################

url = 'https://raw.githubusercontent.com/Erik-Chan/Crude-Oil-Data/master/Data/Cleaned_WTI_WSC.csv'

X_t = genOU()

df = pd.read_csv(url)

df['DateTime'] = pd.to_datetime(df['DateTime'])

df = df.sort_values(by=['DateTime'])

column_names = ['DateTime', 'WTI', 'WCS', 'WCS_Interpolated', 'WTI_Interpolated', 'WTI_WCS_diff']

#df = df.reindex(columns=column_names)
df = df.reset_index(drop=True)
print(df.head())

priceData = df[['DateTime', 'WCS_Interpolated', 'WTI_Interpolated', 'WTI_WCS_diff']]

priceData = df.assign(Edmonton=lambda x: x['WTI_WCS_diff'])

priceData = priceData.assign(Chicago=lambda x: x['WTI_WCS_diff'])

priceData = priceData.assign(Wood_River=lambda x: x['WTI_WCS_diff'])

priceData = priceData.assign(Cushing=lambda x: x['WTI_WCS_diff'])

priceData['Edmonton'] = X_t[:,0]
priceData['Chicago'] = X_t[:,1]
priceData['Wood_River'] = X_t[:,2]
priceData['Cushing'] = X_t[:,3]

###################################################################################################################
# Setting up OU simulations
###################################################################################################################

@sdepy.integrate
def my_ou(t, x, theta=1., k=1., sigma=1.):
    return {'dt': k * (x-theta), 'dw': sigma}
T = 1
t = np.linspace(0, T, 655)

x = my_ou(x0= 16.31, k = 0.011928352054776574, theta = 16.31,
          sigma = 1.00597006920309, paths = 1, steps = len(t))(t)
#x = [item for sublist in x for item in sublist]

mean = (5, 4.5, 5.25, 0)
cov = [[1.0000, 0.8819, 0.8118, 0.5096],
       [0.8819, 1.0000, 0.9744, 0.3065],
       [0.8118, 0.9744, 1.0000, 0.2832],
       [0.5096, 0.3065, 0.2832, 1.0000]]
x = np.random.multivariate_normal(mean, cov, (655))

print(x)

###################################################################################################################
# This follows the model on page 21 here:
# http://individual.utoronto.ca/izhu/files/Zhu_FinalPaper2020.pdf
###################################################################################################################

def calc_congestion(beta):
    # time in days
    T = range(len(priceData))

    # The set \mathcal{S} as in the paper
    verts = ['Edmonton', 'Chicago', 'Wood_River', 'Cushing']
    #verts = ['Edmonton', 'Chicago', 'Wood_River', 'Cushing', 'Phantom']
    S = range(len(verts))
    localPrices = priceData[verts].to_numpy()
    localPrices[:,:] = np.round(localPrices[:,:],2)

    # Initialize optimization model object
    model = Model()
    # plt.scatter(sorted(localPrices[:,0]), sorted(localPrices[:,1]))
    # plt.show()
    ###################################################################################################################
    # Constants
    ###################################################################################################################

    # This parameter may need to be relooked at if we investigate spreads
    M = np.max(localPrices[:, :]) - np.min(localPrices[:, :])

    # We set eta_t to be identically equal to 0 because it is causing overfitting problems.
    # eta_t = [0 for t in T]
    # eta_t = [localPrices[:, 0]]
    # print('My eta_t are:', eta_t)
    # plt.plot(eta_t)

    # plt.show()

    lambda_ts = localPrices

    # print('My lambda_t are:', lambda_t)
    # plt.plot(lambda_t)
    # plt.show()
    ###################################################################################################################
    # Variables
    ###################################################################################################################
    # These are the eta_t
    #eta_t = np.array([model.add_var() for t in T])

    # These are the alpha_s
    alpha_s = np.array([model.add_var() for s in S])

    # These are the rho_s
    rho_s = np.array([model.add_var() for s in S])

    # These are the eps_st
    eps_st = np.array([[model.add_var() for t in T] for s in S])

    # These are the w_st
    w_st = np.array([[model.add_var() for t in T] for s in S])

    # These are the psi^t
    psi_t = np.array([model.add_var(var_type=BINARY) for t in T])

    # These are the gamma_s^t
    gamma_st = np.array([[model.add_var(var_type=BINARY) for t in T] for s in S])

    # These are the pi_st
    pi_st = np.array([[model.add_var(var_type=BINARY) for t in T] for s in S])
    ###################################################################################################################
    # Constraints
    ###################################################################################################################

    for s in S:
        for t in T:
            # These are constraints (18b)
            #model += eta_t[t] + rho_s[s] + eps_st[s][t] + w_st[s][t] == lambda_ts[t][s]
            model += rho_s[s] + eps_st[s][t] + w_st[s][t] == lambda_ts[t][s]
            # Constraint (18c)
            model += eps_st[s][t] >= -alpha_s[s]

            # Constraint (18d)
            model += eps_st[s][t] <= alpha_s[s]

            # Constraint (18e)
            model += w_st[s][t] <= psi_t[t] * M

    # Constraint (18f)
    model += xsum(psi_t) <= np.floor(beta * len(T))

    for s in S:
        for t in T:
            # Constraint (18g)
            model += w_st[s][t] <= pi_st[s][t] * M

            # Constraint (18h)
            model += eps_st[s][t] + (1 - pi_st[s][t]) * M >= alpha_s[s]

    # Constraint (18i)
    for t in T:
        model += xsum(gamma_st[:, t]) >= psi_t[t]

    # Constraint (18j)
    model += eps_st[s][t] <= -alpha_s[s] + (1 - gamma_st[s][t]) * M

    # Constraint (18k)
    # These are binary constraints see below.

    # Constraint (17l)
    for s in S:
        for t in T:
            model += w_st[s][t] >= 0

    # Set Non-negativity for alpha and rho
    # for s in range(len(verts)):
    # model += alpha_s[s] >= 0
    # model += rho_s[s] >= 2.35
    #for s in S:
        #for t in T:
            #model += eps_st[s][t] >= 0
    ###################################################################################################################
    # Optional Constraints and Variables 21a, 21b, 21c
    ###################################################################################################################
    toggle = 0
    if toggle:
        m = 3

        def T_ub(t, m, T):
            return min(len(T), t + m)

        def T_lb(t, m):
            return max(0, t - m)

        # Variable 21c
        nu_t = np.array([model.add_var(var_type=BINARY) for t in T])
        # These end points may need to be say, t_end+1 etc but we get an index out or range
        # if we do that. I think this should be fine.

        # Constraint 21a
        for t in T:
            t_star = t
            t_end = T_ub(t, m, T)
            psi_t_star = [psi_t[_t] for _t in range(t, t_end)]
            model += xsum(psi_t_star) >= nu_t[t] * (len(T) - t_end)

        # Constraint 21b
        for t in T:
            t_star = T_lb(t, m)
            nu_t_star = [nu_t[_t] for _t in range(t_star, t)]
            model += psi_t[t] <= xsum(nu_t_star)

    ###################################################################################################################
    # Objective Function
    ###################################################################################################################

    # Objective function (20a)
    model.objective = minimize(xsum(alpha_s))

    ###################################################################################################################
    # Optimization
    ###################################################################################################################

    # If toggle_optimize != 0, we proceed with the optimization.
    toggle_optimize = 1
    display_parameters = 1
    if toggle_optimize:
        model.optimize()
        # if model.num_solutions:
        if display_parameters:
            print('The solution for alpha_s at the minimum is :', [opt.x for opt in alpha_s])
            print('The solution for rho_s at the minimum is :', [opt.x for opt in rho_s])
            Plot_W_List = []
            for s in S:
                W_list = []
                eps_list = []
                for t in T:
                    W_list.append(w_st[s][t].x)
                    W_list = [round(w, 2) for w in W_list]
                    eps_list.append(eps_st[s][t].x)
                print('The solution for til_W at city {} is:'.format(verts[s]), W_list)
                Plot_W_List.append(W_list)
                print('The solution for eps_st at city {} is:'.format(verts[s]), eps_list)
                print('The sum of eps at city {} is:'.format(verts[s]), sum(eps_list))

            psi_list = []
            for t in T:
                psi_list.append(psi_t[t].x)
            print('The solution for psi^{t} is:', psi_list)
            print('The sum of psi is:', np.sum(psi_list))

            flat_gamma = [item for sublist in gamma_st for item in sublist]
            flat_gamma = [gam.x for gam in flat_gamma]
            print('The sum of the gamma are:', sum(flat_gamma))
            # print('My eta_t are:', [sol.x for sol in eta_t])

    return Plot_W_List, [opt.x for opt in alpha_s]


# beta = np.array([0. , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.44, 0.49,
# 0.55, 0.6 , 0.65, 0.7 , 0.74, 0.8 , 0.85, 0.9 , 0.95])
beta = np.arange(0.05, 1, 0.05)
omegaList = []
W_list = []
beta = [1]
for b in beta:
    W, obj_val = calc_congestion(b)
    omegaList.append([sum(s) for s in W])
    W_list.append(W)

    for s in W_list:
        for ss in s:
            plt.plot(df['DateTime'], ss)
            plt.show()

#outputDataW = pd.DataFrame({'beta': np.round(beta, 2), 'W': W_list, 'Objective Value': obj_val, 'Sum w': np.round(omegaList, 2)})
# outputData = pd.DataFrame({'beta': np.round(beta,2), 'Sum w': np.round(omegaList,2)})
# outputData.to_csv('zbeta_data.csv')

#outputDataW.to_csv('Total data.csv')