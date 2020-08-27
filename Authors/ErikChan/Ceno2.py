import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sdepy
import itertools
from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY, Var

# Plot Parameters
###########################################
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['lines.linewidth'] = 1.
###########################################
np.random.seed(1)


@sdepy.integrate
def my_ou(t, x, theta=1., k=1., sigma=1.):
    return {'dt': k * (theta - x), 'dw': sigma}

#The vertex class is just there to store the nodes used in the network model.
class Vertex:
    # Initialize the vertex set
    def __init__(self, vertices=None):
        if vertices is None:
            vertices = []
        self.vertex = []
        self.vertex = [v for v in vertices]

    def add_vertex(self, v1):
        if v1 in self.vertex:
            print("Vertex %d already exists" % v1)
            return
        self.vertex.append(v1)

    def remove_vertex(self, v1):
        if v1 not in self.vertex:
            print("No vertex labelled %d exists" % v1)
            return
        self.vertex.remove(v1)

    def __len__(self):
        return len(self.vertex)

    def get_vertex(self):
        return self.vertex

    def __add__(self, other):
        self.add_vertex(other)
        return self

url = 'https://raw.githubusercontent.com/Erik-Chan/Crude-Oil-Data/master/Data/Cleaned_WTI_WSC.csv'
df = pd.read_csv(url)
df['DateTime'] = pd.to_datetime(df['DateTime'])
column_names = ['DateTime', 'WTI', 'WCS', 'WCS_Interpolated', 'WTI_Interpolated', 'WTI_WCS_diff']
df = df.reindex(columns=column_names)

priceData = df[['DateTime', 'WCS_Interpolated', 'WTI_Interpolated', 'WTI_WCS_diff']]

#This is just a switch to disable the optimization part of the code when testing things to reduce the runtime
MSE = 1

if MSE:
    # Time vector
    T = 1
    t = np.linspace(0, T, len(priceData.index))

    # The set \mathcal{S} as in the paper
    verts = Vertex(vertices=['Hardest', 'Cushing'])

    localPrices = priceData[['WCS_Interpolated', 'WTI_Interpolated']].values.tolist()


    #Initialize optimiation model object
    model = Model()

    ###################################################################################################################
    # Constants
    ###################################################################################################################
    M = np.max(localPrices) - np.min(localPrices)
    beta = 1
    ###################################################################################################################
    # Variables
    ###################################################################################################################

    # These are the psi^t
    psi_t = [model.add_var(var_type=BINARY) for _t in t]

    # These are the gamma_s^t
    gamma_st = [[model.add_var(var_type=BINARY) for _t in t] for s in verts.get_vertex()]

    # These are the alpha_s
    alpha_s = [model.add_var() for s in verts.get_vertex()]

    # These are the eta^t and independent of s
    eta_t = [model.add_var() for _t in t]

    # These are the rho_s
    rho_s = [model.add_var() for s in verts.get_vertex()]

    # These are \bar{w}_{s}^{t}.
    til_W_st = [[model.add_var() for _t in t] for s in verts.get_vertex()]

    #Dummy variable to search for infeasible regions
    dum_s = [model.add_var() for s in verts.get_vertex()]
    ###################################################################################################################
    # Constraints
    ###################################################################################################################

    # These are constraints (20b)
    for s in range(len(verts)):
        rho = rho_s[s]
        alpha = alpha_s[s]
        til_W_t = til_W_st[s]
        localPrice_t = [row[s] for row in localPrices]
        for _t in range(len(t)):
            til_W = til_W_t[_t]
            eta = eta_t[_t]
            localPrice = localPrice_t[_t]
            model += eta + rho + alpha + til_W == localPrice
            #model += eta + rho + alpha + til_W + dum_s[s]== localPrice

    # Set Non-negativity for alpha and rho
    for s in range(len(verts)):
        model += alpha_s[s] >= 0
        model += rho_s[s] >= 0

    # Constraint (20c)
    for s in range(len(verts)):
        til_W_t = til_W_st[s]
        alpha = alpha_s[s]
        for _t in range(len(t)):
            model += til_W_t[_t] >= -2*alpha

    # Constraint (20d) This seems to be a possible source of infeasible solutions
    for s in range(len(verts)):
        til_W_t = til_W_st[s]
        for _t in range(len(t)):
            model += til_W_t[_t] <= psi_t[_t]*M

    # Constraint (20e) This seems to be another source of infeasible solutions
    # I set T*len(t) to get the beta as a proportion of days
    model += xsum(psi_t) <= np.floor(beta * T*len(t))
    #Just a print statement to debug
    print('The value of np.floor(beta * T) is:', np.floor(beta * T*len(t)))

    # Constraint (20f)
    for s in range(len(verts)):
        til_W_t = til_W_st[s]
        gamma_t = gamma_st[s]
        alpha = alpha_s[s]
        for _t in range(len(t)):
            model += til_W_t[_t] <= -2 * alpha + (1 - gamma_t[_t]) * M

    # Constraint (20g)
    # Transposing gamma_st for convenience
    gamma_ts = [list(i) for i in zip(*gamma_st)]

    for _t in range(len(t)):
        model += xsum(gamma_ts[_t]) >= psi_t[_t]

    ###################################################################################################################
    # Objective Function
    ###################################################################################################################

    #Objective function (20a)
    model.objective = minimize(xsum(alpha_s[i] for i in range(len(verts))))

    ###################################################################################################################
    # Optimization
    ###################################################################################################################

    #If toggle_optimize != 0, we proceed with the optimization.
    toggle_optimize = 1
    display_parameters = 1
    if toggle_optimize:
        model.optimize()
        if display_parameters:
            print('The solution for alpha_s at the minimum is :', alpha_s[0].x, alpha_s[1].x)
            print('The solution for rho_s at the minimum is :', rho_s[0].x, rho_s[1].x)
            eta_list = []
            for _t in range(len(t)):
                eta_list.append(eta_t[_t].x)
            print('The solution for eta_t at the minimum is:', eta_list)

            for s in range(len(verts)):
                W_list = []
                til_W_tmp = til_W_st[s]
                for _t in range(len(t)):
                    W_list.append(til_W_tmp[_t].x)
                    W_list = [round(w, 2) for w in W_list]
                print('The solution for til_W at city {} is:'.format(s), W_list)

            psi_list = []
            for _t in range(len(t)):
                psi_list.append(psi_t[_t].x)
            print('The solution for psi^{t} is:', psi_list)
            print('The sum of psi is:', np.sum(psi_list))
            flat_gamma = [item for sublist in gamma_st for item in sublist]
            #print(np.sum(flat_gamma.x))
            #print(flat_gamma)
            flat_gamma = [gam.x for gam in flat_gamma]
            print('The sum of the gamma are:', sum(flat_gamma))

            localPriceHardesty = np.array([row[0] for row in localPrices])
            localPriceCushing = np.array([row[1] for row in localPrices])
            localPricesDiff = np.array(priceData['WTI_WCS_diff'].values.tolist())
            W_list = []
            for _t in range(len(t)):
                W_list.append(til_W_st[1][_t].x)
            W_list = np.array([round(w, 2) for w in W_list])
            #plt.plot(localPriceHardesty, label = 'Hardest')
            #plt.plot(localPriceCushing, label = 'Cushing')
            #plt.plot(localPricesDiff, label = 'Diff')
            #plt.plot(localPriceHardesty+W_list, label = 'Hardesty + W')

            plt.plot(W_list, label = 'Surcharge')
            plt.plot(localPricesDiff, label = 'Spread')
            plt.legend()
            plt.show()
