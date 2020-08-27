import pandas as pd
import matplotlib.pyplot as plt

Oilimport = pd.read_csv('Weekly_U.S._Imports_from_Canada_of_Crude_Oil.csv', skiprows=4)
print(Oilimport['Weekly U.S. Imports from Canada of Crude Oil Thousand Barrels per Day'])
Oil_18_20 = Oilimport['Weekly U.S. Imports from Canada of Crude Oil Thousand Barrels per Day'][3:142]
plt.plot(Oil_18_20)
plt.show()