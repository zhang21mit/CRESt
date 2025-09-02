import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA

#rf=joblib.load("HEA.joblib")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


a=pd.read_excel('Data.xlsx')

ab=a['abandoned'].to_numpy()!=1
y=a['max_power'].to_numpy()

a1=a[['Pd','Pt','Cu','Au','Ir','Ce','Nb','Cr']].to_numpy()
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(a1[ab])


a1=a1[ab]
y=y[ab]


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error




X_train, X_test, y_train, y_test = train_test_split(a1, y, test_size=0.2, random_state=0)
model = GradientBoostingRegressor(n_estimators=4096, random_state=0, learning_rate=1e-2,verbose=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(np.abs(y_test-y_pred).mean())
print(np.corrcoef(y_test,y_pred)[0,1])
y_pred = model.predict(a1)

print(max(y_pred))