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


a=pd.read_excel('Data S3.xlsx')

ab=a['abandoned'].to_numpy()!=1
y=a['max_power'].to_numpy()

a1=a[['Pd','Pt','Cu','Au','Ir','Ce','Nb','Cr']].to_numpy()
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(a1[ab])


a1=a1[ab]
y=y[ab]

#plt.plot(pca.explained_variance_ratio_)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error



rf =joblib.load("rf.pkl")

X_train, X_test, y_train, y_test = train_test_split(a1, y, test_size=0.2, random_state=0)
#rf = GradientBoostingRegressor(n_estimators=1024, random_state=0, learning_rate=0.8e-2,verbose=1)
# 訓練模型
#rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# 計算均方誤差

print(np.abs(y_test-y_pred).mean())
print(np.corrcoef(y_test,y_pred)[0,1])
y_pred = rf.predict(a1)

print(max(y_pred))


#%%

iteration=10
batch_size=20



import numpy as np
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
from bayes_opt import UtilityFunction
from scipy.optimize import minimize


data=np.load('emb.npy')

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

data=normalize_l2(data[:,:64])#64
#norm_data=(data-data.mean(axis=0))/data.var(axis=0)**0.5

import numpy as np
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
from bayes_opt import UtilityFunction
from scipy.optimize import minimize

dim=5


pca = PCA(n_components=dim)#,whiten=True)

output=pca.fit_transform(data)
low_bound=output.min(axis=0)
up_bound=output.max(axis=0)
print(sum(pca.explained_variance_ratio_))
#%%
t=[]
t1=[]
def black_box_function(**kwargs):
    element=[]
    for x in kwargs:
        element.append(kwargs[x])
    element=np.array(element).reshape(1,-1)
    now=pca.inverse_transform(element).reshape(-1,)
    
    
    
    def objective_function(x):
        x=np.array(x).reshape(1,-1)
        t.append(now)
        return ((np.dot(x,data[:8])-now)**2).sum()
    
    
    x0 = np.zeros(8)+1/8
    bounds=[]
    for i in x0:
        bounds.append((0,1))
        
    result = minimize(objective_function, x0, method='SLSQP', bounds=bounds)
    element=np.array(result.x)+1e-8
    
    element/=element.sum()
    return rf.predict(element.reshape(1,-1))[0]

         



col_log=[]

for seed in range(1,21):
    pbounds={}
    for i in range(dim):
        pbounds['x%d'%(i)]= (low_bound[i],up_bound[i])
    optimizer = BayesianOptimization(f=None, pbounds=pbounds, verbose=0, random_state=seed,allow_duplicate_points=True)
    
    acquisition_function = UtilityFunction(kind="ucb",kappa=1.5)
    
    count=0
    
    log=[]
    for _ in range(iteration):
        in_log=[]
        in_ans=[]
        for _ in range(batch_size):
            
            next_point = optimizer.suggest(acquisition_function)
            while next_point in [i['params'] for i in optimizer.res]:
                next_point = optimizer.suggest(acquisition_function)
            temp=black_box_function(**next_point)
            
            in_ans.append(temp)
            in_log.append(next_point)
            log.append(temp)
        
        for i in range(batch_size):
            optimizer.register(params=in_log[i], target=in_ans[i])
        count+=1
        print('%d : %.3f'%(count,max(in_ans)))

    col_log.append(log)
    np.save('llm_bo',col_log)







