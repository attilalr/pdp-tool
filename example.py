import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from pdp_tool import pdp

# Lets use de iris dataset for example
data = load_iris()

# lets change to a df
df = pd.DataFrame(data=data.data, columns=data.feature_names)

df['target'] = data.target
print (df)

# Categorical pdp
# for categorical analisys the categorical variable must be bool 0 or 1
# if you have more than 2 classes you must hot-encode before
df['target_is_0'] = (df['target']==0).astype('int')
df['target_is_1'] = (df['target']==1).astype('int')
df['target_is_2'] = (df['target']==2).astype('int')

print (df)

features = ['sepal length (cm)', 
  'sepal width (cm)', 
  'petal length (cm)', 
  'petal width (cm)',
  ]

# target 0
pdp(df, features, 'target_is_0', n=4, writefolder=None, digits=2, figsize=(8,6))

# target 1
pdp(df, features, 'target_is_1', n=4, writefolder=None, digits=2, figsize=(8,6))

# target 2
pdp(df, features, 'target_is_2', n=4, writefolder=None, digits=2, figsize=(8,6))

# Continous pdp
# we can use pdp for a y continous variable too
# n is the number of bins, digits control the digits in xlabel
pdp(df, features, 'petal width (cm)', n=6, writefolder=None, digits=2, figsize=(8,6))

