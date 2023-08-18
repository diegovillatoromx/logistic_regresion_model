# running the logistics regression model with statsmodel library
import statsmodels.api as sm
import numpy as np

# function to create logistics regression model with statsmodel library
def logistic_regression(df,class_col,cols_to_exclude):

  cols=df.select_dtypes(include=np.number).columns.tolist() 
  X=df[cols]
  X = X[X.columns.difference([class_col])]
  X = X[X.columns.difference(cols_to_exclude)]

  y=df[class_col]
  logit_model=sm.Logit(y,X)
  result=logit_model.fit()
  #print(result.summary2())


