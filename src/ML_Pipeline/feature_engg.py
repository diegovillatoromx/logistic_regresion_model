from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn import preprocessing


class_col = 'churn'
cols_to_exclude=['customer_id','phone_no', 'year']

 #function to select variables with thershold
def var_threshold_selection(df,cols_to_exclude,class_col,threshold):
  cols=df.select_dtypes(include=np.number).columns.tolist() #finding all the numerical columns from the dataframe
  X=df[cols] #creating a dataframe only with the numerical columns
  X = X[X.columns.difference(cols_to_exclude)] #columns to exclude
  X = X[X.columns.difference([class_col])]
  ## Scaling variables
  scaler = preprocessing.StandardScaler().fit(X)
  X_scaled = scaler.transform(X)
  var_thr = VarianceThreshold(threshold = threshold) #Removing both constant and quasi-constant
  var_thr.fit(X_scaled)
  var_thr.get_support()
  selected_cols = X.columns[var_thr.get_support()]
  print("The selected features are: ")
  print(list(selected_cols))


# function to select variables using RFE
def rfe_selection(df,cols_to_exclude,class_col,model):
  import warnings
  warnings.filterwarnings("ignore")
  import numpy as np
  from sklearn.feature_selection import RFE
  cols=df.select_dtypes(include=np.number).columns.tolist() #finding all the numerical columns from the dataframe
  X=df[cols] #creating a dataframe only with the numerical columns
  X = X[X.columns.difference(cols_to_exclude)] #columns to exclude
  X = X[X.columns.difference([class_col])]
  y = df[class_col]
  rfe = RFE(model)
  rfe = rfe.fit(X, y)
  global selected_cols
  selected_cols = X.columns[rfe.support_]
  print("The selected features are: ")
  print(list(selected_cols))