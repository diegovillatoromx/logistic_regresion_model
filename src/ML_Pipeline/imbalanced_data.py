from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,classification_report
from sklearn.utils import resample
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE

# function to create model with balanced weights
def run_model_bweights(X_train,X_test,y_train,y_test):
    global logreg
    logreg = LogisticRegression(random_state = 13,class_weight = 'balanced')
    logreg.fit(X_train, y_train)
    global y_pred
    y_pred = logreg.predict(X_test)
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    #print(classification_report(y_test, y_pred))
    #print("The area under the curve is: %0.2f"%logit_roc_auc)



# function to create model with defining weights
def run_model_aweights(X_train,X_test,y_train,y_test,w):
    global logreg
    logreg = LogisticRegression(random_state = 13,class_weight=w)
    logreg.fit(X_train, y_train)
    global y_pred
    y_pred = logreg.predict(X_test)
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    #print(classification_report(y_test, y_pred))
    #print("The area under the curve is: %0.2f"%logit_roc_auc)



#function to create model with adjusting imbalanced data
def adjust_imbalance (X_train,y_train,class_col):
  X = pd.concat([X_train, y_train], axis=1)
  # separate the 2 classes
  class0 = X[X[class_col]==0]
  class1 = X[X[class_col]==1]

  # Case 1 - bootstraps from the minority class
  if len(class1)<len(class0):
    resampled = resample(class1,
                              replace=True, 
                              n_samples=len(class0), 
                              random_state=10) 
    resampled_df = pd.concat([resampled, class0])

  # Case 1 - ressamples from the majority class
  else:
    resampled = resample(class1,
                              replace=False, 
                              n_samples=len(class0), 
                              random_state=10) 
    resampled_df = pd.concat([resampled, class0])
  return resampled_df



#function to create a model using SMOTE
def prepare_model_smote(df,class_col,cols_to_exclude): 
  cols=df.select_dtypes(include=np.number).columns.tolist() 
  X=df[cols]
  X = X[X.columns.difference([class_col])]
  X = X[X.columns.difference(cols_to_exclude)]
  y=df[class_col]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
  sm = SMOTE(random_state=0, sampling_strategy=1.0)
  X_train, y_train = sm.fit_resample(X_train, y_train) 
  return(X_train, X_test, y_train, y_test)