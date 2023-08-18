# import the required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from ML_Pipeline.utils import read_data,inspection,null_values
from ML_Pipeline.encoding import encode_categories
from ML_Pipeline.stats_model import logistic_regression
from ML_Pipeline.ml_model import prepare_model,run_model
from ML_Pipeline.evaluate_metrics import confusion_matrix,roc_curve
from ML_Pipeline.imbalanced_data import run_model_bweights,run_model_aweights,adjust_imbalance,prepare_model_smote
from ML_Pipeline.feature_engg import var_threshold_selection,rfe_selection


# Read the initial datasets
df = read_data("../input/data_regression.csv")

# Inspection and cleaning the data
x = inspection(df)

# Drop the null values
df = null_values(df)

# Encoding categorical variables
var = ['gender','multi_screen','mail_subscribed']
en = encode_categories(df,var)



                 ### Run the logistic regression model with statsmodel ###
model_stats = logistic_regression(df,class_col = 'churn',cols_to_exclude=['customer_id','phone_no', 'year'])
pickle.dump(model_stats, open('../output/model_stats.pkl', 'wb'))



                 ### Run the logistics regression model with sklearn ###

##Selecting only the numerical columns and excluding the columns we specified in the function
X_train, X_test, y_train, y_test = prepare_model(df,class_col='churn',
                                                 cols_to_exclude=['customer_id','phone_no', 'year'])

# run the model
model_log,y_pred = run_model(X_train,X_test,y_train,y_test)

## performance metric ##
conf_matrix = confusion_matrix(y_test,y_pred)

#print(conf_matrix)
roc_val = roc_curve(model_log,X_test,y_test)

## Save the model ##
pickle.dump(model_log, open('../output/model1.pkl', 'wb'))



                     ### Dealing with imbalanced data ###
                    
# model with default parameter for balancing the data
balanced_model1 = run_model_bweights(X_train,X_test,y_train,y_test)
pickle.dump(balanced_model1, open('../output/balanced1_model.pkl', 'wb'))

# model with w as an input to balance the data
balanced_model2 = run_model_aweights(X_train,X_test,y_train,y_test,{0:90, 1:10})
pickle.dump(balanced_model2, open('../output/balanced2_model.pkl', 'wb'))

#model with adjusting the imbalance data
resampled_df = adjust_imbalance (X_train,y_train,class_col = 'churn')
X_train,X_test,y_train,y_test = prepare_model(resampled_df,class_col = 'churn',cols_to_exclude=['customer_id','phone_no', 'year'])
adj_bal = run_model(X_train,X_test,y_train,y_test)
pickle.dump(adj_bal, open('../output/adjusted_model.pkl', 'wb'))

# model with SMOTE
X_train,X_test,y_train,y_test = prepare_model_smote(df,class_col = 'churn',
                                                    cols_to_exclude=['customer_id','phone_no', 'year'])
smote_model,y_pred = run_model(X_train,X_test,y_train,y_test)
pickle.dump(smote_model, open('../output/smote_model.pkl', 'wb'))



                        ### Perform Feature Engineering ###
 # perform feature selection with var feature
var_feature = var_threshold_selection(df,cols_to_exclude=['customer_id','phone_no', 'year'],
                                      class_col = 'churn',threshold=1)

 # create a LR model with var features
X_train,X_test,y_train,y_test = prepare_model(resampled_df,class_col = 'churn',cols_to_exclude=['customer_id','phone_no', 'year',
                                                                'gender', 'age',
                                                                'no_of_days_subscribed', 'multi_screen', 'mail_subscribed', 'minimum_daily_mins', 
                                                                'weekly_max_night_mins', 'videos_watched', 
                                                                'customer_support_calls', 'churn', 'gender_code', 'multi_screen_code',
                                                                'mail_subscribed_code'])
model_var_feat = run_model(X_train,X_test,y_train,y_test)
pickle.dump(model_var_feat, open('../output/model_var_feat.pkl', 'wb'))

 # perform feature selection with var feature
rfe_sel = rfe_selection(df,class_col = 'churn',
                         cols_to_exclude = ['customer_id','phone_no', 'year'],
                         model = model_log)

X_train,X_test,y_train,y_test = prepare_model(resampled_df,class_col = 'churn',cols_to_exclude=['customer_id','phone_no', 'year',
                                                                'gender', 'age',
                                                                'no_of_days_subscribed', 'multi_screen', 'mail_subscribed', 
                                                                'weekly_max_night_mins', 
                                                                 'gender_code', 'multi_screen_code',
                                                                'mail_subscribed_code'])
model_rfe_feat = run_model(X_train,X_test,y_train,y_test)
pickle.dump(model_rfe_feat, open('../output/model_rfe_feat.pkl', 'wb'))
