# Building a logistic regresion model with python

## Business Objective
Predicting a qualitative response for observation can be referred to as classifying that
observation since it involves assigning the observation to a category or class.
Classification forms the basis for Logistic Regression. Logistic Regression is a 
supervised algorithm used to predict a dependent variable that is categorical or
discrete. Logistic regression models the data using the sigmoid function.
Churned Customers are those who have decided to end their relationship with their
existing company. In our case study, we will be working on a churn dataset.
XYZ is a service-providing company that provides customers with a one-year
subscription plan for their product. The company wants to know if the customers will
renew the subscription for the coming year or not.

## Architecture Diagram
<img src="architecture_diagram.png" >

## Aim
Build a logistics regression learning model on the given dataset to determine whether
the customer will churn or not.

## Approach
- Importing the required libraries and reading the dataset.
- Inspecting and cleaning up the data
- Perform data encoding on categorical variables
- Exploratory Data Analysis (EDA)
  - Data Visualization
- Feature Engineering
  - Dropping of unwanted columns
- Model Building
  - Using the statsmodel library
- Model Building
  - Performing train test split
  - Logistic Regression Model
- Model Validation (predictions)
  - Accuracy score
  - Confusion matrix
  - ROC and AUC
  - Recall score
  - Precision score
  - F1-score
- Handling the unbalanced data
  - With balanced weights
  - Random weights
  - Adjusting imbalanced data
  - Using SMOTE
- Feature Selection
  - Barrier threshold selection
  - RFE method
- Save the model in the form of a pickle file

## Tech Stack
- Language
   - Python
- Libraries
  - numpy, pandas, matplotlib, seaborn, sklearn, pickle, imblearn,
statsmodel 

## Data Description
The CSV consists of around 2000 rows and 16 columns in the [dataset](https://github.com/diegovillatoromx/logistic_regresion_model/blob/main/Data/data_regression.csv)
### Features:
- Year
- Customer_id - unique id
- Phone_no - customer phone no 
- Gender -Male/Female
- Age
- No of days subscribed - the number of days since the subscription
- Multi-screen - does the customer have a single/ multiple screen subscription
- Mail subscription - customer receive mails or not
- Weekly mins watched - number of minutes watched weekly
- Minimum daily mins - minimum minutes watched
- Maximum daily mins - maximum minutes watched
- Weekly nights max mins - number of minutes watched at night time
- Videos watched - total number of videos watched
- Maximum_days_inactive - days since inactive
- Customer support calls - number of customer support calls
- Churn
  - 0 No
  - 1 Yes 
    
## Modular Code Overview

```
  Data
    |_data_regression.csv

  src
    |_Engine.py
    |_ML_pipeline
              |_encoding.py
              |_evaluate_metrics.py
              |_feature_engg.py
              |_imbalanced_data.py
              |_ml_model.py
              |_stats_model.py
              |_rescale_variables.py
              |_scaler.py
              |_train_model.py
              |_utils.py

  lib
    |_logistic_regresion.ipynb

  output
    |_adjusted_model.pkl
    |_balanced_model1.pkl
    |_balanced_model2.pkl
    |_log_ROC.pkl
    |_model_rfe_feat.pkl
    |_model_stats.pkl
    |_model_var_feat.pkl
    |_model1.pkl
    |_smote_model.pkl
```
1. Data Folder - It contains all the data that we have for analysis. There is one csv
file in our case:
   - Data_regression.csv
2. Src folder -This is the most important folder of the project. This folder contains
all the modularized code for all the above steps in a modularized manner. This
folder consists of:
   - Engine.py
   - ML_Pipeline
     - The ML_pipeline is a folder that contains all the functions put into different
      python files, which are appropriately named. These python functions are
      then called inside the engine.py file.

3. Output folder – The output folder contains the best-fitted models that we trained
for this data. These models can  be easily loaded and used for future use and
the user need not have to train all the models from the beginning.
Note: This model is built over a chunk of data. One can obtain the model for the
entire data by running engine.py by taking the entire data to train the models.

4. Lib folder - This is a reference folder. It contains the [ipython notebook tutorial](https://github.com/diegovillatoromx/logistic_regresion_model/blob/main/lib/logistic_regression.ipynb).
