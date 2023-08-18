from sklearn.preprocessing import OrdinalEncoder

# create a function for performing encoding for categorical variables
def encode_categories (df,variables):
  ord_enc = OrdinalEncoder()
  for v in variables:
    name = v+'_code'
    df[name] = ord_enc.fit_transform(df[[v]])
    #print('The encoded values for '+ v + ' are:')
    #print(df[name].unique())