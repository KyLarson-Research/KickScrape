#File Name: rfrkickpipe.py
#Author: Kyle Carlton Larson
#Purpose: to expand upon prior techniques for generating success/fail
#predictions from Kickstarter dataset using pipeline
numerical_cols=[]
n_cols = set(X_train.columns) -set(non_numeric)
for n in n_cols:
    numerical_cols.append(n)
print(numerical_cols)
#what if we include the categorical variables?
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy = 'constant')

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer( transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, non_numeric)])

kick_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
kick_pipe.fit(X_train, y_train)
preds = kick_pipe.predict(X_valid)

score = mean_absolute_error(y_valid, preds)
print('MAE:'+str(score))