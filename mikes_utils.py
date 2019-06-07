from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd

def print_nan_cols(df, moderate_only = False):
    values = list(zip(list(df.isnull().columns), list(df.isnull().any())))
    filtered = list(filter(lambda x: x[1][1] == True, enumerate(values)))
    contains_nan = [y for x, y in filtered]
    proportion_nan = [sum(df[x].isnull()) / len(df[x]) for x, y in contains_nan]
    proportion_nan = [(x[0], proportion_nan[i]) for i, x in enumerate(contains_nan)]
    for col, propo_nan in proportion_nan:
        if abs(propo_nan) > 0.3 or moderate_only is False:
            print(col, ': ', propo_nan)
            
def print_moderate_correlations(df, col_to_correlate):
    cols = df[df.columns].corr().columns
    if df[col_to_correlate].dtype.name == 'category':
        df[col_to_correlate] = df[col_to_correlate].cat.codes
    corrs = df[df.columns].corr()[col_to_correlate]
    for col, corr in zip(cols, corrs):
        if abs(corr) > 0.4 and col != col_to_correlate:
            print(col, ': ', corr)
            
def get_moderate_corrs(df, col_to_correlate):
    cols = df[df.columns].corr().columns
    corrs = df[df.columns].corr()[col_to_correlate]
    for col, corr in zip(cols, corrs):
        if abs(corr) < 0.05 and col != col_to_correlate:
            df = df.drop(columns=[col])
    return df
            
def convert_categorical_to_numbers(to_change_df, numbers=True):
    for col, dtype in zip(to_change_df.columns, to_change_df.dtypes):
        if (dtype == object):
            to_change_df[col] = to_change_df[col].astype('category')
    if numbers:
        return pd.get_dummies(to_change_df)
    else:
        return to_change_df
  

def replace_missing_with_ml(df, predict_missing_df, col_to_predict, is_classify = False):

    predict_missing_df[col_to_predict] = df[col_to_predict]
    adjusted_missing = predict_missing_df[predict_missing_df[col_to_predict].isnull() == False]

    y = adjusted_missing[col_to_predict].values
    x = adjusted_missing.drop(columns=[col_to_predict]).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    
    if is_classify:
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42)
    else:
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42)
        
    rf.fit(x_train, y_train)
    r2 = r2_score(y_test, rf.predict(x_test))
    mse = mean_squared_error(y_test, rf.predict(x_test))
    rmse = mse**(1/2)
    print('R2          : ', round(r2, 4))
    print('RMSE        : ', round(rmse, 2))

    missing_df = predict_missing_df[predict_missing_df[col_to_predict].isnull()]
    x_missing = missing_df.drop(columns=[col_to_predict]).values
    predictions = rf.predict(x_missing).astype(int)

    df.loc[df[col_to_predict].isnull(), col_to_predict] = predictions
    
    predict_missing_df = predict_missing_df.drop(columns=[col_to_predict])
    return df, predict_missing_df

def adjust_skewness(df):
    numerics = list()

    for col, dtype in zip(df.columns, df.dtypes):
        if dtype.name != 'object' and dtype.name != 'category':
            numerics.append(col)

    skewed_feats = df[numerics].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 0.7]

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], lam)
        
    return df