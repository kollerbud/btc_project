import streamlit as st
import pandas as pd
import datetime
import joblib
import altair as alt


def load_data():
    # read from github
    df_data = pd.read_csv('https://raw.githubusercontent.com/kollerbud/btc_project/main/data/Data_ex_FRED.csv', parse_dates=['Time'])
    df_fred = pd.read_csv('https://raw.githubusercontent.com/kollerbud/btc_project/main/data/FRED_HARDCODED2.csv', parse_dates=['DATE'])
    # drop Unnamed columns
    df_fred.drop(df_fred.filter(regex='Unname'), axis=1, inplace=True)
    df_data.drop(df_data.filter(regex='Unname'), axis=1, inplace=True)
    # join both dataframe
    df_combine = pd.merge(
                    left=df_fred,
                    right=df_data,
                    how='inner',
                    left_on=['DATE'],
                    right_on=['Time'])
    # delete extra time
    del df_combine['Time']
    # drop null rows in "crypto_market"
    df_combine.dropna(subset=['crypto_market'], inplace=True)
    # remove columns with "..", "Country", "Country Code" etc..
    obj_col = df_combine.select_dtypes(include=['object']).columns
    df_combine = df_combine.loc[:, ~(df_combine.columns.isin(obj_col))]

    df_combine = df_combine.reset_index(drop=True).set_index('DATE')

    return df_combine

def select_data_chunk(start: str, end: str):
    # change to pandas datetime type
    _start = pd.to_datetime(start)
    _end = pd.to_datetime(end)
    # load entire dataset
    df = load_data()
    # slicing to between two dates
    df_chunk = df.loc[(df.index>=_start) & (df.index<=_end)]

    return df_chunk

def load_models():

    use_models = {}
    try:
        use_models['random_forest']=joblib.load('models/random_forest.pkl')
    except FileNotFoundError:
        use_models['random_forest']=joblib.load('src/app/models/random_forest.pkl')

    try:
        use_models['xgboost']=joblib.load('models/xgb_model.pkl')
    except FileNotFoundError:
        use_models['xgboost']=joblib.load('src/app/models/xgb_model.pkl')



    return use_models

def run_random_forest(model, feature_cols, last_price):

    cols_in_model = model.feature_names_in_
    pred_features = feature_cols.loc[:, cols_in_model]
    prediction = model.predict(pred_features)

    inv_pred = (1+prediction).cumprod()*last_price

    return inv_pred

def run_xgboost(model, feature_cols, last_price):

    # reg.get_booster().feature_names
    cols_in_model = model.get_booster().feature_names
    pred_features = feature_cols.loc[:, cols_in_model]
    prediction = model.predict(pred_features)

    inv_pred = (1+prediction).cumprod()*last_price

    return inv_pred


with st.sidebar:

    dates = st.date_input(label='Select Date Range',
                          min_value=datetime.date(2017,2,28),
                          max_value=datetime.date(2023,9,30),
                          value=[datetime.date(2017,2,28),
                                 datetime.date(2017,3,31)]
                        ) # return two dates: (datetime.date(2017, 2, 28), datetime.date(2017, 3, 31))

    select_models = st.multiselect(
        label='Select Model(s) to use',
        options=['Linear Regression', 'SVM', 'Random Forest', 'Xgboost', 'etc..'],
        default=['Random Forest']
    ) # return list of keywords ['Random Forest']

    button = st.button(label='Run models')

button = True
if button:
    chunk = select_data_chunk(start=dates[0], end=dates[1])

    features = chunk.loc[:, ~(chunk.columns.isin(['crypto_market']))]
    target = chunk.loc[:, ['crypto_market']]

    avail_models = load_models() # dict {key: model}
    
    if 'Random Forest' in select_models:
        predictions = run_random_forest(
                        model=avail_models['random_forest'],
                        feature_cols=features,
                        last_price=float(target.iloc[0].values))


    if 'Xgboost' in select_models:
        predictions = run_xgboost(
                        model=avail_models['xgboost'],
                        feature_cols=features,
                        last_price=float(target.iloc[0].values))
        
        
