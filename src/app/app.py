import streamlit as st
import warnings
warnings.simplefilter(action='ignore')
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def load_data():
    df_predictions = pd.read_csv('https://raw.githubusercontent.com/kollerbud/btc_project/main/src/app/predictions.csv', index_col=['DATE'], parse_dates=['DATE'])
    df_features = pd.read_csv('https://raw.githubusercontent.com/kollerbud/btc_project/main/src/app/features.csv')

    return (df_predictions, df_features)


with st.sidebar:

    dates = st.date_input(label='Select Date Range',
                          min_value=datetime.date(2023,1,1),
                          max_value=datetime.date(2023,9,30),
                          value=[datetime.date(2023,1,1),
                                 datetime.date(2023,2,1)]
                        ) # return two dates: (datetime.date(2017, 2, 28), datetime.date(2017, 3, 31))

    select_models = st.multiselect(
        label='Select Model(s) to use',
        options=['KNN', 'OLS', 'SVR',
                 'LSTM', 'Random Forest', 'Xgboost'],
        default=['OLS', 'LSTM']
    )

    button = st.button(label='Show predictions')


if button:
    # load data
    predict, features = load_data()
    # prepare selected models
    cols = []
    cols.append('Crypto_Price')
    for select in ['KNN', 'OLS', 'SVR', 'LSTM',
                   'Random Forest', 'Xgboost']:
        if select in select_models:
            cols.append(select)
    # prepared top features for selected models
    feature_df = pd.DataFrame()
    for col in cols:
        _df = features.loc[:,features.columns.str.match(col)]
        if not _df.empty:
            feature_df = pd.concat([feature_df, _df], axis=1)

    # make line chart
    st.title('Selected Model Predictions vs Actual Price')
    fig, ax = plt.subplots(figsize=(10,6))
    chart_data = predict[cols]
    for model in chart_data:
        ax.plot(predict.index, chart_data[model], label=model)

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    st.pyplot(fig)
    
    st.title('Top features in selected models')
    st.dataframe(feature_df, hide_index=False)

