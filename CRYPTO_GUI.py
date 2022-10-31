import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from datetime import date
import datetime

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px

from PIL import Image
from urllib.request import urlopen

import seaborn as sns
from pandas_datareader.data import DataReader




st.set_page_config(layout="wide")
imageBTC = Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1.png'))
#display image
st.image(imageBTC, width = 100)

st.title('Cryptocurrency Price Prediction Model')
st.text('Mainly Created for SOLiGence')

# url = requests.get("https://cointelegraph.com/rss/tag/blockchain")

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


# #side bar and main panel
# col1.header('Select the option')
#
# col2, col3 = st.columns((2,1))


crypto_symbols = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'DOGE-USD', 'SOL-USD','USDT-USD','XRP-USD','ADA-USD','XLM-USD',
                  'BNB-USD', 'DOT-USD','DASH-USD','BSV-USD','VET-USD','AVAX-USD','EOS-USD','ZEC-USD','CEL','HEX','TRX']

selected_coins = st.sidebar.selectbox("Select dataset for prediction", crypto_symbols)

period = st.sidebar.slider('Choose prediction date:', 1, 7, 31)



@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Start Loading data...')
data = load_data(selected_coins)
data_load_state.text('Loading data...done!')


# data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])

def cls_plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Closing_Price'))
    fig.layout.update(title_text='Closing Price Versus Time', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

cls_plot_raw_data()

def vlm_plot_raw_data():
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], name='Volume_Traded'))
    fig2.layout.update(title_text='Volume Traded Versus Time', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig2)

vlm_plot_raw_data()


crypt_symbols = ['BTC-USD','ETH-USD','LTC-USD','XRP-USD','XLM-USD','ADA-USD','DOGE-USD','BNB-USD','EOS-USD','TRX-USD']

# Grab all the closing prices for the tech stock list into one DataFrame
closing_df = DataReader(crypt_symbols, 'yahoo', start='2018-01-01', end =date.today().strftime("%Y-%m-%d"))['Close']

# Make a new tech returns DataFrame
crypt_map = closing_df.pct_change()

st.set_option('deprecation.showPyplotGlobalUse', False)
# df_Btt_corr = df_Btt.corr(method='pearson')
fig=plt.figure(figsize=(8,6))

sns.heatmap(crypt_map.corr(), annot=True)
st.pyplot()

data_ma = data.set_index('Date')



st.subheader('Closing Price Vs Time Chart with 20MA & 250MA(Moving Average)')
ma20 = data_ma.Close.rolling(20).mean()
ma250 = data_ma.Close.rolling(250).mean()
fig5 = plt.figure(figsize=(12,6))
plt.plot(ma20, 'r')
plt.plot(ma250, 'y')
plt.plot(data_ma.Close, 'b')
st.pyplot(fig5)

df_train = data[['Date', 'Close']]

df_train = df_train.rename(columns={'Date':"ds", "Close": 'y'})

m = Prophet(
    changepoint_range=0.8,  # percentage of dataset to train on
    yearly_seasonality='auto',  # taking yearly seasonality into account
    weekly_seasonality='auto',  # taking weekly seasonality into account
    daily_seasonality=False,  # taking daily seasonality into account
    seasonality_mode='multiplicative'
    # additive (for more linear data) or multiplicative seasonality (for more non-linear data)
)
m.fit(df_train)

future = m.make_future_dataframe(period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

final_p = forecast.yhat.iloc[-1]
final_d = forecast.ds.iloc[-1]
st.sidebar.write(f"The prediction for the chosen date {final_d} is {final_p}")

df_diff = forecast.set_index('ds')[['yhat']].join(df_train.set_index('ds').y).reset_index()
df_diff.dropna(inplace=True)
df_diff['profit_loss'] = df_diff.yhat - df_diff.y
st.write(df_diff.tail())

final_pl = df_diff.profit_loss.iloc[-1]

if final_pl > 1:
    st.sidebar.write(f"The profit is $:{final_pl}")
else:
    st.sidebar.write(f"The loss is $:{final_pl}")

# # import requests
# # from bs4 import BeautifulSoup

# # url = requests.get('https://cointext.com/news/tag/bitcoin/feed/')
# # soup = BeautifulSoup(url.content, 'xml')
# # entries = soup.find_all('item')
# #
# # for item in entries:
# #     title = item.title.text
# #     st.write(f"{title}...")
# #