import streamlit as st
from textblob import TextBlob
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from textblob import TextBlob

def predictStockMarket(input_date, input_polarity):
    df1 = pd.read_csv(r'C:\Users\risha\OneDrive\Desktop\PROJECT\NFLX_cp.csv')
    df2 = pd.read_csv(r'C:\Users\risha\OneDrive\Desktop\Nflx_news.csv')

    df1.head()

    df1.head()

    df2.head()

    d1 = df1.merge(df2, on='Date')

    d1.drop(['News_x'], axis=1)

    l=[]
    x = d1['News_y'].values
    for i in x:
      b=TextBlob(i)
      l.append(b.sentiment.polarity)

    d1 = d1.assign(News_s=l)

    missing_values = d1.isnull().sum()

    d1.head()

    d1.drop(['News_x'], axis=1, inplace=True)

    missing_values = d1.isnull().sum()

    Q1 = d1['Close'].quantile(0.25)
    Q3 = d1['Close'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_no_outliers = d1[(d1['Close'] >= lower_bound) & (d1['Close'] <= upper_bound)]

    Q1 = d1['News_s'].quantile(0.25)
    Q3 = d1['News_s'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_no_outliers = d1[(d1['News_s'] >= lower_bound) & (d1['News_s'] <= upper_bound)]

    d1.head()

    d1.describe()

    d1.drop(['High'], axis=1, inplace=True)
    d1.drop(['Low'], axis=1, inplace=True)
    d1.drop(['Adj Close'], axis=1, inplace=True)
    d1.drop(['Volume'], axis=1, inplace=True)

    d1['Date'] = pd.to_datetime(d1['Date'], errors='coerce')
    d1.drop(['News_y'], axis=1, inplace=True)
    d1.set_index('Date', inplace=True)
    d1.head()

    d1['Days'] = (d1.index - d1.index.min()).days

    X = d1[['Days', 'News_s']]
    y = d1['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    degree = 2
    model = LinearRegression()

    model.fit(X_train, y_train)

    X_test['Days'].fillna(X_test['Days'].mean(), inplace=True)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    future_date =  pd.Timestamp(input_date)
    future_days = (future_date - d1.index.min()).days
    future_polarity =  input_polarity

    future_data = pd.DataFrame({'Days': [future_days], 'News_s': [future_polarity]})

    future_prediction = model.predict(future_data)

    print(future_prediction[0])
    return future_prediction[0]

import streamlit as st
from textblob import TextBlob
import pandas as pd

def analyze_sentiment(news_statement):
    analysis = TextBlob(news_statement)
    return analysis.sentiment.polarity

def predict_stock_market(date, polarity):
    prediction = f"Prediction for {date}: High polarity, {polarity * 100}% chance of increase"
    return prediction

st.title("Stock Price Prediction App")

@st.cache(allow_output_mutation=True)
def get_previous_results():
    return []

previous_results = get_previous_results()

news_statement = st.text_area("Enter a News Statement")
pol = 0

if st.button("Analyze Sentiment"):
    pol = analyze_sentiment(news_statement)
    st.success(f"Sentiment Polarity: {pol}")

date = st.date_input("Select a date")

if st.button("Predict Stock Price"):
    prediction = predictStockMarket(date, pol)
    previous_results.append({"news_statement": news_statement, "polarity": round(pol, 2), "prediction": prediction})
    pd.DataFrame(previous_results).to_csv("previous_results.csv", index=False)
    st.success(f"Predicted Stock Price: {prediction}")

with st.sidebar:
    st.subheader("Previous Search Results:")
    for result in previous_results:
        st.markdown(f"**News Statement:** {result['news_statement']}")
        st.markdown(f"**Prediction:** {result.get('prediction', 'N/A')}")
        st.markdown("---")
