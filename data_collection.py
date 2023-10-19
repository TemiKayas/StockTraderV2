import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

def fetch_data(tickers, start_date='2020-01-01', end_date='2022-01-01'):
    """
    Fetches stock market data for given tickers and time range.
    
    :param tickers: List of stock tickers to fetch data for.
    :param start_date: Start date for historical data.
    :param end_date: End date for historical data.
    :return: Aggregated DataFrame of stock data.
    """
    data = pd.DataFrame()
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data['Ticker'] = ticker  # add a column to identify the ticker
        data = pd.concat([data, stock_data])
    return data

if __name__ == '__main__':
    # Define tickers for which you want data.
    # For demonstration purposes, I'm using three tickers. Modify as needed.
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # Fetch data
    data = fetch_data(tickers)

    # Split the data. For demonstration, assume 'Close' is your target column.
    X = data.drop('Close', axis=1)
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Specify path for saving data
    path = '../data/processed_data/'
    
    # Save the datasets
    joblib.dump(X_train, path + 'X_train.pkl')
    joblib.dump(X_test, path + 'X_test.pkl')
    joblib.dump(y_train, path + 'y_train.pkl')
    joblib.dump(y_test, path + 'y_test.pkl')

    print("Data collection and processing complete!")
