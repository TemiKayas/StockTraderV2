import pandas as pd
import joblib

def preprocess_data(X, y):
    """
    Preprocesses the data by handling non-numeric features and potential null values.
    
    :param X: Features DataFrame.
    :param y: Target DataFrame or Series.
    :return: Preprocessed X and y.
    """

    # If Ticker information might be useful for your model, you can one-hot encode it.
    # Else, you can just drop it.
    X = pd.get_dummies(X, columns=['Ticker'], drop_first=True)

    # Handling potential null values. For simplicity, we fill them with the mean of the column.
    # Depending on the nature of the data, you might opt for median, mode, or other imputation methods.
    X = X.fillna(X.mean())

    return X, y

if __name__ == '__main__':
    # Load the datasets
    path = '../data/processed_data/'
    X_train = joblib.load(path + 'X_train.pkl')
    X_test = joblib.load(path + 'X_test.pkl')
    y_train = joblib.load(path + 'y_train.pkl')
    y_test = joblib.load(path + 'y_test.pkl')

    # Preprocess the data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Save the preprocessed datasets
    joblib.dump(X_train, path + 'X_train_preprocessed.pkl')
    joblib.dump(X_test, path + 'X_test_preprocessed.pkl')
    joblib.dump(y_train, path + 'y_train_preprocessed.pkl')
    joblib.dump(y_test, path + 'y_test_preprocessed.pkl')

    print("Data preprocessing complete!")

