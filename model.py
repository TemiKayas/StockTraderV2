import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_save_models(X_train, y_train):
    # Train the Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    joblib.dump(rf, '../models/random_forest_model.pkl')

    # Train the Gradient Boosting Regressor
    gb = GradientBoostingRegressor(n_estimators=100)
    gb.fit(X_train, y_train)
    joblib.dump(gb, '../models/gradient_boosting_model.pkl')

    print("Models trained and saved!")

if __name__ == '__main__':
    # Load the preprocessed datasets
    path = '../data/processed_data/'
    X_train = joblib.load(path + 'X_train_preprocessed.pkl')
    y_train = joblib.load(path + 'y_train_preprocessed.pkl')

    # Train and save the models
    train_and_save_models(X_train, y_train)

    print("Model training complete!")
