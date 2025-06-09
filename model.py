import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

def predict_with_lr(data):
    df = data[['Close']].copy()
    df['Target'] = df['Close'].shift(-7)
    df.dropna(inplace=True)
    df['Days'] = range(len(df))
    X = df[['Days']]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    }, index=y_test.index)
    return result

def predict_with_arima(data):
    df = data[['Close']].copy()
    df.dropna(inplace=True)
    split_index = int(len(df) * 0.8)
    train = df['Close'].iloc[:split_index]
    test = df['Close'].iloc[split_index:]
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    result = pd.DataFrame({
        "Actual": test.values,
        "Predicted": predictions
    }, index=test.index)
    return result