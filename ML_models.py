import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

#  Perform PCA on the transposed data to reduce dimensionality.
def perform_pca(data, n_components=10):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data.T)
    return reduced_data

# Function perform K-means clustering on reduced data.
def kmeans_clustering(data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

# Fit ARIMA model on training data
def fit_arima_model(train_data, order=(5, 1, 0)):
    arima_model = ARIMA(train_data, order=order)
    arima_model_fit = arima_model.fit()
    return arima_model_fit

# Fit LSTM model on training data
def fit_lstm_model(train_data):
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(60, len(scaled_train_data)):
        X_train.append(scaled_train_data[i - 60:i, 0])
        y_train.append(scaled_train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    lstm_model.add(LSTM(units=50, return_sequences=False))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
    return lstm_model, scaler

# Fit SVM model on training data
def fit_svm_model(train_data):
    svm_model = SVR(kernel='rbf')
    train_indices = np.arange(len(train_data)).reshape(-1, 1)
    svm_model.fit(train_indices, train_data)
    return svm_model

# Fit Random Forest model on training data
def fit_random_forest_model(train_data, n_estimators=100):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    train_indices = np.arange(len(train_data)).reshape(-1, 1)
    rf_model.fit(train_indices, train_data)
    return rf_model

# Fit Prophet model on training data
def fit_prophet_model(train_data):
    df_prophet = train_data.reset_index()
    df_prophet.columns = ['ds', 'y']
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)
    return prophet_model
