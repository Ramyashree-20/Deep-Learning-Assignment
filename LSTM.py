import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
import urllib.request
import os
import gc
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def download_airline_dataset():
    """
    Download airline passengers dataset
    """
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    filename = 'airline-passengers.csv'
    
    if not os.path.exists(filename):
        print("Downloading airline passengers dataset...")
        try:
            urllib.request.urlretrieve(url, filename)
            print("Download complete!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    else:
        print("Dataset already exists.")
    
    return filename

dataset_file = download_airline_dataset()
if dataset_file is None:
    print("Failed to download dataset. Exiting.")
    exit(1)

data = pd.read_csv(dataset_file)
print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

dataset = data.iloc[:,1].values
print(f"\nTotal data points: {len(dataset)}")

plt.figure(figsize=(12, 5))
plt.plot(dataset, linewidth=2, color='blue')
plt.xlabel("Time (Months)", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.title("International Airline Passengers (Original Data)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('original_data.png', dpi=300, bbox_inches='tight')
plt.show()

dataset = dataset.reshape(-1, 1)
dataset = dataset.astype("float32") 
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# Train-test split
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train = dataset[0:train_size, :]
test = dataset[train_size:len(dataset), :]
print(f"\nTrain size: {len(train)}, Test size: {len(test)}")

time_stamp = 12 

def create_sequences(data, time_stamp):
    dataX, dataY = [], []
    for i in range(len(data) - time_stamp - 1):
        a = data[i:(i + time_stamp), 0]
        dataX.append(a)
        dataY.append(data[i + time_stamp, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_sequences(train, time_stamp)
testX, testY = create_sequences(test, time_stamp)

# Reshape for LSTM [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

print(f"\nTraining data shape: {trainX.shape}")
print(f"Testing data shape: {testX.shape}")

print("\nBuilding Improved Bidirectional LSTM model...")

# Use simpler model for memory-constrained systems
LITE_MODE = False  # Set to True if you have <4GB RAM

if LITE_MODE:
    print("Using LITE mode (reduced model size)")
    model = Sequential([
        Bidirectional(LSTM(32, return_sequences=True), input_shape=(time_stamp, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(16)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
else:
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(time_stamp, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

model.compile(
    loss='mean_squared_error',
    optimizer=Adam(learning_rate=0.001),
    metrics=['mae']
)

model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1
)

# Train model
print("\nTraining model...")
history = model.fit(
    trainX, trainY,
    epochs=100,
    batch_size=8,
    validation_split=0.1,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Save model architecture
plot_model(model, to_file='improved_lstm_model.png', show_shapes=True, show_layer_names=True)
print("Model architecture saved to 'improved_lstm_model.png'")

# Make predictions
print("\nMaking predictions...")
trainPredict = model.predict(trainX, verbose=0)
testPredict = model.predict(testX, verbose=0)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY_inv = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_inv = scaler.inverse_transform([testY])

# Calculate metrics
trainScore_rmse = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:,0]))
trainScore_mae = mean_absolute_error(trainY_inv[0], trainPredict[:,0])
testScore_rmse = math.sqrt(mean_squared_error(testY_inv[0], testPredict[:,0]))
testScore_mae = mean_absolute_error(testY_inv[0], testPredict[:,0])

print(f'\n=== Model Performance ===')
print(f'Train RMSE: {trainScore_rmse:.2f}')
print(f'Train MAE: {trainScore_mae:.2f}')
print(f'Test RMSE: {testScore_rmse:.2f}')
print(f'Test MAE: {testScore_mae:.2f}')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('MAE', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
print("Training history saved to 'lstm_training_history.png'")
plt.show()
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stamp:len(trainPredict)+time_stamp, :] = trainPredict
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_stamp*2)+1:len(dataset)-1, :] = testPredict
plt.figure(figsize=(14, 6))
plt.plot(scaler.inverse_transform(dataset), label="Actual Data", linewidth=2, color='blue')
plt.plot(trainPredictPlot, label="Training Predictions", linewidth=2, alpha=0.7, color='green')
plt.plot(testPredictPlot, label="Testing Predictions", linewidth=2, alpha=0.7, color='red')
plt.xlabel("Time (Months)", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.title("LSTM Predictions vs Actual Data", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_predictions.png', dpi=300, bbox_inches='tight')
print("Predictions plot saved to 'lstm_predictions.png'")
plt.show()

# Save model
model.save('improved_lstm_airline.h5')
print("\nModel saved to 'improved_lstm_airline.h5'")