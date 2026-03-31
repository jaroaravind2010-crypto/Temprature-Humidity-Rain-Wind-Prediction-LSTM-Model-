<<<<<<< HEAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


def main():
    print("1. Loading data...")

    file_path = "delhi_weather_datasets.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found in the current directory.")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    print("2. Cleaning data...")

    if "datetime_utc" not in df.columns:
        raise KeyError("'datetime_utc' column not found in dataset.")

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
    df = df.dropna(subset=["datetime_utc"])

    df.set_index("datetime_utc", inplace=True)
    df.sort_index(inplace=True)

    features = ["tempm", "rain", "hum", "wspdm"]
    feature_labels = [
        "Temperature (°C)",
        "Rain (mm)",
        "Humidity (%)",
        "Wind Speed (km/h)"
    ]

    for feature in features:
        if feature not in df.columns:
            raise KeyError(f"'{feature}' column not found in dataset.")

    data = df[features].copy()

    # Convert to numeric
    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Fill missing values
    data = data.ffill().bfill()

    if data.isnull().sum().sum() > 0:
        raise ValueError("Data still contains NaN values.")

    dataset = data.values
    timestamps = data.index

    print("3. Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # ✅ Save scaler for use in app.py
    joblib.dump(scaler, "scaler.pkl")
    print("  ✓ Scaler saved to 'scaler.pkl'")

    print("4. Creating sequences...")
    seq_length = 48
    forecast_horizon = 12   # ← 12-hour forecast

    X, y = [], []

    for i in range(len(scaled_data) - seq_length - forecast_horizon + 1):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length: i + seq_length + forecast_horizon])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("Not enough data to create sequences.")

    train_size = int(len(X) * 0.8)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    num_features = X_train.shape[2]

    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing shapes:  X={X_test.shape}, y={y_test.shape}")

    print("5. Building LSTM Model...")

    model = Sequential([
        Input(shape=(seq_length, num_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(forecast_horizon * num_features),
        tf.keras.layers.Reshape((forecast_horizon, num_features))
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    print("6. Training model...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=25,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    print("7. Evaluating model...")
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {loss:.6f}")

    model_path = "lstm_multivariate_model.keras"
    model.save(model_path)
    print(f"  ✓ Model saved to '{model_path}'")

    print("8. Generating predictions...")
    predictions = model.predict(X_test)

    predictions_2d = predictions.reshape(-1, num_features)
    y_test_2d = y_test.reshape(-1, num_features)

    predictions_actual = scaler.inverse_transform(predictions_2d).reshape(predictions.shape)
    y_test_actual = scaler.inverse_transform(y_test_2d).reshape(y_test.shape)

    # ─────────────────────────────────────────────
    # Plot 1: Training vs Validation Loss
    # ─────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"],     label="Train Loss",      linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Training vs Validation Loss (12-Hour Horizon)", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("multivariate_training_loss.png", dpi=150)
    plt.close()
    print("  ✓ Saved: multivariate_training_loss.png")

    # ─────────────────────────────────────────────
    # Plot 2: Actual vs Predicted — Test Set
    # ─────────────────────────────────────────────
    plot_samples = min(500, len(y_test_actual))

    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    fig.suptitle(
        "Actual vs Predicted — 1st Hour of 12-Hour Forecast (Test Set)",
        fontsize=16, fontweight="bold"
    )

    for i in range(num_features):
        axes[i].plot(y_test_actual[:plot_samples, 0, i],
                     label="Actual",    color="#2196F3", alpha=0.85, linewidth=1.5)
        axes[i].plot(predictions_actual[:plot_samples, 0, i],
                     label="Predicted", color="#FF5722", alpha=0.85, linewidth=1.5, linestyle="--")
        axes[i].set_title(feature_labels[i], fontsize=12)
        axes[i].set_ylabel(feature_labels[i])
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Test Sample Index")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig("multivariate_predictions.png", dpi=150)
    plt.close()
    print("  ✓ Saved: multivariate_predictions.png")

    # ─────────────────────────────────────────────
    # Plot 3: Next 12 Hours — Future Prediction
    # ─────────────────────────────────────────────
    print("9. Generating NEXT 12-HOUR future prediction...")

    last_sequence       = scaled_data[-seq_length:]            # (48, 4)
    last_sequence_input = last_sequence[np.newaxis, ...]       # (1, 48, 4)

    next_12_scaled    = model.predict(last_sequence_input)     # (1, 12, 4)
    next_12_scaled_2d = next_12_scaled[0]                      # (12, 4)
    next_12_actual    = scaler.inverse_transform(next_12_scaled_2d)  # (12, 4)

    last_timestamp   = timestamps[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=forecast_horizon,
        freq="h"
    )

    context_timestamps = timestamps[-seq_length:]
    context_actual     = dataset[-seq_length:]

    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    fig.suptitle(
        "Next 12-Hour Weather Forecast from Last Known Data",
        fontsize=16, fontweight="bold"
    )

    for i in range(num_features):
        axes[i].plot(context_timestamps, context_actual[:, i],
                     label="Historical (last 48 h)", color="#2196F3", linewidth=1.5, alpha=0.7)
        axes[i].plot(future_timestamps, next_12_actual[:, i],
                     label="Predicted (next 12 h)", color="#FF5722", linewidth=2,
                     linestyle="--", marker="o", markersize=5)
        axes[i].axvspan(future_timestamps[0], future_timestamps[-1], alpha=0.08, color="#FF5722")
        axes[i].set_title(feature_labels[i], fontsize=12)
        axes[i].set_ylabel(feature_labels[i])
        axes[i].legend(loc="upper left")
        axes[i].grid(True, alpha=0.3)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        axes[i].tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig("next_12_hour_prediction.png", dpi=150)
    plt.close()
    print("  ✓ Saved: next_12_hour_prediction.png")

    # ─────────────────────────────────────────────
    # Print forecast table
    # ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("       NEXT 12-HOUR WEATHER FORECAST")
    print("=" * 65)
    print(f"  {'Hour':<22} {'Temp(°C)':>10} {'Rain(mm)':>10} {'Hum(%)':>8} {'Wind(km/h)':>12}")
    print("-" * 65)
    for idx, ts in enumerate(future_timestamps):
        row = next_12_actual[idx]
        print(f"  {str(ts):<22} {row[0]:>10.2f} {row[1]:>10.2f} {row[2]:>8.2f} {row[3]:>12.2f}")
    print("=" * 65)

    print("\n✅ All files saved:")
    print("   • lstm_multivariate_model.keras")
    print("   • scaler.pkl                  ← used by app.py")
    print("   • multivariate_training_loss.png")
    print("   • multivariate_predictions.png")
    print("   • next_12_hour_prediction.png")


if __name__ == "__main__":
=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


def main():
    print("1. Loading data...")

    file_path = "delhi_weather_datasets.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found in the current directory.")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    print("2. Cleaning data...")

    if "datetime_utc" not in df.columns:
        raise KeyError("'datetime_utc' column not found in dataset.")

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
    df = df.dropna(subset=["datetime_utc"])

    df.set_index("datetime_utc", inplace=True)
    df.sort_index(inplace=True)

    features = ["tempm", "rain", "hum", "wspdm"]
    feature_labels = [
        "Temperature (°C)",
        "Rain (mm)",
        "Humidity (%)",
        "Wind Speed (km/h)"
    ]

    for feature in features:
        if feature not in df.columns:
            raise KeyError(f"'{feature}' column not found in dataset.")

    data = df[features].copy()

    # Convert to numeric
    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Fill missing values
    data = data.ffill().bfill()

    if data.isnull().sum().sum() > 0:
        raise ValueError("Data still contains NaN values.")

    dataset = data.values
    timestamps = data.index

    print("3. Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # ✅ Save scaler for use in app.py
    joblib.dump(scaler, "scaler.pkl")
    print("  ✓ Scaler saved to 'scaler.pkl'")

    print("4. Creating sequences...")
    seq_length = 48
    forecast_horizon = 12   # ← 12-hour forecast

    X, y = [], []

    for i in range(len(scaled_data) - seq_length - forecast_horizon + 1):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length: i + seq_length + forecast_horizon])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("Not enough data to create sequences.")

    train_size = int(len(X) * 0.8)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    num_features = X_train.shape[2]

    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing shapes:  X={X_test.shape}, y={y_test.shape}")

    print("5. Building LSTM Model...")

    model = Sequential([
        Input(shape=(seq_length, num_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(forecast_horizon * num_features),
        tf.keras.layers.Reshape((forecast_horizon, num_features))
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    print("6. Training model...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=25,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    print("7. Evaluating model...")
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {loss:.6f}")

    model_path = "lstm_multivariate_model.keras"
    model.save(model_path)
    print(f"  ✓ Model saved to '{model_path}'")

    print("8. Generating predictions...")
    predictions = model.predict(X_test)

    predictions_2d = predictions.reshape(-1, num_features)
    y_test_2d = y_test.reshape(-1, num_features)

    predictions_actual = scaler.inverse_transform(predictions_2d).reshape(predictions.shape)
    y_test_actual = scaler.inverse_transform(y_test_2d).reshape(y_test.shape)

    # ─────────────────────────────────────────────
    # Plot 1: Training vs Validation Loss
    # ─────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"],     label="Train Loss",      linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Training vs Validation Loss (12-Hour Horizon)", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("multivariate_training_loss.png", dpi=150)
    plt.close()
    print("  ✓ Saved: multivariate_training_loss.png")

    # ─────────────────────────────────────────────
    # Plot 2: Actual vs Predicted — Test Set
    # ─────────────────────────────────────────────
    plot_samples = min(500, len(y_test_actual))

    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    fig.suptitle(
        "Actual vs Predicted — 1st Hour of 12-Hour Forecast (Test Set)",
        fontsize=16, fontweight="bold"
    )

    for i in range(num_features):
        axes[i].plot(y_test_actual[:plot_samples, 0, i],
                     label="Actual",    color="#2196F3", alpha=0.85, linewidth=1.5)
        axes[i].plot(predictions_actual[:plot_samples, 0, i],
                     label="Predicted", color="#FF5722", alpha=0.85, linewidth=1.5, linestyle="--")
        axes[i].set_title(feature_labels[i], fontsize=12)
        axes[i].set_ylabel(feature_labels[i])
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Test Sample Index")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig("multivariate_predictions.png", dpi=150)
    plt.close()
    print("  ✓ Saved: multivariate_predictions.png")

    # ─────────────────────────────────────────────
    # Plot 3: Next 12 Hours — Future Prediction
    # ─────────────────────────────────────────────
    print("9. Generating NEXT 12-HOUR future prediction...")

    last_sequence       = scaled_data[-seq_length:]            # (48, 4)
    last_sequence_input = last_sequence[np.newaxis, ...]       # (1, 48, 4)

    next_12_scaled    = model.predict(last_sequence_input)     # (1, 12, 4)
    next_12_scaled_2d = next_12_scaled[0]                      # (12, 4)
    next_12_actual    = scaler.inverse_transform(next_12_scaled_2d)  # (12, 4)

    last_timestamp   = timestamps[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=forecast_horizon,
        freq="h"
    )

    context_timestamps = timestamps[-seq_length:]
    context_actual     = dataset[-seq_length:]

    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    fig.suptitle(
        "Next 12-Hour Weather Forecast from Last Known Data",
        fontsize=16, fontweight="bold"
    )

    for i in range(num_features):
        axes[i].plot(context_timestamps, context_actual[:, i],
                     label="Historical (last 48 h)", color="#2196F3", linewidth=1.5, alpha=0.7)
        axes[i].plot(future_timestamps, next_12_actual[:, i],
                     label="Predicted (next 12 h)", color="#FF5722", linewidth=2,
                     linestyle="--", marker="o", markersize=5)
        axes[i].axvspan(future_timestamps[0], future_timestamps[-1], alpha=0.08, color="#FF5722")
        axes[i].set_title(feature_labels[i], fontsize=12)
        axes[i].set_ylabel(feature_labels[i])
        axes[i].legend(loc="upper left")
        axes[i].grid(True, alpha=0.3)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        axes[i].tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig("next_12_hour_prediction.png", dpi=150)
    plt.close()
    print("  ✓ Saved: next_12_hour_prediction.png")

    # ─────────────────────────────────────────────
    # Print forecast table
    # ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("       NEXT 12-HOUR WEATHER FORECAST")
    print("=" * 65)
    print(f"  {'Hour':<22} {'Temp(°C)':>10} {'Rain(mm)':>10} {'Hum(%)':>8} {'Wind(km/h)':>12}")
    print("-" * 65)
    for idx, ts in enumerate(future_timestamps):
        row = next_12_actual[idx]
        print(f"  {str(ts):<22} {row[0]:>10.2f} {row[1]:>10.2f} {row[2]:>8.2f} {row[3]:>12.2f}")
    print("=" * 65)

    print("\n✅ All files saved:")
    print("   • lstm_multivariate_model.keras")
    print("   • scaler.pkl                  ← used by app.py")
    print("   • multivariate_training_loss.png")
    print("   • multivariate_predictions.png")
    print("   • next_12_hour_prediction.png")


if __name__ == "__main__":
>>>>>>> ba5d9fe8fee6dff968b8932933f6a352e8b72da6
    main()