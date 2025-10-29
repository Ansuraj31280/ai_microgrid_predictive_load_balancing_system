import argparse
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def create_sliding_windows(features_array: np.ndarray, target_array: np.ndarray, seq_len: int, pred_horizon: int):
    """
    Generate sliding windows for many-to-many forecasting.

    X shape: (num_samples, seq_len, num_features)
    y shape: (num_samples, pred_horizon)
    """
    X, y = [], []
    total_len = seq_len + pred_horizon
    for start_idx in range(0, len(features_array) - total_len + 1):
        end_idx = start_idx + seq_len
        horizon_end = end_idx + pred_horizon
        X.append(features_array[start_idx:end_idx])
        y.append(target_array[end_idx:horizon_end])
    return np.asarray(X), np.asarray(y)


def ensure_directories(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM forecaster for 60-minute load prediction")
    parser.add_argument("--data", default=str(Path("data") / "cleaned_iot_data.csv"), help="Path to cleaned_iot_data.csv")
    parser.add_argument("--target", default="co_net_pow", help="Target column name for load")
    parser.add_argument("--seq_len", type=int, default=12, help="Input sequence length (12 = 60 minutes if 5-min intervals)")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction horizon length (12 = next 60 minutes)")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation fraction from train segment")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test fraction from full series (time-based split)")
    parser.add_argument("--models_dir", default=str(Path("ml_models")), help="Directory to save model and scaler")
    parser.add_argument("--model_name", default="lstm_forecaster.h5", help="Model filename (H5)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    # Paths
    data_path = Path(args.data)
    models_dir = Path(args.models_dir)
    ensure_directories([models_dir])

    # Fail fast if TensorFlow is unavailable (installed separately due to Python 3.13 constraints)
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dropout, Dense
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    except Exception as e:
        print("ERROR: TensorFlow/Keras is required to train the model but is not installed or failed to import.")
        print("Hint: On Python 3.13, TensorFlow may be unavailable. Consider using Python 3.12, then:")
        print("  pip install tensorflow==2.17.0")
        print("Full error:\n", e)
        sys.exit(2)

    if not data_path.exists():
        print(f"ERROR: Data file not found at: {data_path}")
        print("Please place 'cleaned_iot_data.csv' into the 'data' directory and re-run.")
        sys.exit(1)

    # 1) Load data
    df = pd.read_csv(data_path)

    if "timestamp" not in df.columns:
        print("ERROR: Expected a 'timestamp' column in the dataset.")
        sys.exit(1)
    if args.target not in df.columns:
        print(f"ERROR: Target column '{args.target}' not found in dataset.")
        sys.exit(1)

    # Convert timestamp to datetime and set index
    # Try epoch seconds, then milliseconds, then generic parser
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    if ts.isna().mean() > 0.5:
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    if ts.isna().mean() > 0.5:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"])  # drop rows with invalid timestamps
    df = df.set_index("timestamp").sort_index()

    # Handle missing values with time-based interpolation, then ffill/bfill for edges
    df = df.interpolate(method="time").ffill().bfill()

    # Keep only numeric columns for modeling to avoid string columns like 'summary', 'icon'
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    # Ensure target is numeric (coerce if needed)
    if args.target in df.columns and args.target not in numeric_df.columns:
        numeric_df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    numeric_df = numeric_df.dropna(subset=[args.target])

    # Feature selection: all numeric non-target columns as features
    feature_cols = [c for c in numeric_df.columns if c != args.target]
    if len(feature_cols) == 0:
        print("ERROR: No feature columns found aside from target.")
        sys.exit(1)

    # Prevent leakage: time-based split into train and test
    n_total = len(numeric_df)
    test_size = int(n_total * args.test_size)
    if test_size < (args.seq_len + args.horizon + 1):
        test_size = args.seq_len + args.horizon + 1
    train_df = numeric_df.iloc[: n_total - test_size]
    test_df = numeric_df.iloc[n_total - test_size :]

    # Scale using training data only
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols + [args.target]])

    # Save scaler for inference
    scaler_path = models_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "feature_cols": feature_cols,
            "target_col": args.target,
        }, f)
    print(f"Saved scaler to: {scaler_path}")

    # Transform splits
    train_scaled = pd.DataFrame(
        scaler.transform(train_df[feature_cols + [args.target]]),
        index=train_df.index,
        columns=feature_cols + [args.target],
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df[feature_cols + [args.target]]),
        index=test_df.index,
        columns=feature_cols + [args.target],
    )

    # Further split train into train/val (time-aware): last val_size fraction used for val
    n_train = len(train_scaled)
    n_val = int(n_train * args.val_size)
    if n_val < (args.seq_len + args.horizon + 1):
        n_val = args.seq_len + args.horizon + 1
    train_core = train_scaled.iloc[: n_train - n_val]
    val_core = train_scaled.iloc[n_train - n_val :]

    # Arrays
    train_features = train_core[feature_cols].to_numpy(dtype=np.float32)
    train_target = train_core[[args.target]].to_numpy(dtype=np.float32).flatten()

    val_features = val_core[feature_cols].to_numpy(dtype=np.float32)
    val_target = val_core[[args.target]].to_numpy(dtype=np.float32).flatten()

    test_features = test_scaled[feature_cols].to_numpy(dtype=np.float32)
    test_target = test_scaled[[args.target]].to_numpy(dtype=np.float32).flatten()

    # Create sequences (many-to-many for target)
    X_train, y_train_seq = create_sliding_windows(
        train_features, train_target, seq_len=args.seq_len, pred_horizon=args.horizon
    )
    X_val, y_val_seq = create_sliding_windows(
        val_features, val_target, seq_len=args.seq_len, pred_horizon=args.horizon
    )
    X_test, y_test_seq = create_sliding_windows(
        test_features, test_target, seq_len=args.seq_len, pred_horizon=args.horizon
    )

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("ERROR: Not enough data to create sequences. Consider reducing seq_len/horizon or provide more data.")
        sys.exit(1)

    num_features = X_train.shape[-1]

    # 4) Define LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(args.seq_len, num_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(64, activation="relu"),
        Dense(args.horizon),  # predict next horizon values of target
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # 5) Training with callbacks
    model_path = models_dir / args.model_name
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=str(model_path), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train,
        y_train_seq,
        validation_data=(X_val, y_val_seq),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    # Final evaluation
    test_loss = model.evaluate(X_test, y_test_seq, verbose=0)
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Best model saved to: {model_path}")


if __name__ == "__main__":
    main()


