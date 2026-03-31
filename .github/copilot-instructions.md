## Quick orientation

- **Purpose:** This repo contains small TensorFlow/Keras experiments that train multivariate LSTM/GRU models on a local CSV of Delhi weather data and emit model files and diagnostic PNGs.
- **Key scripts:** `train_lstm.py` and `train_gru.py` (both are self-contained scripts with a `main()` entrypoint).
- **Dataset:** `delhi_weather_datasets.csv` — scripts expect this CSV in the repo root and a `datetime_utc` column.

## Architecture & data flow (high level)

- Data ingestion: `train_*.py` reads `delhi_weather_datasets.csv`, trims column names, parses `datetime_utc`, sets it as index.
- Features used: `tempm`, `rain`, `hum`, `wspdm` — the code converts to numeric, forward/back fills missing values, then scales with `sklearn.preprocessing.MinMaxScaler`.
- Sequence creation: fixed sequence length of 24 timesteps; supervised targets are the next timestep for all features (multivariate forecasting).
- Train / test split: first 80% training, last 20% testing; training uses `validation_split=0.1` when fitting.
- Models: `tf.keras.Sequential` networks — LSTM (saved as `lstm_multivariate_model.keras`) and GRU (saved as `gru_multivariate_model.keras`).

## Concrete developer workflows

- Setup (Windows):

  1. Create/activate the venv if needed: `python -m venv lstm_env` then run `lstm_env\\Scripts\\activate.bat` or `Activate.ps1`.
  2. Install deps: `pip install -r requirements.txt` (the repo includes `requirements.txt` with `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`).

- Run training (examples):

  - LSTM: `python train_lstm.py`
  - GRU: `python train_gru.py`

- Outputs produced by runs (repo root): `*.keras` model files and PNGs: `*_training_loss.png`, `*_predictions.png`, `multivariate_training_loss.png`, `multivariate_predictions.png`.

## Important code patterns & conventions

- Scripts are single-file experiments (no argparse). To change hyperparameters, edit the constants in the top half of the script (e.g., `seq_length = 24`, `batch_size=64`, `epochs=25`, `validation_split=0.1`).
- Early stopping is used with `patience=5` and `restore_best_weights=True` — reproductions should respect that when porting code.
- Scaling: predictions are inverse-transformed with the same `MinMaxScaler` instance; don't re-fit the scaler when converting predictions back to original units.

## Troubleshooting hints (project-specific)

- If you get `FileNotFoundError`: ensure `delhi_weather_datasets.csv` is in the repo root and not blocked by another process.
- If you hit `KeyError` for `datetime_utc` or any feature: check column names (scripts call `df.columns = df.columns.str.strip()` to remove stray whitespace).
- If plots are blank or NaNs appear: verify `data.ffill().bfill()` completed and `data.isnull().sum().sum()` equals zero (scripts already assert this).

## Integration & environment notes

- Virtualenv folder: `lstm_env` is present in the repo; use its `Scripts\\activate` on Windows to reproduce the original environment.
- Keras/TensorFlow: the environment includes `keras`/`tensorflow` — prefer the venv's Python executable to avoid version mismatches.

## Where to look for changes

- Model architecture and hyperparameters: `train_lstm.py`, `train_gru.py`.
- Data cleaning and expected columns: top of both training scripts (look for `features = [...]` and `datetime_utc` parsing).

---
If anything here is unclear or you want more detail (for example, a recommended CLI wrapper, unit tests for data-processing, or a small evaluation script), tell me which area to expand. 
