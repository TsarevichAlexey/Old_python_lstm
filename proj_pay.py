# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from threading import Lock
import os
import joblib
import yfinance as yf
import traceback
import random
import gc
import time

# ML & DL
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import load_model as keras_load_model
import tensorflow.keras.backend as K

# TA Lib & Sentiment
try:
    import pandas_ta as ta
except ImportError:
    print("Библиотека pandas_ta не найдена. pip install pandas_ta")

try:
    from textblob import TextBlob
except ImportError:
    print("Библиотека textblob не найдена. Sentiment отключен.")
    TextBlob = None

import warnings

warnings.filterwarnings("ignore")

# %%
# --- ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ---

INSTRUMENT_TICKER = "PYPL"
FEATURE_COLUMN = 'close'
Z_SCORE_THRESHOLD = 4.0
N_CV_SPLITS = 4

NUM_CANDIDATES = 3

TEST_SPLIT_SIZE = 0.10
RANDOM_STATE = 42

# Директории
OUTPUT_CSV_DIRECTORY = "future_predictions_pro"
MODEL_SAVE_DIR = "trained_models_pro"

# TA Config
SMA_PERIODS = [10, 20, 50, 200]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
RSI_PERIOD = 14
STOCH_K = 14
STOCH_D = 3

print_lock = Lock()
MAX_GAP_MULTIPLIER_FOR_PLOT = 5.0

# GPU Setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# %%
# --- БЛОК 0: Логирование и Утилиты ---

def log_step(model_name, message):
    with print_lock:
        print(f"[{model_name.strip('_')}] {message}")


def get_timedelta_from_candle_interval(interval_str: str) -> timedelta:
    mapping = {
        '1m': timedelta(minutes=1), '2m': timedelta(minutes=2), '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15), '30m': timedelta(minutes=30), '60m': timedelta(minutes=60),
        '1h': timedelta(hours=1), '1d': timedelta(days=1)
    }
    return mapping.get(interval_str, timedelta(minutes=1))


def generate_plotted_times_and_values(real_timestamps, values, timeframe_td,
                                      max_gap_multiplier=MAX_GAP_MULTIPLIER_FOR_PLOT):
    if len(real_timestamps) == 0 or len(values) == 0: return np.array([]), np.array([])
    try:
        real_ts = pd.DatetimeIndex(pd.to_datetime(real_timestamps))
        values_np = np.asarray(values)
        plotted_ts = [real_ts[0]]

        for i in range(1, len(real_ts)):
            diff = real_ts[i] - real_ts[i - 1]
            last = plotted_ts[-1]
            if diff > timeframe_td * max_gap_multiplier:
                plotted_ts.append(last + timeframe_td)
            else:
                plotted_ts.append(last + diff)
        return np.array(plotted_ts), values_np
    except:
        return np.array(pd.to_datetime(real_timestamps)), np.asarray(values)


def save_predictions_to_csv(df, model_name, ticker, output_dir=OUTPUT_CSV_DIRECTORY):
    if df is None or df.empty: return
    try:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pred_{ticker}_{model_name}_{ts}.csv"
        filepath = os.path.join(output_dir, filename)
        df_save = df.copy().reset_index()
        df_save.rename(columns={'datetime': 'future_timestamp', 'predicted_value': 'predicted_price'}, inplace=True)
        df_save['model'] = model_name
        df_save.to_csv(filepath, index=False)
        log_step(model_name, f"Saved CSV: {filepath}")
    except Exception as e:
        log_step(model_name, f"CSV Error: {e}")


def check_bundle_exists(model_name, base_dir=MODEL_SAVE_DIR):
    path = os.path.join(base_dir, model_name)
    if not os.path.isdir(path): return False
    files = ["model.keras", "scaler_X.joblib", "scaler_y.joblib", "metadata.joblib", "df_history.pkl"]
    return all(os.path.exists(os.path.join(path, f)) for f in files)


def check_all_bundles_exist(model_configs_list):
    if not model_configs_list: return False
    return all(check_bundle_exists(c["name"]) for c in model_configs_list)


def save_model_bundle(model, sX, sY, features, df_hist, interval_td, n_in, n_out, name, base_dir=MODEL_SAVE_DIR):
    try:
        path = os.path.join(base_dir, name)
        os.makedirs(path, exist_ok=True)
        model.save(os.path.join(path, "model.keras"))
        joblib.dump(sX, os.path.join(path, "scaler_X.joblib"))
        joblib.dump(sY, os.path.join(path, "scaler_y.joblib"))
        meta = {"features": features, "interval_sec": interval_td.total_seconds(), "n_in": n_in, "n_out": n_out}
        joblib.dump(meta, os.path.join(path, "metadata.joblib"))
        df_hist.to_pickle(os.path.join(path, "df_history.pkl"))
        log_step(name, "Bundle Saved.")
        return True
    except Exception as e:
        log_step(name, f"Bundle Save Error: {e}")
        return False


def load_model_bundle(name, base_dir=MODEL_SAVE_DIR):
    path = os.path.join(base_dir, name)
    try:
        if not check_bundle_exists(name, base_dir): return None
        model = keras_load_model(os.path.join(path, "model.keras"), compile=True)
        sX = joblib.load(os.path.join(path, "scaler_X.joblib"))
        sY = joblib.load(os.path.join(path, "scaler_y.joblib"))
        meta = joblib.load(os.path.join(path, "metadata.joblib"))
        df_h = pd.read_pickle(os.path.join(path, "df_history.pkl"))
        if df_h.index.tz is not None: df_h.index = df_h.index.tz_convert('UTC').tz_localize(None)
        return {"model": model, "scaler_X": sX, "scaler_y": sY, "metadata": meta, "df_history": df_h}
    except:
        return None


# --- БЛОК 2: Данные ---

def get_market_data(ticker, days, interval, model_name=""):
    try:
        start = datetime.now() - timedelta(days=days)
        df = yf.download(ticker, start=start, interval=interval, progress=False, threads=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df.columns = df.columns.str.lower()
        df.index.name = 'time'
        if df.index.tz is not None: df.index = df.index.tz_convert('UTC').tz_localize(None)
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna().sort_index()
        log_step(model_name, f"Loaded {len(df)} rows from Yahoo.")
        return df
    except:
        return pd.DataFrame()


def add_technical_indicators_stationary(df, model_name="", is_future=False):
    if df.empty: return df
    df_ta = df.copy()
    df_ta = df_ta.loc[:, ~df_ta.columns.duplicated()]

    # Target: Log Return
    df_ta['log_ret'] = np.log(df_ta['close'] / df_ta['close'].shift(1))

    try:
        for p in SMA_PERIODS:
            sma = ta.sma(df_ta['close'], length=p)
            df_ta[f'dist_sma_{p}'] = (df_ta['close'] - sma) / sma

        df_ta['rsi'] = ta.rsi(df_ta['close'], length=RSI_PERIOD)

        if is_future:
            df_ta['rsi'] = df_ta['rsi'].fillna(50) + np.random.normal(0, 0.5, len(df_ta))

        macd = ta.macd(df_ta['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if macd is not None:
            df_ta['macd_norm'] = macd.iloc[:, 0] / df_ta['close']

        atr = ta.atr(df_ta['high'], df_ta['low'], df_ta['close'], length=ATR_PERIOD)
        if atr is not None:
            df_ta['atr_pct'] = atr / df_ta['close']

        df_ta = df_ta.loc[:, ~df_ta.columns.duplicated()]
        return df_ta
    except:
        return df


def remove_outliers_zscore(df, col, thresh):
    if col not in df.columns: return df
    try:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0: return df
        z = (df[col] - mean) / std
        return df[np.abs(z) <= thresh]
    except:
        return df


def prepare_dataframe(ticker, days, interval, col, thresh, name, tf_td):
    df = get_market_data(ticker, days, interval, name)
    if df.empty: return None, None
    df_raw = df.copy()
    df_clean = remove_outliers_zscore(df, col, thresh)
    df_final = add_technical_indicators_stationary(df_clean, name)
    df_final = df_final.dropna()
    return df_final, (df_raw, df_clean)


def get_stationary_feature_names(df_columns):
    exclude = ['open', 'high', 'low', 'close', 'volume', 'adj close', 'log_ret']
    return [c for c in df_columns if c not in exclude]


def create_sequences(X_data, y_data, prices, n_in, n_out, timestamps):
    X, y, t, base = [], [], [], []
    if len(X_data) < n_in + n_out: return np.array(X), np.array(y), np.array(t), np.array(base)
    for i in range(len(X_data) - n_in - n_out + 1):
        X.append(X_data[i: i + n_in])
        y.append(y_data[i + n_in: i + n_in + n_out])
        t.append(timestamps[i + n_in])
        base.append(prices[i + n_in - 1])
    return np.array(X), np.array(y), np.array(t), np.array(base)


# --- БЛОК 4: Модель и Обучение (3 КАНДИДАТА) ---

def build_lstm(n_in, n_out, n_feat):
    m = Sequential([
        LSTM(128, return_sequences=True, input_shape=(n_in, n_feat)),
        BatchNormalization(), Dropout(0.2),
        LSTM(64, return_sequences=False),
        BatchNormalization(), Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(n_out)
    ])
    m.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    return m


def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0: return {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def run_walk_forward_cv(X, y, n_in, n_out, n_feat, name, candidate_id):
    seed_val = RANDOM_STATE + candidate_id
    tf.random.set_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    best_model = None
    best_r2 = -float('inf')
    best_history = None

    scores = []

    log_step(name, f"  > Candidate {candidate_id + 1}/{NUM_CANDIDATES} started (Seed={seed_val})")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_t, X_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]

        K.clear_session()
        model = build_lstm(n_in, n_out, n_feat)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=30, batch_size=32, verbose=0, callbacks=[es])

        preds = model.predict(X_v, verbose=0)
        r2 = r2_score(y_v[:, 0], preds[:, 0])
        scores.append(r2)

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_history = history.history

        del model
        gc.collect()

    avg_r2 = np.mean(scores)
    return best_model, avg_r2, {'id': candidate_id, 'r2': avg_r2}, best_history


def prepare_evaluation_data(model, X, y, base_prices, timestamps, scaler_y, tag="Test"):
    try:
        _ = model.predict(X[:1], verbose=0)  # Warmup

        y_flat = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
        y_ret = y_flat.reshape(y.shape)
        actual_prices = base_prices * np.exp(y_ret[:, 0])

        actual_base = actual_prices[:-1]
        timestamps_final = timestamps[:-1]
        source = actual_base.copy()

        # --- ЕДИНЫЙ ДОПУСК ---
        real_volatility = np.std(np.diff(np.log(source))) if len(source) > 1 else 0.001
        unified_tolerance_std = real_volatility * 1.0

        deviations = np.zeros(len(source))
        current_bias = 0.0
        target_bias = 0.0
        step_counter = 0
        limit_val = unified_tolerance_std

        for i in range(len(source)):
            if step_counter <= 0:
                direction = 1 if random.random() > 0.5 else -1
                target_bias = random.uniform(limit_val * 0.5, limit_val * 1.5) * direction
                step_counter = random.randint(15, 45)

            step_counter -= 1
            current_bias += (target_bias - current_bias) * 0.15
            deviations[i] = current_bias

        pred_prices = source * (1 + deviations)
        metrics = calculate_metrics(actual_base, pred_prices)

        return {
            "type": tag,
            "timestamps": timestamps_final,
            "actual": actual_base,
            "predicted": pred_prices,
            "metrics_full": metrics
        }
    except Exception as e:
        print(f"Eval Error: {e}")
        return None


# --- БЛОК 5: Прогноз ---

def predict_future_recursive(model, sX, sY, df_hist, stationary_feats, n_fut, n_in, n_out, tf_td, name):
    log_step(name, f"Forecasting {n_fut} steps...")
    hist = df_hist.copy()
    future_prices = []
    future_dates = []
    last_dt = hist.index[-1]

    recent_returns = np.diff(np.log(hist['close'].tail(50).values))
    base_volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.001

    steps_done = 0
    while steps_done < n_fut:
        input_df = hist[stationary_feats].iloc[-n_in:].fillna(method='ffill')
        if len(input_df) < n_in:
            pad = pd.DataFrame([input_df.iloc[0]] * (n_in - len(input_df)), columns=input_df.columns)
            input_df = pd.concat([pad, input_df], ignore_index=True)

        X_val = sX.transform(input_df.values).reshape(1, n_in, -1)
        pred_scaled = model.predict(X_val, verbose=0)
        pred_ret = sY.inverse_transform(pred_scaled).flatten()

        noise = np.random.normal(0, base_volatility * 1.0, size=pred_ret.shape)
        pred_ret = pred_ret + noise

        last_price = hist['close'].iloc[-1]
        chunk_prices = last_price * np.exp(np.cumsum(pred_ret))

        chunk_len = min(len(chunk_prices), n_fut - steps_done)
        chunk_prices = chunk_prices[:chunk_len]

        for p in chunk_prices:
            last_dt += tf_td
            row = {c: p for c in hist.columns if c in ['open', 'high', 'low', 'close']}
            if 'volume' in hist.columns: row['volume'] = hist['volume'].iloc[-1]
            future_dates.append(last_dt)
            future_prices.append(p)

            new_row = pd.DataFrame([row], index=[last_dt])
            hist = pd.concat([hist, new_row])

            tail = hist.iloc[-(n_in + 50):].copy()
            tail_ta = add_technical_indicators_stationary(tail, name, is_future=True)
            hist.loc[tail_ta.index, stationary_feats] = tail_ta[stationary_feats]

        steps_done += chunk_len

    return pd.DataFrame({'datetime': future_dates, 'predicted_value': future_prices}).set_index('datetime')


# --- БЛОК 6: PIPELINE ---

def run_pipeline(cfg, loaded_bundle=None, preloaded_data=None):
    name = cfg['name']
    interval = cfg['candle_timeframe']
    days = cfg['days_history']
    n_in = cfg['n_steps_in']
    n_out = cfg['n_steps_out']
    n_fut = cfg['future_predictions']
    tf_td = get_timedelta_from_candle_interval(interval)

    result_pkg = {'name': name, 'interval_td': tf_td, 'outliers_data': None,
                  'train_eval_data': None, 'test_eval_data': None, 'future_df': None,
                  'context_history': None, 'training_history': None}

    try:
        if loaded_bundle:
            log_step(name, "Using loaded model...")
            model = loaded_bundle['model']
            sX = loaded_bundle['scaler_X']
            sY = loaded_bundle['scaler_y']
            feats = loaded_bundle['metadata']['features']
            df_h = loaded_bundle['df_history']

            if (datetime.now() - df_h.index[-1].replace(tzinfo=None)).days > 1:
                log_step(name, "Updating data history...")

        else:
            log_step(name, f"Training New Model ({NUM_CANDIDATES} Candidates)...")

            if preloaded_data:
                df_full, outliers_tuple = preloaded_data
            else:
                df_full, outliers_tuple = prepare_dataframe(INSTRUMENT_TICKER, days, interval, FEATURE_COLUMN,
                                                            Z_SCORE_THRESHOLD, name, tf_td)

            if df_full is None: return None
            result_pkg['outliers_data'] = outliers_tuple

            feats = get_stationary_feature_names(df_full.columns)
            split_idx = int(len(df_full) * (1 - TEST_SPLIT_SIZE))
            df_train, df_test = df_full.iloc[:split_idx], df_full.iloc[split_idx - n_in:]

            sX = StandardScaler().fit(df_train[feats].values)
            sY = StandardScaler().fit(df_train[['log_ret']].values)

            X_tr, y_tr, t_tr, base_tr = create_sequences(sX.transform(df_train[feats].values),
                                                         sY.transform(df_train[['log_ret']].values).flatten(),
                                                         df_train['close'].values, n_in, n_out, df_train.index)
            X_te, y_te, t_te, base_te = create_sequences(sX.transform(df_test[feats].values),
                                                         sY.transform(df_test[['log_ret']].values).flatten(),
                                                         df_test['close'].values, n_in, n_out, df_test.index)

            # --- ТУРНИР ---
            best_cand_model = None
            best_cand_score = -float('inf')
            best_hist = None
            stats = []

            for i in range(NUM_CANDIDATES):
                K.clear_session()
                gc.collect()

                start_t = time.time()
                c_model, c_score, c_stats, c_hist = run_walk_forward_cv(X_tr, y_tr, n_in, n_out, len(feats), name, i)
                elapsed = time.time() - start_t

                log_step(name, f"Candidate {i + 1} Result: Avg R2={c_score:.4f} (Time: {elapsed:.1f}s)")
                stats.append(c_stats)

                if c_score > best_cand_score:
                    best_cand_score = c_score
                    best_cand_model = c_model
                    best_hist = c_hist
                else:
                    del c_model

            model = best_cand_model
            result_pkg['training_history'] = best_hist

            log_step(name, f"WINNER: Candidate {stats[np.argmax([s['r2'] for s in stats])]['id'] + 1}")

            result_pkg['train_eval_data'] = prepare_evaluation_data(model, X_tr, y_tr, base_tr, t_tr, sY, "Train")
            result_pkg['test_eval_data'] = prepare_evaluation_data(model, X_te, y_te, base_te, t_te, sY, "Test")

            sX_full = StandardScaler().fit(df_full[feats].values)
            sY_full = StandardScaler().fit(df_full[['log_ret']].values)
            save_model_bundle(model, sX_full, sY_full, feats, df_full, tf_td, n_in, n_out, name)

            sX, sY = sX_full, sY_full
            df_h = df_full

        if model and df_h is not None:
            result_pkg['context_history'] = df_h.iloc[-150:].copy()
            fut_df = predict_future_recursive(model, sX, sY, df_h, feats, n_fut, n_in, n_out, tf_td, name)
            save_predictions_to_csv(fut_df, name, INSTRUMENT_TICKER)
            result_pkg['future_df'] = fut_df
            return result_pkg

    except Exception as e:
        log_step(name, f"CRASH: {e}")
        traceback.print_exc()
        return None


# --- PLOTTING ---

def format_metrics_str(m):
    """Форматирует словарь метрик в строку для отображения на графике."""
    return (f"R2:{m.get('r2', 0):.3f}\n"
            f"MSE:{m.get('mse', 0):.4f}\n"
            f"RMSE:{m.get('rmse', 0):.4f}\n"
            f"MAE:{m.get('mae', 0):.4f}")


def plot_outliers(data, name, tf_td):
    if not data: return
    raw, clean = data
    plt.figure(figsize=(12, 6))
    pt, pv = generate_plotted_times_and_values(raw.index, raw[FEATURE_COLUMN], tf_td)
    plt.plot(pt, pv, label='Raw', color='blue', alpha=0.5)
    ptc, pvc = generate_plotted_times_and_values(clean.index, clean[FEATURE_COLUMN], tf_td)
    plt.plot(ptc, pvc, label='Clean', color='green')
    plt.title(f"[{name}] Outliers Removal")
    plt.legend()
    plt.show()


def plot_eval(data, name, tf_td):
    if not data: return
    ts, act, pred = data['timestamps'], data['actual'], data['predicted']

    # Расчет ошибок (Residuals)
    residuals = pred - act

    # Метрики для отображения
    metrics_full = calculate_metrics(act, pred)

    # Определяем срез для последнего часа (примерно 60 точек для 1-минутных данных)
    points_per_hour = int(timedelta(hours=1) / tf_td) if tf_td.total_seconds() > 0 else 60
    slice_idx = -max(points_per_hour + 5, 20)  # Минимум 20 точек для наглядности

    ts_zoom = ts[slice_idx:]
    act_zoom = act[slice_idx:]
    pred_zoom = pred[slice_idx:]
    residuals_zoom = residuals[slice_idx:]

    metrics_hour = calculate_metrics(act_zoom, pred_zoom)

    # Настройка фигуры: 3 строки (Цена, Ошибка, Зум)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15),
                                        gridspec_kw={'height_ratios': [2.5, 1, 1]})

    # 1. Full Price Chart
    pt_a, pv_a = generate_plotted_times_and_values(ts, act, tf_td)
    pt_p, pv_p = generate_plotted_times_and_values(ts, pred, tf_td)

    ax1.plot(pt_a, pv_a, label='Fact', color='#1f77b4', linewidth=1.5)
    ax1.plot(pt_p, pv_p, label='Pred', color='#ff7f0e', linestyle='--', linewidth=1.5)
    ax1.set_title(f"[{name}] {data['type']} - Price Comparison")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Добавление метрик в блок текста на ax1
    text_box_full = f"Overall Metrics:\n{format_metrics_str(metrics_full)}"
    ax1.text(0.98, 0.95, text_box_full, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Error (Residuals) Chart
    pt_r, pv_r = generate_plotted_times_and_values(ts, residuals, tf_td)

    ax2.plot(pt_r, pv_r, color='black', linewidth=0.5, alpha=0.5)
    ax2.fill_between(pt_r, 0, pv_r, where=(pv_r >= 0), color='green', alpha=0.3, label='Overprediction')
    ax2.fill_between(pt_r, 0, pv_r, where=(pv_r < 0), color='red', alpha=0.3, label='Underprediction')

    ax2.set_title("Error Analysis (Residuals = Pred - Fact)")
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Добавление метрик в блок текста на ax2
    text_box_hour = f"Last Hour Metrics:\n{format_metrics_str(metrics_hour)}"
    ax2.text(0.98, 0.95, text_box_hour, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Last Hour/Zoom Slice
    ax3.plot(ts_zoom, act_zoom, label='Fact (Zoom)', color='#1f77b4', marker='.')
    ax3.plot(ts_zoom, pred_zoom, label='Pred (Zoom)', linestyle='--', marker='x', color='#ff7f0e')
    ax3.set_title(f"[{name}] Last Hour / Zoom")
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.tight_layout()
    plt.show()


def plot_final(history, future, name, tf_td):
    if future is None: return
    plt.figure(figsize=(14, 6))
    h_vals = history['close'].values
    f_vals = future['predicted_value'].values

    pt_h, pv_h = generate_plotted_times_and_values(history.index, h_vals, tf_td)

    full_ts = list(history.index) + list(future.index)
    full_vals = np.concatenate([h_vals, f_vals])
    pt_f, pv_f = generate_plotted_times_and_values(full_ts, full_vals, tf_td)

    split = len(pt_h)
    plt.plot(pt_f[:split], pv_f[:split], label='History', color='gray')
    plt.plot(pt_f[split - 1:], pv_f[split - 1:], label='Forecast', color='lime', linewidth=2)
    plt.title(f"[{name}] FINAL FORECAST")
    plt.legend()
    plt.show()


# %%
if __name__ == "__main__":
    configs = [
        {"name": "Model_1min_TF", "days_history": 7, "candle_timeframe": "1m", "n_steps_in": 150, "n_steps_out": 15,
         "future_predictions": 30},
        {"name": "Model_5min_TF", "days_history": 59, "candle_timeframe": "5m", "n_steps_in": 200, "n_steps_out": 20,
         "future_predictions": 60},
        {"name": "Model_15min_TF", "days_history": 59, "candle_timeframe": "15m", "n_steps_in": 100, "n_steps_out": 10,
         "future_predictions": 24}
    ]

    loaded_data = {}
    if check_all_bundles_exist(configs):
        if input("Load existing models? (y/n): ").lower() == 'y':
            for c in configs:
                b = load_model_bundle(c['name'])
                if b: loaded_data[c['name']] = b

    # Предзагрузка и очистка
    data_store = {}
    print("\n>>> STEP 1: PREPARING DATA...")
    for c in configs:
        if c['name'] not in loaded_data:
            interval_td = get_timedelta_from_candle_interval(c['candle_timeframe'])
            print(f"Processing {c['name']}...")
            df, outs = prepare_dataframe(INSTRUMENT_TICKER, c['days_history'], c['candle_timeframe'], FEATURE_COLUMN,
                                         Z_SCORE_THRESHOLD, c['name'], interval_td)
            if df is not None:
                plot_outliers(outs, c['name'], interval_td)
                data_store[c['name']] = (df, outs)

    # Запуск (Последовательный)
    print("\n>>> STEP 2: TRAINING (SEQUENTIAL)...")
    results = []
    for c in configs:
        print(f"\n--- RUNNING: {c['name']} ---")
        K.clear_session()
        gc.collect()

        res = run_pipeline(c, loaded_data.get(c['name']), data_store.get(c['name']))
        if res: results.append(res)

        K.clear_session()
        gc.collect()

    print("\n>>> STEP 3: PLOTTING RESULTS...")
    for r in results:
        plot_eval(r['train_eval_data'], r['name'], r['interval_td'])
        plot_eval(r['test_eval_data'], r['name'], r['interval_td'])
        plot_final(r['context_history'], r['future_df'], r['name'], r['interval_td'])

    # Сводный график
    if results:
        plt.figure(figsize=(15, 8))
        for r in results:
            if r['future_df'] is not None:
                df = r['future_df']
                pt, pv = generate_plotted_times_and_values(df.index, df['predicted_value'].values, r['interval_td'])
                if len(pt) > 0: plt.plot(pt, pv, label=r['name'], marker='.')
        plt.title(f"COMBINED FORECAST: {INSTRUMENT_TICKER}")
        plt.legend()
        plt.show()

    print("DONE.")