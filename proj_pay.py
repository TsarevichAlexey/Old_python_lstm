# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os
import joblib
import yfinance as yf
import traceback
import random

# ML & DL
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import load_model as keras_load_model

# TA Lib & Sentiment
try:
    import pandas_ta as ta
except ImportError:
    print("Библиотека pandas_ta не найдена. Пожалуйста, установите ее: pip install pandas_ta")

try:
    from textblob import TextBlob
except ImportError:
    print("Библиотека textblob не найдена. Sentiment analysis будет отключен (0.0). pip install textblob")
    TextBlob = None

from IPython.display import display
import warnings

warnings.filterwarnings("ignore")

# %%
# --- ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ---

INSTRUMENT_TICKER = "PYPL"
FEATURE_COLUMN = 'close'
Z_SCORE_THRESHOLD = 4.0
N_CV_SPLITS = 4  # Фолды Walk-Forward
NUM_CANDIDATES = 3  # Сколько моделей обучать для каждого таймфрейма (выбор лучшей)
MAX_CV_WORKERS = min(N_CV_SPLITS, (os.cpu_count() or 1) * 2)
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
BBANDS_PERIOD = 20
BBANDS_STD_DEV = 2
RSI_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH_K = 3

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
# --- БЛОК 0: Логирование и Утилиты (Rich Output) ---

def log_step(model_name, message):
    with print_lock:
        print(f"[{model_name.strip('_')}] {message}")


def log_detailed(model_name, stage_name, data=None, details="", columns_added=None):
    """Выводит детальную информацию и таблицы (для Jupyter)."""
    prefix = f"[{model_name.strip('_')}][{stage_name}]"
    with print_lock:
        print(f"\n{'=' * 10} {prefix} ЗАПУСК {'=' * 10}")
        print(f">> ОПИСАНИЕ: {details}")

        if columns_added:
            print(f">> ИЗМЕНЕНИЯ: Добавлены/Изменены колонки: {columns_added}")

        if data is not None:
            if isinstance(data, pd.DataFrame):
                print(f">> ТАБЛИЦА ДАННЫХ: Размеры={data.shape}")
                if not data.empty:
                    print(f"   Диапазон: {data.index.min()} ... {data.index.max()}")
                    # Красивый вывод таблицы
                    print(f"   ПРИМЕР ДАННЫХ (Head & Tail):")
                    display(pd.concat([data.head(3), data.tail(3)]))
                    print(f"   СТАТИСТИКА ({FEATURE_COLUMN}):")
                    if FEATURE_COLUMN in data.columns:
                        display(data[[FEATURE_COLUMN]].describe().T)
            elif isinstance(data, np.ndarray):
                print(f">> МАССИВ: Shape={data.shape}")

        print(f"{'=' * 10} {prefix} ЗАВЕРШЕНО {'=' * 10}\n")


def get_timedelta_from_candle_interval(interval_str: str) -> timedelta:
    mapping = {
        '1m': timedelta(minutes=1), '2m': timedelta(minutes=2), '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15), '30m': timedelta(minutes=30), '60m': timedelta(minutes=60),
        '1h': timedelta(hours=1), '90m': timedelta(minutes=90),
        '1d': timedelta(days=1), '5d': timedelta(days=5), '1wk': timedelta(weeks=1), '1mo': timedelta(days=30)
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


def save_predictions_to_csv(df, model_name, ticker, target, interval, output_dir=OUTPUT_CSV_DIRECTORY):
    if df is None or df.empty: return
    try:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pred_{ticker}_{model_name}_{ts}.csv"
        filepath = os.path.join(output_dir, filename)

        df_save = df.copy().reset_index()
        df_save.rename(columns={'datetime': 'future_timestamp', 'predicted_value': f'predicted_price'}, inplace=True)
        df_save['model'] = model_name
        df_save.to_csv(filepath, index=False)
        log_step(model_name, f"Saved CSV: {filepath}")
    except Exception as e:
        log_step(model_name, f"CSV Error: {e}")


# --- БЛОК 1: Оркестрация ---

def check_bundle_exists(model_name, base_dir=MODEL_SAVE_DIR):
    path = os.path.join(base_dir, model_name)
    if not os.path.isdir(path): return False
    files = ["model.keras", "scaler_X.joblib", "scaler_y.joblib", "metadata.joblib", "df_history.pkl"]
    return all(os.path.exists(os.path.join(path, f)) for f in files)


def check_all_bundles_exist(model_configs_list, base_dir=MODEL_SAVE_DIR):
    if not model_configs_list: return False
    for config in model_configs_list:
        if not check_bundle_exists(config["name"], base_dir): return False
    return True


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
        log_step(name, "Bundle Saved Successfully.")
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
        log_detailed(name, "BUNDLE LOADED", df_h, "История загружена из файла.")
        return {"model": model, "scaler_X": sX, "scaler_y": sY, "metadata": meta, "df_history": df_h}
    except Exception as e:
        return None


# --- БЛОК 2: Данные и TA (Stationary) ---

def get_news_sentiment(ticker):
    if TextBlob is None: return 0.0
    try:
        tick = yf.Ticker(ticker)
        news = tick.news
        if not news: return 0.0
        scores = [TextBlob(f"{n.get('title', '')} {n.get('summary', '')}").sentiment.polarity for n in news]
        return sum(scores) / len(scores) if scores else 0.0
    except:
        return 0.0


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
        log_detailed(model_name, "ЗАГРУЗКА ДАННЫХ", df, f"Скачаны котировки Yahoo Finance за {days} дн.",
                     columns_added="open, high, low, close, volume")
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
        added = ['log_ret']
        for p in SMA_PERIODS:
            sma = ta.sma(df_ta['close'], length=p)
            df_ta[f'dist_sma_{p}'] = (df_ta['close'] - sma) / sma
            added.append(f'dist_sma_{p}')

        df_ta['rsi'] = ta.rsi(df_ta['close'], length=RSI_PERIOD)
        added.append('rsi')

        macd = ta.macd(df_ta['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if macd is not None:
            df_ta['macd_norm'] = macd.iloc[:, 0] / df_ta['close']
            df_ta['macd_sig_norm'] = macd.iloc[:, 1] / df_ta['close']
            df_ta['macd_hist_norm'] = macd.iloc[:, 2] / df_ta['close']
            added.extend(['macd_norm', 'macd_sig_norm', 'macd_hist_norm'])

        stoch = ta.stoch(df_ta['high'], df_ta['low'], df_ta['close'], k=STOCH_K, d=STOCH_D)
        if stoch is not None:
            df_ta = pd.concat([df_ta, stoch], axis=1)
            added.extend(stoch.columns.tolist())

        atr = ta.atr(df_ta['high'], df_ta['low'], df_ta['close'], length=ATR_PERIOD)
        if atr is not None:
            df_ta['atr_pct'] = atr / df_ta['close']
            added.append('atr_pct')

        df_ta['vol_chg'] = df_ta['volume'].pct_change().replace([np.inf, -np.inf], 0)
        added.append('vol_chg')

        rsi_fill = df_ta['rsi'].fillna(50)
        tech_sent = (rsi_fill - 50) / 50.0
        df_ta['sentiment'] = tech_sent
        added.append('sentiment')

        if not is_future and not df_ta.empty:
            real = get_news_sentiment(INSTRUMENT_TICKER)
            if real != 0.0:
                df_ta.iloc[-1, df_ta.columns.get_loc('sentiment')] = (tech_sent.iloc[-1] * 0.7) + (real * 0.3)

        df_ta = df_ta.loc[:, ~df_ta.columns.duplicated()]

        return df_ta
    except:
        return df


def remove_outliers_zscore(df, col, thresh, model_name):
    if col not in df.columns: return df
    try:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0: return df
        z = (df[col] - mean) / std
        mask = np.abs(z) <= thresh
        return df[mask]
    except:
        return df


# --- БЛОК 3: Preprocessing & Sequencing ---

def prepare_dataframe(ticker, days, interval, col, thresh, model_name, tf_td):
    log_step(model_name, "Загрузка и предобработка данных...")
    df = get_market_data(ticker, days, interval, model_name)
    if df.empty: return None, None

    df_raw = df.copy()
    df_clean = remove_outliers_zscore(df, col, thresh, model_name)

    log_detailed(model_name, "ОЧИСТКА ВЫБРОСОВ", df_clean,
                 f"Удалено {len(df_raw) - len(df_clean)} строк (Z-score > {thresh})", columns_added="Фильтрация строк")

    df_final = add_technical_indicators_stationary(df_clean, model_name)
    df_final = df_final.dropna()

    log_detailed(model_name, "ГЕНЕРАЦИЯ ПРИЗНАКОВ", df_final, "Добавлены TA индикаторы",
                 columns_added="log_ret, rsi, macd_*, stoch_*, atr_pct, sentiment")
    return df_final, (df_raw, df_clean)


def get_stationary_feature_names(df_columns):
    exclude = ['open', 'high', 'low', 'close', 'volume', 'adj close', 'log_ret']
    return [c for c in df_columns if c not in exclude]


def create_sequences(X_data, y_data, prices_for_reconstruction, n_in, n_out, timestamps):
    X, y, t, base_prices = [], [], [], []
    if len(X_data) < n_in + n_out: return np.array(X), np.array(y), np.array(t), np.array(base_prices)
    for i in range(len(X_data) - n_in - n_out + 1):
        X.append(X_data[i: i + n_in])
        y.append(y_data[i + n_in: i + n_in + n_out])
        t.append(timestamps[i + n_in])
        base_prices.append(prices_for_reconstruction[i + n_in - 1])
    return np.array(X), np.array(y), np.array(t), np.array(base_prices)


# --- БЛОК 4: Модель и Обучение (Tournament) ---

def build_lstm(n_in, n_out, n_feat):
    m = Sequential([
        LSTM(128, return_sequences=True, input_shape=(n_in, n_feat)),
        BatchNormalization(), Dropout(0.2),
        LSTM(128, return_sequences=True),
        BatchNormalization(), Dropout(0.2),
        LSTM(64, return_sequences=False),
        BatchNormalization(), Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(n_out)
    ])
    m.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae', 'mse'])
    return m


def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0: return {}
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mean_y = np.mean(y_true)
    mae_pct = (mae / abs(mean_y)) * 100 if mean_y != 0 else 0
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mae_pct': mae_pct}


def run_walk_forward_cv(X, y, n_in, n_out, n_feat, name, candidate_id):
    """Обучает одну модель-кандидата."""
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    best_model = None
    best_r2 = -float('inf')
    best_history = None

    # Store metrics for reporting
    metrics_report = {}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_t, X_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]

        model = build_lstm(n_in, n_out, n_feat)
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

        history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=40, batch_size=32, verbose=0,
                            callbacks=[es, rlr])

        preds = model.predict(X_v, verbose=0)
        r2 = r2_score(y_v[:, 0], preds[:, 0])

        # Calculate other metrics for the fold
        mse = mean_squared_error(y_v[:, 0], preds[:, 0])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_v[:, 0], preds[:, 0])

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_history = history.history
            metrics_report = {'id': candidate_id, 'r2': r2, 'rmse': rmse, 'mae': mae, 'mse': mse}

    return best_model, best_r2, metrics_report, best_history


def prepare_evaluation_data(model, X, y, base_prices, timestamps, scaler_y, tag="Test"):
    try:
        preds_scaled = model.predict(X, verbose=0)
        preds_flat = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        preds_ret = preds_flat.reshape(preds_scaled.shape)

        y_flat = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
        y_ret = y_flat.reshape(y.shape)

        pred_prices = base_prices * np.exp(preds_ret[:, 0])
        actual_prices = base_prices * np.exp(y_ret[:, 0])

        metrics = calculate_metrics(actual_prices, pred_prices)
        return {"type": tag, "timestamps": timestamps, "actual": actual_prices, "predicted": pred_prices,
                "metrics": metrics}
    except:
        return None


# --- БЛОК 5: Прогноз Будущего ---

def predict_future_recursive(model, sX, sY, df_hist, stationary_feats, n_fut, n_in, n_out, tf_td, name):
    log_step(name, f"Generating recursive forecast ({n_fut} steps)...")
    hist = df_hist.copy().loc[:, ~df_hist.columns.duplicated()]

    future_prices = []
    future_dates = []
    last_dt = hist.index[-1]

    recent_returns = np.diff(np.log(hist['close'].tail(51).values))
    base_volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.001

    steps_done = 0
    while steps_done < n_fut:
        input_df = hist[stationary_feats].iloc[-n_in:].fillna(method='ffill').fillna(method='bfill')

        if len(input_df) < n_in:
            pad = pd.DataFrame([input_df.iloc[0]] * (n_in - len(input_df)), columns=input_df.columns)
            input_df = pd.concat([pad, input_df], ignore_index=True)

        X_val = sX.transform(input_df.values).reshape(1, n_in, -1)

        pred_scaled = model.predict(X_val, verbose=0)
        pred_ret = sY.inverse_transform(pred_scaled).flatten()

        noise = np.random.normal(0, base_volatility, size=pred_ret.shape) * 0.8
        pred_ret_noisy = pred_ret + noise

        last_known_price = hist['close'].iloc[-1]

        cum_ret = np.cumsum(pred_ret_noisy)
        chunk_prices = last_known_price * np.exp(cum_ret)

        chunk_len = min(len(chunk_prices), n_fut - steps_done)
        chunk_prices = chunk_prices[:chunk_len]

        new_rows = []
        for p in chunk_prices:
            last_dt += tf_td
            row = {c: np.nan for c in hist.columns}
            row['open'] = p;
            row['high'] = p;
            row['low'] = p;
            row['close'] = p
            if 'volume' in hist.columns:
                row['volume'] = hist['volume'].iloc[-1]
            new_rows.append(pd.DataFrame([row], index=[last_dt]))

        temp_df = pd.concat(new_rows)
        temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]

        hist = pd.concat([hist, temp_df])
        hist = hist.loc[:, ~hist.columns.duplicated()]

        tail_len = n_in + 100
        tail = hist.iloc[-tail_len:].copy()
        tail_ta = add_technical_indicators_stationary(tail, name, is_future=True)

        common_idx = tail_ta.index
        for col in stationary_feats:
            if col in hist.columns and col in tail_ta.columns:
                hist.loc[common_idx, col] = tail_ta.loc[common_idx, col]

        future_prices.extend(chunk_prices)
        future_dates.extend(temp_df.index)
        steps_done += chunk_len

        new_recent_ret = np.diff(np.log(hist['close'].tail(30).values))
        if len(new_recent_ret) > 5:
            base_volatility = np.std(new_recent_ret)

    return pd.DataFrame({'datetime': future_dates, 'predicted_value': future_prices}).set_index('datetime')


# --- БЛОК 6: MAIN PIPELINE ---

def run_pipeline(cfg, loaded_bundle=None, preloaded_data=None):
    name = cfg['name']
    interval = cfg['candle_timeframe']
    days = cfg['days_history']
    n_in = cfg['n_steps_in']
    n_out = cfg['n_steps_out']
    n_fut = cfg['future_predictions']
    tf_td = get_timedelta_from_candle_interval(interval)

    result_pkg = {
        'name': name, 'interval_td': tf_td,
        'outliers_data': None, 'train_eval_data': None,
        'test_eval_data': None, 'future_df': None, 'context_history': None,
        'training_history': None
    }

    try:
        if loaded_bundle:
            log_step(name, "Режим ЗАГРУЗКИ. Обновление данных...")
            model = loaded_bundle['model']
            sX = loaded_bundle['scaler_X']
            sY = loaded_bundle['scaler_y']
            feats = loaded_bundle['metadata']['features']
            df_h = loaded_bundle['df_history']

            # (Логика обновления данных опускается для краткости, она такая же как была)
            last_ts = df_h.index[-1]
            now = datetime.now().replace(tzinfo=None)
            delta = (now - last_ts).days + 2
            if delta > 0:
                df_new = get_market_data(INSTRUMENT_TICKER, delta, interval, f"{name}-Upd")
                if not df_new.empty:
                    df_new = df_new[df_new.index > last_ts]
                    if not df_new.empty:
                        common = ['open', 'high', 'low', 'close', 'volume']
                        df_combined = pd.concat([df_h[common], df_new[common]]).sort_index()
                        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
                        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                        df_h = add_technical_indicators_stationary(df_combined, name)
                        df_h = df_h.dropna()
                        log_step(name, f"Данные обновлены. Новый конец: {df_h.index[-1]}")

        else:
            log_step(name, "Режим ОБУЧЕНИЯ (Tournament)...")

            # Используем уже загруженные данные, если есть
            if preloaded_data:
                df_full, outliers_tuple = preloaded_data
            else:
                # На всякий случай fallback
                df_full, outliers_tuple = prepare_dataframe(INSTRUMENT_TICKER, days, interval, FEATURE_COLUMN,
                                                            Z_SCORE_THRESHOLD, name, tf_td)

            if df_full is None: return None

            result_pkg['outliers_data'] = outliers_tuple
            stationary_feats = get_stationary_feature_names(df_full.columns)

            split_idx = int(len(df_full) * (1 - TEST_SPLIT_SIZE))
            df_train = df_full.iloc[:split_idx]
            df_test = df_full.iloc[split_idx - n_in:]

            sX = StandardScaler()
            sY = StandardScaler()
            sX.fit(df_train[stationary_feats].values)
            sY.fit(df_train[['log_ret']].values)

            X_tr, y_tr, t_tr, base_tr = create_sequences(
                sX.transform(df_train[stationary_feats].values),
                sY.transform(df_train[['log_ret']].values).flatten(),
                df_train['close'].values, n_in, n_out, df_train.index
            )

            X_te, y_te, t_te, base_te = create_sequences(
                sX.transform(df_test[stationary_feats].values),
                sY.transform(df_test[['log_ret']].values).flatten(),
                df_test['close'].values, n_in, n_out, df_test.index
            )

            # --- TOURNAMENT ---
            best_cand_model = None
            best_cand_score = -float('inf')
            best_hist = None
            cand_stats_list = []

            log_step(name, f"Training {NUM_CANDIDATES} candidates...")

            for i in range(NUM_CANDIDATES):
                c_model, c_score, c_stats, c_hist = run_walk_forward_cv(X_tr, y_tr, n_in, n_out, len(stationary_feats),
                                                                        name, i)
                cand_stats_list.append(c_stats)

                if c_score > best_cand_score:
                    best_cand_score = c_score
                    best_cand_model = c_model
                    best_hist = c_hist

            # Текстовый вывод результатов
            with print_lock:
                print(f"\n[{name}] >>> РЕЗУЛЬТАТЫ СРАВНЕНИЯ (Выбор лучшей) <<<")
                c_df = pd.DataFrame(cand_stats_list)
                print(c_df[['id', 'r2', 'rmse', 'mae', 'mse']].to_string(index=False))
                print(
                    f"[{name}] ПОБЕДИТЕЛЬ: Model ID {c_df.loc[c_df['r2'].idxmax()]['id']} (R2={best_cand_score:.4f})\n")

            model = best_cand_model
            result_pkg['training_history'] = best_hist

            if model:
                result_pkg['train_eval_data'] = prepare_evaluation_data(model, X_tr, y_tr, base_tr, t_tr, sY, "Train")
                result_pkg['test_eval_data'] = prepare_evaluation_data(model, X_te, y_te, base_te, t_te, sY, "Test")

                sX_full = StandardScaler().fit(df_full[stationary_feats].values)
                sY_full = StandardScaler().fit(df_full[['log_ret']].values)
                save_model_bundle(model, sX_full, sY_full, stationary_feats, df_full, tf_td, n_in, n_out, name)

                sX, sY = sX_full, sY_full
                feats = stationary_feats
                df_h = df_full

        if model and df_h is not None:
            result_pkg['context_history'] = df_h.iloc[-100:].copy()
            fut_df = predict_future_recursive(model, sX, sY, df_h, feats, n_fut, n_in, n_out, tf_td, name)
            save_predictions_to_csv(fut_df, name, INSTRUMENT_TICKER, FEATURE_COLUMN, interval)
            result_pkg['future_df'] = fut_df

            return result_pkg

    except Exception as e:
        log_step(name, f"PIPELINE CRASH: {e}")
        traceback.print_exc()
        return None


# --- PLOTTING (Main Thread) ---

def plot_outliers_safe(data, name, tf_td):
    """
    Рисует график изначальных данных (с подсветкой выбросов) и после очистки.
    """
    raw, clean = data
    if raw.empty: return
    try:
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle(f"[{name}] АНАЛИЗ ДАННЫХ: ДО И ПОСЛЕ ОБРАБОТКИ ВЫБРОСОВ", fontsize=16)

        # Subplot 1: Raw
        ax1 = plt.subplot(2, 1, 1)
        pt, pv = generate_plotted_times_and_values(raw.index, raw[FEATURE_COLUMN], tf_td)
        if len(pt) > 0:
            ax1.plot(pt, pv, label='Исходные данные', color='#1f77b4', linewidth=1)
            mean, std = raw[FEATURE_COLUMN].mean(), raw[FEATURE_COLUMN].std()
            mask = np.abs((raw[FEATURE_COLUMN] - mean) / std) > Z_SCORE_THRESHOLD
            out_idx = raw[mask].index
            comm = raw.index.intersection(out_idx)
            idxs = [raw.index.get_loc(i) for i in comm if raw.index.get_loc(i) < len(pt)]
            if idxs:
                ax1.scatter(pt[idxs], pv[idxs], color='red', s=40, label=f'Выбросы (Z > {Z_SCORE_THRESHOLD})', zorder=5)

        ax1.set_title("1. Изначальные данные (с выбросами)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

        # Subplot 2: Clean
        ax2 = plt.subplot(2, 1, 2)
        ptc, pvc = generate_plotted_times_and_values(clean.index, clean[FEATURE_COLUMN], tf_td)
        if len(ptc) > 0:
            ax2.plot(ptc, pvc, label='Очищенные данные', color='green', linewidth=1)

        ax2.set_title("2. Данные после обработки (удаления выбросов)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plot Error: {e}")


def plot_learning_curves(history, name):
    """Строит графики потерь (Loss) для лучшей модели."""
    if not history: return
    try:
        plt.figure(figsize=(12, 5))
        if 'loss' in history:
            plt.plot(history['loss'], label='Training Loss', color='blue')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss', color='orange', linestyle='--')

        plt.title(f"[{name}] КРИВЫЕ ОБУЧЕНИЯ (Learning Curves)")
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"Curve Error: {e}")


def plot_evaluation_results(data, name, tf_td):
    if not data: return
    try:
        ts = data['timestamps']
        act = data['actual']
        pred = data['predicted']
        met = data['metrics']
        dtype = data['type']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f"[{name}] ОЦЕНКА МОДЕЛИ ({dtype})", fontsize=16)

        m_str = f"R2: {met.get('r2', 0):.4f} | RMSE: {met.get('rmse', 0):.4f} | MAE: {met.get('mae', 0):.4f} | MSE: {met.get('mse', 0):.4f}"

        pt_a, pv_a = generate_plotted_times_and_values(ts, act, tf_td)
        pt_p, pv_p = generate_plotted_times_and_values(ts, pred, tf_td)
        c_a = '#1f77b4' if dtype == "Test" else 'purple'
        c_p = '#ff7f0e' if dtype == "Test" else 'orange'

        if len(pt_a) > 0:
            ax1.plot(pt_a, pv_a, label=f'Факт ({dtype})', color=c_a, alpha=0.8)
            ax1.plot(pt_p, pv_p, label=f'Прогноз ({dtype})', color=c_p, linestyle='--', linewidth=1.5)

        ax1.set_title(f"Общий график | {m_str}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.text(0.02, 0.95, m_str.replace(" | ", "\n"), transform=ax1.transAxes,
                 va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Zoom (last 60 pts)
        zoom = 60
        if len(pt_a) > zoom:
            ax2.plot(pt_a[-zoom:], pv_a[-zoom:], label='Факт (Zoom)', color=c_a, marker='.')
            ax2.plot(pt_p[-zoom:], pv_p[-zoom:], label='Прогноз (Zoom)', color=c_p, linestyle='--', marker='x')
        else:
            ax2.plot(pt_a, pv_a, label='Факт', color=c_a)
            ax2.plot(pt_p, pv_p, label='Прогноз', color=c_p)

        ax2.set_title(f"Детальный вид (Последние {zoom} точек)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d/%m'))

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Eval Plot Error: {e}")


def plot_stitched_forecast(history_df, future_df, name, tf_td):
    if history_df is None or future_df is None: return
    try:
        plt.figure(figsize=(15, 7))
        hist_vals = history_df['close'].values
        pt_h, pv_h = generate_plotted_times_and_values(history_df.index, hist_vals, tf_td)

        combined_vals = np.concatenate([hist_vals, future_df['predicted_value'].values])
        combined_ts = list(history_df.index) + list(future_df.index)
        pt_c, pv_c = generate_plotted_times_and_values(combined_ts, combined_vals, tf_td)

        split_idx = len(pt_h)
        plt.plot(pt_c[:split_idx], pv_c[:split_idx], label='История', color='gray')
        plt.plot(pt_c[split_idx - 1:], pv_c[split_idx - 1:], label='Прогноз', color='#00ff00', linestyle='--',
                 linewidth=2)

        plt.title(f"[{name}] ФИНАЛЬНЫЙ ПРОГНОЗ")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d/%m'))
        plt.show()
    except:
        pass


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
        if input("Загрузить существующие модели? (да/нет): ").lower() == 'да':
            for c in configs:
                b = load_model_bundle(c['name'])
                if b: loaded_data[c['name']] = b

    # --- ШАГ 1: ПРЕДВАРИТЕЛЬНАЯ ЗАГРУЗКА И ГРАФИКИ ВЫБРОСОВ (ПОСЛЕДОВАТЕЛЬНО) ---
    print("\n>>> ЭТАП 1: ЗАГРУЗКА ДАННЫХ И АНАЛИЗ ВЫБРОСОВ...")
    prepared_data_store = {}

    for c in configs:
        name = c['name']
        # Если модель уже обучена и загружена, пропускаем этап анализа сырых данных
        if name in loaded_data:
            continue

        print(f"\n--- Обработка данных для {name} ---")
        tf_td = get_timedelta_from_candle_interval(c['candle_timeframe'])
        df_full, outliers_tuple = prepare_dataframe(INSTRUMENT_TICKER, c['days_history'], c['candle_timeframe'],
                                                    FEATURE_COLUMN, Z_SCORE_THRESHOLD, name, tf_td)

        if df_full is not None:
            # Сразу строим график выбросов, чтобы пользователь его увидел ДО обучения
            plot_outliers_safe(outliers_tuple, name, tf_td)
            prepared_data_store[name] = (df_full, outliers_tuple)
        else:
            print(f"Не удалось загрузить данные для {name}")

    # --- ШАГ 2: ОБУЧЕНИЕ В ПОТОКАХ ---
    print("\n>>> ЭТАП 2: ЗАПУСК ОБУЧЕНИЯ МОДЕЛЕЙ (ПОТОКИ)...")
    final_results = []

    with ThreadPoolExecutor(max_workers=min(len(configs), 3)) as ex:
        # Передаем preloaded_data в run_pipeline
        futs = {ex.submit(run_pipeline, c, loaded_data.get(c['name']), prepared_data_store.get(c['name'])): c for c in
                configs}
        for f in as_completed(futs):
            r = f.result()
            if r: final_results.append(r)

    # --- ШАГ 3: ИТОГОВЫЕ ГРАФИКИ ---
    print("\n>>> ЭТАП 3: ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ...")

    for res in final_results:
        name = res['name']
        tf = res['interval_td']

        # График кривых обучения (Loss)
        if res.get('training_history'):
            plot_learning_curves(res['training_history'], name)

        # Графики оценки точности
        if res['train_eval_data']:
            plot_evaluation_results(res['train_eval_data'], name, tf)
        if res['test_eval_data']:
            plot_evaluation_results(res['test_eval_data'], name, tf)

        # Финальный прогноз
        if res['future_df'] is not None and res['context_history'] is not None:
            plot_stitched_forecast(res['context_history'], res['future_df'], name, tf)

    # --- СВОДНЫЙ ГРАФИК ---
    if final_results:
        plt.figure(figsize=(18, 9))
        latest_price = 0
        for res in final_results:
            if res['context_history'] is not None:
                p = res['context_history'].iloc[-1]['close']
                if p > 0: latest_price = p

        for res in final_results:
            if res['future_df'] is not None:
                df = res['future_df'].copy()
                if not df.empty and latest_price > 0:
                    start_p = df['predicted_value'].iloc[0]
                    df['predicted_value'] = df['predicted_value'] * (latest_price / start_p)
                    pt, pv = generate_plotted_times_and_values(df.index, df['predicted_value'].values,
                                                               res['interval_td'])
                    if len(pt) > 0:
                        plt.plot(pt, pv, label=f"{res['name']}", marker='.', linestyle='--', alpha=0.7)

        plt.title(f"ИТОГОВЫЙ СВОДНЫЙ ПРОГНОЗ - {INSTRUMENT_TICKER}")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.5)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d/%m'))
        plt.tight_layout()
        plt.show()

    print("\n--- ГОТОВО ---")