import numpy as np
import pandas as pd
import os
import joblib
import warnings
import math
from datetime import datetime, timedelta
import time


from vnstock import Vnstock

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tensorflow.keras.utils import Sequence 


warnings.filterwarnings('ignore')
np.random.seed(42)

STOCK_LIST_TRAINED = [
    'ACB', 'BID', 'CTG', 'HDB', 'LPB', 'MBB', 'SHB',
    'STB', 'TCB', 'TPB', 'VCB', 'VIB'
]


BASE_ARTIFACT_DIR = r"E:\important\AI\time_series" 
PREPROCESS_DIR = os.path.join(BASE_ARTIFACT_DIR, 'preprocessed_data')
MODEL_DIR = os.path.join(BASE_ARTIFACT_DIR, 'models')
MODEL_FILENAME = 'sgd_regressor_model (3).pkl'

PRIMARY_TIMEFRAME = '5m'
REQUIRED_OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
WINDOW_SIZE = 5
HORIZON = 1
TARGET_COL_NAME = 'close'
USE_ROBUST_SCALER = False


NEW_DATA_DIR = "data"
os.makedirs(NEW_DATA_DIR, exist_ok=True)

# Hàm Lấy Dữ liệu Mới từ vnstock
def fetch_new_data(symbols, start_date_str, end_date_str, interval='5m', output_dir=None):
    """
    Lấy dữ liệu lịch sử mới nhất cho danh sách các mã cổ phiếu.
    """
    print(f"Đang lấy dữ liệu từ {start_date_str} đến {end_date_str} cho {len(symbols)} mã...")
    data_dict = {}
    vnstock_instance = Vnstock()

    # Đảm bảo symbols là list
    if isinstance(symbols, str):
        symbols = [symbols]

    for s in symbols:
        print(f"  Đang lấy dữ liệu cho: {s}")
        try:
            stock_instance = vnstock_instance.stock(symbol=s, source='VCI')
            time.sleep(0.5) # Giảm tốc độ request
            df = stock_instance.quote.history(start=start_date_str, end=end_date_str, interval=interval)

            if df is not None and not df.empty:
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                elif not isinstance(df.index, pd.DatetimeIndex):
                     print(f"Cảnh báo [{s}]: Không tìm thấy cột 'time' hoặc index không phải DatetimeIndex.")
                     continue

                df.sort_index(inplace=True)

                if all(col in df.columns for col in REQUIRED_OHLCV_COLS):
                    df = df[REQUIRED_OHLCV_COLS]
                    for col in REQUIRED_OHLCV_COLS:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    initial_rows = len(df)
                    df.dropna(subset=REQUIRED_OHLCV_COLS, inplace=True)
                    rows_after_dropna = len(df)
                    if initial_rows > rows_after_dropna:
                        print(f"  [{s}] Đã loại bỏ {initial_rows - rows_after_dropna} hàng có NaN trong OHLCV.")

                    if not df.empty:
                        data_dict[s] = df
                        print(f"  [{s}] Lấy dữ liệu thành công. Shape: {df.shape}")
                        if output_dir:
                            output_path = os.path.join(output_dir, f"{s}_newdata_{interval}.csv")
                            df.to_csv(output_path)
                    else:
                        print(f"  [{s}] Dữ liệu trống sau khi làm sạch NaN.")
                else:
                    print(f"  Cảnh báo [{s}]: Dữ liệu trả về thiếu các cột OHLCV cần thiết.")
            else:
                print(f"  [{s}] Không lấy được dữ liệu hoặc dữ liệu trống.")

        except Exception as e:
            print(f"  Lỗi khi lấy dữ liệu cho mã {s}: {e}")
            time.sleep(1)

    print(f"Hoàn tất lấy dữ liệu. Số mã thành công: {len(data_dict)}")
    return data_dict

# Hàm Tạo Đặc trưng Kỹ thuật
def add_technical_indicators(df, symbol_name=""):
    df_with_features = df.copy()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_with_features.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_with_features.columns]
        print(f"CẢNH BÁO [{symbol_name}]: DataFrame thiếu các cột bắt buộc để tính chỉ báo: {missing}. Bỏ qua mã này.")
        return pd.DataFrame() # Trả về DF rỗng

    open_price = df_with_features['open']
    high = df_with_features['high']
    low = df_with_features['low']
    close = df_with_features['close']
    volume = df_with_features['volume'].clip(lower=1e-9)

    # EMA Ngắn hạn
    df_with_features['ema_5'] = close.ewm(span=5, adjust=False).mean()
    df_with_features['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df_with_features['ema_26'] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df_with_features['macd'] = df_with_features['ema_12'] - df_with_features['ema_26']
    df_with_features['macd_signal'] = df_with_features['macd'].ewm(span=9, adjust=False).mean()
    df_with_features['macd_diff'] = df_with_features['macd'] - df_with_features['macd_signal']

    # RSI (EWM)
    rsi_window = 9
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0/rsi_window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0/rsi_window, adjust=False).mean()
    loss_safe = loss.replace(0, 1e-9)
    rs = gain / loss_safe
    df_with_features['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    df_with_features['rsi'] = df_with_features['rsi'].fillna(50)

    # ATR (EWM)
    atr_window = 10
    high_low = high - low
    high_close_prev = abs(high - close.shift())
    low_close_prev = abs(low - close.shift())
    ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
    true_range = ranges.max(axis=1)
    df_with_features['atr'] = true_range.ewm(alpha=1.0/atr_window, adjust=False).mean()

    # Volume EMA
    vol_ema_window = 20
    df_with_features['volume_ema_20'] = volume.ewm(span=vol_ema_window, adjust=False).mean()

    # VWAP (Rolling)
    vwap_window = 20
    tp = (high + low + close) / 3
    vwap_numerator = (tp * volume).rolling(window=vwap_window).sum()
    vwap_denominator = volume.rolling(window=vwap_window).sum().replace(0, 1e-9)
    df_with_features['vwap_roll'] = vwap_numerator / vwap_denominator

    # Volatility (Std Dev of Log Returns)
    vol_window_short = 10
    vol_window_long = 20
    log_return = np.log(close / close.shift()).fillna(0)
    df_with_features[f'volatility_{vol_window_short}'] = log_return.rolling(window=vol_window_short).std() * np.sqrt(vol_window_short)
    df_with_features[f'volatility_{vol_window_long}'] = log_return.rolling(window=vol_window_long).std() * np.sqrt(vol_window_long)

    # Động lực của Chỉ báo
    df_with_features['rsi_roc_1'] = df_with_features['rsi'].diff()
    df_with_features['macd_diff_roc_1'] = df_with_features['macd_diff'].diff()

    # Chuẩn hóa Khoảng cách EMA
    ema_12_safe = df_with_features['ema_12'].replace(0, np.nan)
    df_with_features['ema_5_12_spread_pct'] = (df_with_features['ema_5'] - df_with_features['ema_12']) / ema_12_safe * 100

    # Động lực Biến động
    vol_long_safe = df_with_features[f'volatility_{vol_window_long}'].replace(0, np.nan)
    df_with_features['vol_short_long_ratio'] = df_with_features[f'volatility_{vol_window_short}'] / vol_long_safe
    df_with_features['atr_roc_1'] = df_with_features['atr'].diff()

    # Động lực Khối lượng
    vol_ema_safe = df_with_features['volume_ema_20'].replace(0, np.nan)
    df_with_features['volume_spike_factor'] = volume / vol_ema_safe
    df_with_features['price_change_x_volume'] = close.diff() * volume
    df_with_features['log_return_x_volume'] = log_return * volume

    # Đặc trưng Vi cấu trúc Nến
    df_with_features['candle_body_size'] = abs(close - open_price)
    df_with_features['candle_upper_wick'] = high - np.maximum(open_price, close)
    df_with_features['candle_lower_wick'] = np.minimum(open_price, close) - low
    candle_range = (high - low).replace(0, np.nan)
    df_with_features['candle_body_ratio'] = df_with_features['candle_body_size'] / candle_range

    # Tương tác Giá - VWAP - ATR
    atr_safe = df_with_features['atr'].replace(0, np.nan)
    df_with_features['price_vwap_dist_atr'] = (close - df_with_features['vwap_roll']) / atr_safe

    # Chỉ báo được làm mượt
    df_with_features['rsi_smoothed_3'] = df_with_features['rsi'].ewm(span=3, adjust=False).mean()

    # Xử lý NaN cuối cùng
    df_with_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_len = len(df_with_features)
    df_with_features.dropna(inplace=True) # Quan trọng: dropna sau khi tính toán
    final_len = len(df_with_features)
    rows_dropped = initial_len - final_len
    if rows_dropped > 0:
        print(f"[{symbol_name}] Đã loại bỏ {rows_dropped} hàng đầu do NaN từ tính toán chỉ báo.")

    return df_with_features

# Lớp DataProcessor 
class DataProcessor:
    """Lớp xử lý chuẩn hóa và mã hóa cho dữ liệu."""
    def __init__(self, stock_list_trained):
        self.scaler = None
        self.encoder = None
        self.numeric_columns_ = None
        self.encoded_feature_names_ = None
        self.n_features_in_ = None
        self.fitted_ = False
        self.target_col_index_in_numeric_ = -1
        self.stock_list_trained = stock_list_trained

    def load_fitted_state(self, preprocess_dir):
        """Tải trạng thái đã fit từ các file đã lưu."""
        try:
            self.scaler = joblib.load(os.path.join(preprocess_dir, 'feature_scaler.pkl'))
            self.encoder = joblib.load(os.path.join(preprocess_dir, 'symbol_encoder.pkl'))
            self.numeric_columns_ = joblib.load(os.path.join(preprocess_dir, 'numeric_columns.pkl'))
            self.encoded_feature_names_ = joblib.load(os.path.join(preprocess_dir, 'encoded_feature_names.pkl'))
            self.target_col_index_in_numeric_ = joblib.load(os.path.join(preprocess_dir, 'target_col_index.pkl'))

            if hasattr(self.scaler, 'n_features_in_'):
                 self.n_features_in_ = self.scaler.n_features_in_
            elif hasattr(self.scaler, 'n_samples_seen_'):
                 self.n_features_in_ = len(self.numeric_columns_)
            else:
                 print("Cảnh báo: Không thể xác định n_features_in_ từ scaler đã tải.")
                 self.n_features_in_ = len(self.numeric_columns_)

            self.fitted_ = True
            print("Đã tải thành công Scaler, Encoder và thông tin cột.")

            if TARGET_COL_NAME not in self.numeric_columns_:
                 print(f"LỖI NGHIÊM TRỌNG: Cột target '{TARGET_COL_NAME}' không có trong danh sách cột numeric đã lưu!")
                 self.fitted_ = False
                 return False
            try:
                recalculated_index = self.numeric_columns_.index(TARGET_COL_NAME)
                if recalculated_index != self.target_col_index_in_numeric_:
                    print(f"Cảnh báo: Index cột target đã lưu ({self.target_col_index_in_numeric_}) khác với index tính lại ({recalculated_index}). Sử dụng index tính lại.")
                    self.target_col_index_in_numeric_ = recalculated_index
            except ValueError:
                 print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy cột target '{TARGET_COL_NAME}' trong numeric_columns_ khi kiểm tra lại.")
                 self.fitted_ = False
                 return False

            return True
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file artifact cần thiết: {e}")
            self.fitted_ = False
            return False
        except Exception as e:
            print(f"Lỗi khi tải trạng thái đã fit: {e}")
            self.fitted_ = False
            return False

    def _prepare_combined_data(self, data_dict, symbols_to_process):
        """Kết hợp dữ liệu, reset index và trả về index gốc."""
        all_data = []
        original_row_indices = {}
        current_pos = 0
        valid_symbols_in_dict = [s for s in symbols_to_process if s in data_dict and not data_dict[s].empty]

        if not valid_symbols_in_dict:
             return pd.DataFrame(), {}, [], pd.Index([])

        for symbol in valid_symbols_in_dict:
            df = data_dict[symbol].copy()
            df['symbol_cat_temp'] = symbol # Thêm cột symbol tạm thời
            all_data.append(df)
            start_row = current_pos
            end_row = current_pos + len(df)
            original_row_indices[symbol] = (start_row, end_row)
            current_pos = end_row

        if not all_data:
             return pd.DataFrame(), {}, [], pd.Index([])

        combined_data = pd.concat(all_data, axis=0, ignore_index=False)
        original_dt_index = combined_data.index
        combined_data.reset_index(drop=True, inplace=True)

        return combined_data, original_row_indices, valid_symbols_in_dict, original_dt_index

    def transform(self, data_dict, symbols_to_process):
        """Transform dữ liệu mới dùng scaler và encoder đã fit."""
        if not self.fitted_:
            print("Lỗi: Bộ xử lý chưa được tải hoặc tải lỗi. Không thể transform.")
            return {}

        combined_data, original_row_indices, valid_symbols, original_dt_index = self._prepare_combined_data(data_dict, symbols_to_process)

        if combined_data.empty:
            print("Cảnh báo: Không có dữ liệu hợp lệ để transform.")
            return {}

        current_numeric_cols = combined_data.select_dtypes(include=np.number).columns.tolist()
        cols_to_scale = [col for col in self.numeric_columns_ if col in current_numeric_cols]
        if len(cols_to_scale) != len(self.numeric_columns_):
             missing_in_new_data = [col for col in self.numeric_columns_ if col not in current_numeric_cols]
             print(f"Cảnh báo: Dữ liệu mới thiếu các cột numeric đã dùng để fit scaler: {missing_in_new_data}")
             if TARGET_COL_NAME not in cols_to_scale:
                 print(f"LỖI NGHIÊM TRỌNG: Cột target '{TARGET_COL_NAME}' bị thiếu trong dữ liệu mới sau khi tính chỉ báo!")
                 return {}

        try:
            # Chỉ lấy các cột numeric đã được fit, theo đúng thứ tự
            numeric_data = combined_data[self.numeric_columns_].values
        except KeyError as e:
            print(f"Lỗi KeyError khi truy cập cột numeric: {e}. Có thể do thiếu cột trong dữ liệu mới.")
            print(f"Các cột numeric mong đợi: {self.numeric_columns_}")
            print(f"Các cột có trong combined_data: {combined_data.columns.tolist()}")
            return {}

        scaled_numeric_values = self.scaler.transform(numeric_data)
        scaled_numeric_df = pd.DataFrame(scaled_numeric_values, columns=self.numeric_columns_, index=combined_data.index)

        if 'symbol_cat_temp' not in combined_data.columns:
             print("Lỗi: Thiếu cột 'symbol_cat_temp' để mã hóa.")
             return {}
        symbols_array = combined_data[['symbol_cat_temp']]
        symbol_encoded_values = self.encoder.transform(symbols_array)
        encoded_df = pd.DataFrame(symbol_encoded_values, columns=self.encoded_feature_names_, index=combined_data.index)

        final_df = pd.concat([scaled_numeric_df, encoded_df], axis=1)

        transformed_dict = {}
        for symbol, (start_row, end_row) in original_row_indices.items():
            if start_row < len(final_df) and end_row <= len(final_df):
                symbol_df_processed = final_df.iloc[start_row:end_row].copy()
                if start_row < len(original_dt_index) and end_row <= len(original_dt_index):
                    symbol_dt_index = original_dt_index[start_row:end_row]
                    symbol_df_processed.index = symbol_dt_index
                    transformed_dict[symbol] = symbol_df_processed
                else:
                    print(f"Cảnh báo [{symbol}]: Lỗi index datetime khi tách dữ liệu đã transform.")
            else:
                 print(f"Cảnh báo [{symbol}]: Lỗi index ({start_row}:{end_row}) khi tách dữ liệu đã transform (len={len(final_df)}).")

        return transformed_dict

    def inverse_transform_target(self, scaled_target_values):
        """Chuyển đổi ngược giá trị của cột target về thang đo gốc."""
        if not self.fitted_:
            print("Lỗi: Bộ xử lý chưa được tải hoặc tải lỗi. Không thể inverse_transform.")
            return np.array([])
        if self.target_col_index_in_numeric_ < 0:
             print("Lỗi: Index của cột target không hợp lệ.")
             return np.array([])
        if self.n_features_in_ is None or self.n_features_in_ <= 0:
             print("Lỗi: Số lượng features của scaler không hợp lệ.")
             return np.array([])

        num_scaler_features = self.n_features_in_
        scaled_values_flat = np.array(scaled_target_values).flatten()

        if scaled_values_flat.size == 0:
            return np.array([])

        dummy_array = np.zeros((len(scaled_values_flat), num_scaler_features))

        if self.target_col_index_in_numeric_ >= num_scaler_features:
             print(f"Lỗi: target_col_index_in_numeric_ ({self.target_col_index_in_numeric_}) >= num_scaler_features ({num_scaler_features}).")
             return np.array([])

        dummy_array[:, self.target_col_index_in_numeric_] = scaled_values_flat
        try:
            inversed_array = self.scaler.inverse_transform(dummy_array)
            return inversed_array[:, self.target_col_index_in_numeric_]
        except ValueError as e:
             print(f"Lỗi trong quá trình inverse_transform của scaler: {e}")
             print(f"Shape của dummy_array: {dummy_array.shape}")
             print(f"Số features mong đợi của scaler: {self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else 'Không rõ'}")
             return np.array([])

# Hàm tải mô hình 
# 
def load_model(model_dir, model_filename):
    """Tải mô hình SGDRegressor đã huấn luyện."""
    model_path = os.path.join(model_dir, model_filename)
    try:
        model = joblib.load(model_path)
        print(f"Đã tải mô hình SGDRegressor từ: {model_path}")
        return model
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file mô hình SGD tại '{model_path}'.")
        return None
    except Exception as e:
        print(f"LỖI: Không thể tải mô hình SGD: {e}.")
        return None


# Hàm chuẩn bị dữ liệu cho dự đoán

def prepare_prediction_input(scaled_encoded_df, window_size, feature_names):
    """
    Lấy window_size dòng cuối cùng và làm phẳng thành input cho SGDRegressor.
    Trả về dữ liệu đã làm phẳng và timestamp của điểm dữ liệu cuối cùng trong window.
    """
    if len(scaled_encoded_df) < window_size:
        print(f"Lỗi: Không đủ dữ liệu ({len(scaled_encoded_df)}) để tạo cửa sổ dự đoán (cần {window_size}).")
        return None, None

    # Lấy window_size dòng cuối cùng
    last_window_df = scaled_encoded_df.iloc[-window_size:]

    # Lấy timestamp của điểm cuối cùng trong cửa sổ
    last_timestamp = last_window_df.index[-1]

    # Đảm bảo các cột theo đúng thứ tự feature_names
    try:
        last_window_values = last_window_df[feature_names].values
    except KeyError as e:
        print(f"Lỗi KeyError khi lấy dữ liệu cửa sổ cuối: {e}. Thiếu cột?")
        print(f"Cột mong đợi: {feature_names}")
        print(f"Cột có trong window: {last_window_df.columns.tolist()}")
        return None, None

    # Làm phẳng (flatten) dữ liệu
    # Kích thước: (1, window_size * num_features)
    flattened_input = last_window_values.reshape(1, -1)

    return flattened_input, last_timestamp


# Hàm tính toán timestamp dự đoán

def calculate_prediction_timestamp(last_known_timestamp, horizon, timeframe='5m'):
    """
    Tính toán timestamp cho giá trị dự đoán dựa trên timestamp cuối cùng đã biết,
    horizon và timeframe.
    """
    try:
        if timeframe.endswith('m'):
            minutes_delta = int(timeframe[:-1]) * horizon
            prediction_ts = last_known_timestamp + timedelta(minutes=minutes_delta)
            return prediction_ts
        else:
            print(f"Cảnh báo: Timeframe '{timeframe}' chưa được hỗ trợ đầy đủ để tính timestamp dự đoán. Trả về None.")
            return None
    except Exception as e:
        print(f"Lỗi khi tính toán timestamp dự đoán: {e}")
        return None