
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import các hàm và lớp từ prediction_utils
from prediction_utils import (
    fetch_new_data,
    add_technical_indicators,
    DataProcessor,
    load_model,
    prepare_prediction_input,
    calculate_prediction_timestamp,
    STOCK_LIST_TRAINED,
    PREPROCESS_DIR,
    MODEL_DIR,
    MODEL_FILENAME,
    NEW_DATA_DIR,
    PRIMARY_TIMEFRAME,
    WINDOW_SIZE,
    HORIZON,
    TARGET_COL_NAME
)

# --- Khởi tạo FastAPI và Templates ---
app = FastAPI(title="Stock Prediction API")

# Cấu hình templates (file index.html)
templates = Jinja2Templates(directory="templates")

# Tải các thành phần cốt lõi một lần khi khởi động 
print("--- Khởi tạo ứng dụng FastAPI và tải các thành phần ---")
data_processor = DataProcessor(stock_list_trained=STOCK_LIST_TRAINED)
processor_loaded = data_processor.load_fitted_state(PREPROCESS_DIR)
sgd_model = load_model(MODEL_DIR, MODEL_FILENAME)

if not processor_loaded or sgd_model is None:
    print("LỖI NGHIÊM TRỌNG: Không thể tải Data Processor hoặc Model. API sẽ không hoạt động chính xác.")
    raise RuntimeError("Không thể tải Data Processor hoặc Model. Vui lòng kiểm tra lại đường dẫn và tệp tin.")

# Lấy danh sách features cuối cùng từ processor đã tải
FINAL_FEATURE_NAMES = []
if processor_loaded and data_processor.numeric_columns_ and data_processor.encoded_feature_names_ is not None:
    FINAL_FEATURE_NAMES = data_processor.numeric_columns_ + data_processor.encoded_feature_names_.tolist()
    print(f"Tổng số features mong đợi bởi mô hình: {len(FINAL_FEATURE_NAMES)}")
else:
     print("Cảnh báo: Không thể xác định danh sách features cuối cùng từ DataProcessor.")


# --- Định nghĩa các Pydantic Model cho Response ---
class PredictionPoint(BaseModel):
    symbol: str
    prediction_timestamp: Optional[datetime] = None
    predicted_price: Optional[float] = None
    last_actual_timestamp: Optional[datetime] = None
    last_actual_price: Optional[float] = None
    message: Optional[str] = None

class HistoricalPoint(BaseModel):
    timestamp: datetime
    price: float

class HistoricalData(BaseModel):
    symbol: str
    timestamps: List[datetime]
    prices: List[float]
    last_actual_timestamp: Optional[datetime] = None # Thêm timestamp của giá thực tế cuối
    last_actual_price: Optional[float] = None # Thêm giá thực tế cuối

class PredictionResponse(BaseModel):
    prediction: PredictionPoint
    history: Optional[HistoricalData] = None

# --- Các Endpoints API ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Phục vụ trang HTML chính."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/stocks/", response_model=List[str])
async def get_available_stocks():
    """Trả về danh sách các mã cổ phiếu mà mô hình đã được huấn luyện."""
    return STOCK_LIST_TRAINED

@app.get("/api/predict/{symbol}", response_model=PredictionResponse)
async def predict_stock(symbol: str):
    """Thực hiện dự đoán cho mã cổ phiếu được chỉ định."""
    print(f"\n--- Nhận yêu cầu dự đoán cho mã: {symbol} ---")

    # Kiểm tra xem symbol có hợp lệ không (có trong danh sách đã train)
    if symbol not in STOCK_LIST_TRAINED:
        print(f"Lỗi: Mã {symbol} không nằm trong danh sách đã huấn luyện.")
        raise HTTPException(status_code=400, detail=f"Mã cổ phiếu '{symbol}' không được hỗ trợ (không có trong danh sách huấn luyện).")

    # Kiểm tra xem processor và model đã được tải chưa
    if not processor_loaded or sgd_model is None:
         print("Lỗi: Data Processor hoặc Model chưa sẵn sàng.")
         raise HTTPException(status_code=503, detail="Dịch vụ dự đoán chưa sẵn sàng (lỗi tải mô hình hoặc bộ xử lý).")

    # 1. Lấy dữ liệu mới nhất
    # Xác định khoảng thời gian cần lấy (đủ để tính indicator + window_size)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    try:
        raw_data_dict = fetch_new_data([symbol], start_date_str, end_date_str, PRIMARY_TIMEFRAME, NEW_DATA_DIR)
        if not raw_data_dict or symbol not in raw_data_dict:
            raise HTTPException(status_code=404, detail=f"Không thể lấy dữ liệu gần đây cho mã '{symbol}'.")
        raw_df = raw_data_dict[symbol]
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu cho {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ khi lấy dữ liệu cho '{symbol}'.")

    # Lưu trữ dữ liệu lịch sử gốc để trả về (chỉ lấy giá đóng cửa và timestamp)
    history_df_raw = raw_df[[TARGET_COL_NAME]].copy()
    history_df_raw.dropna(inplace=True) # Đảm bảo không có NaN

    # 2. Thêm chỉ báo kỹ thuật
    print(f"[{symbol}] Đang thêm chỉ báo kỹ thuật...")
    df_with_indicators = add_technical_indicators(raw_df, symbol_name=symbol)
    if df_with_indicators.empty or len(df_with_indicators) < WINDOW_SIZE:
        print(f"[{symbol}] Không đủ dữ liệu sau khi thêm chỉ báo ({len(df_with_indicators)} dòng).")
        raise HTTPException(status_code=400, detail=f"Không đủ dữ liệu cho mã '{symbol}' sau khi tính toán chỉ báo (cần ít nhất {WINDOW_SIZE} điểm).")

    # 3. Chuẩn hóa và Mã hóa dữ liệu
    print(f"[{symbol}] Đang chuẩn hóa và mã hóa...")
    try:
        # Chỉ transform cho mã này
        current_data_dict = {symbol: df_with_indicators}
        scaled_encoded_dict = data_processor.transform(current_data_dict, [symbol])

        if not scaled_encoded_dict or symbol not in scaled_encoded_dict:
             raise ValueError("Kết quả transform rỗng hoặc không chứa symbol.")

        scaled_encoded_df = scaled_encoded_dict[symbol]

        if scaled_encoded_df.empty or len(scaled_encoded_df) < WINDOW_SIZE:
             raise ValueError(f"Không đủ dữ liệu sau transform ({len(scaled_encoded_df)} dòng).")

    except Exception as e:
        print(f"Lỗi trong quá trình transform dữ liệu cho {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý dữ liệu nội bộ cho mã '{symbol}'.")

    # 4. Chuẩn bị Input cho Mô hình (Lấy cửa sổ cuối cùng)
    print(f"[{symbol}] Đang chuẩn bị input dự đoán...")
    prediction_input, last_ts_in_window = prepare_prediction_input(
        scaled_encoded_df,
        WINDOW_SIZE,
        FINAL_FEATURE_NAMES # Sử dụng danh sách features đã lấy từ processor
    )

    if prediction_input is None:
        raise HTTPException(status_code=500, detail=f"Không thể chuẩn bị input dự đoán cho mã '{symbol}'.")

    # 5. Thực hiện Dự đoán
    print(f"[{symbol}] Đang thực hiện dự đoán...")
    try:
        y_pred_scaled = sgd_model.predict(prediction_input) # Dự đoán trên dữ liệu đã scale
    except Exception as e:
        print(f"Lỗi khi mô hình thực hiện predict cho {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình chạy mô hình dự đoán cho '{symbol}'.")

    # 6. Chuyển đổi ngược kết quả
    print(f"[{symbol}] Đang chuyển đổi ngược kết quả...")
    try:
        y_pred_inverse = data_processor.inverse_transform_target(y_pred_scaled)
        if y_pred_inverse.size == 0:
             raise ValueError("Kết quả inverse_transform rỗng.")
        predicted_price = float(y_pred_inverse[0]) # Lấy giá trị dự đoán đầu tiên (và duy nhất)
    except Exception as e:
        print(f"Lỗi khi inverse transform kết quả cho {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý kết quả dự đoán cho '{symbol}'.")

    # 7. Tính toán timestamp dự đoán
    prediction_ts = calculate_prediction_timestamp(last_ts_in_window, HORIZON, PRIMARY_TIMEFRAME)

    # 8. Lấy giá trị thực tế cuối cùng từ dữ liệu gốc
    last_actual_price = None
    last_actual_timestamp = None
    if not history_df_raw.empty:
        last_actual_price = float(history_df_raw[TARGET_COL_NAME].iloc[-1])
        last_actual_timestamp = history_df_raw.index[-1].to_pydatetime() 

    # 9. Chuẩn bị response
    prediction_data = PredictionPoint(
        symbol=symbol,
        prediction_timestamp=prediction_ts,
        predicted_price=predicted_price,
        last_actual_timestamp=last_actual_timestamp,
        last_actual_price=last_actual_price,
        message="Dự đoán thành công."
    )

    # Chuẩn bị dữ liệu lịch sử để trả về cho biểu đồ
    history_limit = 200
    history_df_limited = history_df_raw.iloc[-history_limit:]

    historical_data = HistoricalData(
        symbol=symbol,
        timestamps=[ts.to_pydatetime() for ts in history_df_limited.index],
        prices=history_df_limited[TARGET_COL_NAME].tolist(),
        last_actual_timestamp=last_actual_timestamp,
        last_actual_price=last_actual_price
    )

    print(f"[{symbol}] Dự đoán hoàn tất: Giá={predicted_price:.2f} lúc {prediction_ts}")
    return PredictionResponse(prediction=prediction_data, history=historical_data)


if __name__ == "__main__":
    import uvicorn
    print("http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)