# Mô Tả Dự Án: Dự đoán Giá Cổ phiếu Ngân hàng VN30 với SGDRegressor & FastAPI

## 1. Mục tiêu

Xây dựng hệ thống dự đoán giá đóng cửa tiếp theo (khung 5 phút) cho 12 ngân hàng thuộc VN30, sử dụng dữ liệu lịch sử từ **07/2017 đến 03/2025**. Hệ thống cung cấp kết quả dự đoán qua API web (FastAPI) và giao diện trực quan.

## 2. Dữ liệu sử dụng

- **Danh sách cổ phiếu:** 12 ngân hàng trong rổ VN30.
- **Thời gian:** Từ tháng 07/2017 đến tháng 03/2025.
- **Nguồn dữ liệu:** Lịch sử giá cổ phiếu (OHLCV) lấy từ VCI qua thư viện `vnstock`, khung thời gian 5 phút.

## 3. Quy trình xử lý dữ liệu

- **Thu thập dữ liệu:** Tự động tải dữ liệu đủ dài để tính toán các chỉ báo kỹ thuật.
- **Kỹ thuật đặc trưng (Feature Engineering):**
    - *Xu hướng & Momentum:* EMA (5, 12, 26), MACD, RSI (9), ROC của RSI & MACD Histogram.
    - *Biến động:* ATR (10), Rolling Std (10, 20), tỷ lệ biến động ngắn/dài, ROC của ATR.
    - *Khối lượng:* Volume EMA (20), VWAP (20), các đặc trưng tương tác giá/khối lượng.
    - *Cấu trúc nến:* Kích thước thân nến, bóng nến, tỷ lệ thân/toàn bộ nến.
    - *Tương tác chỉ báo:* Khoảng cách phần trăm giữa các đường EMA, giá và VWAP chuẩn hóa theo ATR.
    - *Làm mượt:* RSI làm mượt (EMA 3).
- **Xử lý giá trị thiếu:** Loại bỏ các hàng NaN phát sinh khi tính chỉ báo, đảm bảo đủ dữ liệu cho mỗi cửa sổ (WINDOW_SIZE).
- **Tiền xử lý:**
    - *Scaling:* Chuẩn hóa tất cả đặc trưng số bằng MinMaxScaler (đã huấn luyện và lưu lại).
    - *Encoding:* Mã hóa mã cổ phiếu bằng OneHotEncoder (đã huấn luyện và lưu lại).
- **Cấu trúc dữ liệu đầu vào:** Dữ liệu được chia thành các cửa sổ trượt (WINDOW_SIZE), mỗi cửa sổ gồm các bước thời gian liên tiếp.

## 4. Mô hình dự đoán

- **Lựa chọn mô hình:** Sử dụng SGDRegressor (Scikit-learn), đã khảo sát LSTM, CNN1D.
- **Đầu vào:** Vector đặc trưng đã làm phẳng từ toàn bộ cửa sổ (WINDOW_SIZE × số đặc trưng).
- **Huấn luyện:** Mô hình được huấn luyện trước trên tập dữ liệu lớn (nhiều mã ngân hàng), lưu trạng thái bằng joblib.
- **Dự đoán:** Dự đoán giá đóng cửa đã scale tại thời điểm HORIZON bước 5 phút tiếp theo.
- **Chuyển đổi ngược:** Kết quả dự đoán được inverse_transform về giá gốc.

## 5. Giao diện & Kết quả

- **API RESTful:** Cho phép yêu cầu dự đoán cho từng mã ngân hàng.
- **Web UI:** Hiển thị biểu đồ giá lịch sử (Chart.js), đánh dấu điểm giá dự đoán tiếp theo và thời gian dự đoán.

