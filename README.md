Mô Tả Dự Án: Hệ thống Dự đoán Giá Cổ phiếu Việt Nam sử dụng SGDRegressor và FastAPI
1. Mục tiêu:
Dự án này xây dựng một hệ thống có khả năng dự đoán giá đóng cửa tiếp theo (khung thời gian 5 phút) cho một danh sách các mã cổ phiếu được lựa chọn trên thị trường chứng khoán Việt Nam. Hệ thống cung cấp kết quả dự đoán thông qua một API web (FastAPI) và giao diện người dùng đơn giản để trực quan hóa.
2. Quy trình Xử lý Dữ liệu
Đây là giai đoạn then chốt, đòi hỏi sự chuẩn bị kỹ lưỡng để cung cấp đầu vào chất lượng cho mô hình:
•	Thu thập Dữ liệu: Dữ liệu lịch sử giá cổ phiếu (OHLCV) theo khung thời gian 5 phút được tự động tải về từ nguồn dữ liệu VCI thông qua thư viện vnstock. Hệ thống chỉ lấy dữ liệu đủ dài trong quá khứ để đảm bảo tính toán được các chỉ báo kỹ thuật phức tạp.
•	Kỹ thuật Đặc trưng (Feature Engineering): Đây là phần được đầu tư nhiều công sức. Thay vì chỉ sử dụng giá OHLCV gốc, một bộ đặc trưng kỹ thuật đa dạng và phức tạp được tính toán từ dữ liệu thô nhằm nắm bắt các khía cạnh khác nhau của động lực thị trường:
o	Xu hướng & Momentum: EMA (5, 12, 26), MACD (MACD line, Signal line, Histogram), RSI (9), ROC (Rate of Change) của RSI và MACD Histogram.
o	Biến động: ATR (10), Độ biến động dựa trên Log Return (Rolling Standard Deviation - 10, 20), Tỷ lệ biến động ngắn/dài, ROC của ATR.
o	Khối lượng: Volume EMA (20), VWAP (Rolling 20), Các đặc trưng tương tác giữa giá/log return và khối lượng (Volume Spike Factor, Price Change x Volume).
o	Cấu trúc Nến: Kích thước thân nến, bóng nến trên/dưới, tỷ lệ thân nến/toàn bộ nến.
o	Tương tác Chỉ báo: Khoảng cách phần trăm giữa các đường EMA, Khoảng cách giữa giá và VWAP chuẩn hóa theo ATR.
o	Làm mượt: RSI làm mượt (EMA 3).
•	Xử lý Giá trị Thiếu (NaN): Các giá trị NaN phát sinh tự nhiên ở các hàng dữ liệu đầu tiên sau khi tính toán các chỉ báo (do yêu cầu dữ liệu quá khứ) sẽ được loại bỏ (dropna). Hệ thống có kiểm tra để đảm bảo vẫn còn đủ dữ liệu (WINDOW_SIZE) sau bước này.
•	Tiền xử lý Chuẩn hóa & Mã hóa:
o	Scaling: Tất cả các đặc trưng số (bao gồm cả giá và các chỉ báo kỹ thuật) được chuẩn hóa sử dụng MinMaxScaler đã được huấn luyện trước đó và lưu lại bằng joblib. Việc này đưa các đặc trưng về cùng một thang đo, rất quan trọng cho các mô hình nhạy cảm với tỷ lệ như SGDRegressor.
o	Encoding: Đặc trưng hạng mục (mã cổ phiếu) được chuyển đổi thành dạng số sử dụng OneHotEncoder đã được huấn luyện trước đó và lưu lại. Điều này cho phép mô hình hiểu được sự khác biệt giữa các mã cổ phiếu mà không áp đặt thứ tự sai lệch.
•	Cấu trúc Dữ liệu Đầu vào: Dữ liệu sau khi xử lý được cấu trúc thành các cửa sổ trượt (sequences) có kích thước cố định (WINDOW_SIZE). Mỗi cửa sổ chứa thông tin từ WINDOW_SIZE bước thời gian 5 phút liên tiếp.
3. Mô hình Dự đoán chính:
	
•	Lựa chọn Mô hình: Khảo sát các mô hình học sâu LSTM, CNN1D và chọn mô hình dự đoán cốt lõi là SGDRegressor từ thư viện Scikit-learn. Đây là một mô hình hồi quy tuyến tính được huấn luyện bằng thuật toán Stochastic Gradient Descent (SGD).
•	Đầu vào Mô hình: SGDRegressor trong dự án này nhận đầu vào là một vector đặc trưng đã được làm phẳng (flattened). Vector này được tạo ra bằng cách nối tất cả các giá trị đặc trưng (đã scale và encode) từ tất cả các bước thời gian trong một cửa sổ (WINDOW_SIZE) lại với nhau. Kích thước đầu vào sẽ là WINDOW_SIZE * số_lượng_đặc_trưng_cuối_cùng.
•	Huấn luyện Mô hình SGDRegressor đã được huấn luyện trước đó trên một tập dữ liệu lịch sử lớn hơn (bao gồm nhiều mã cổ phiếu trong STOCK_LIST_TRAINED) và trạng thái huấn luyện (các hệ số) được lưu lại bằng joblib. API chỉ tải mô hình đã huấn luyện này để thực hiện dự đoán.
•	Dự đoán: Mô hình dự đoán giá trị đã được scale của giá đóng cửa (TARGET_COL_NAME) tại thời điểm HORIZON bước 5 phút trong tương lai, dựa trên cửa sổ dữ liệu đầu vào gần nhất.
•	Chuyển đổi Ngược: Giá trị dự đoán (đang ở thang đo đã scale) được chuyển đổi ngược về thang đo giá gốc bằng phương thức inverse_transform của MinMaxScaler đã lưu.
4. Giao diện và Kết quả:
•	Hệ thống cung cấp một API RESTful cho phép người dùng yêu cầu dự đoán cho một mã cổ phiếu cụ thể.
•	Một trang web đơn giản hiển thị biểu đồ giá lịch sử (sử dụng Chart.js) và đánh dấu điểm giá dự đoán tiếp theo cùng với thời gian dự đoán tương ứng.
