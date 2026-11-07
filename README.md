# Dự báo phân vị và chính sách tồn kho nhạy với lead time cho hàng tươi sống

Mã nguồn của tiểu luận cuối kỳ môn *Các mô hình dự báo trong kinh doanh* (Đại học Kinh tế - Luật, ĐHQG-HCM).

Câu hỏi nghiên cứu: khi lead time dài hơn hoặc dao động, doanh nghiệp nên **rút ngắn lead time** hay **tăng tồn kho an toàn**?
Bài so sánh ba chính sách tồn kho cho 6 nhóm hàng tươi sống (rau củ, bánh mì, gia cầm, sữa, trứng, hải sản) của 12 cửa hàng Favorita:

- P1: safety stock kinh điển (dự báo MA28 + bias + z·sigma)
- P2: newsvendor tĩnh (phân vị cố định theo critical ratio)
- P3: phân vị động (LightGBM quantile, mức phân vị chọn theo chi phí)

Toàn bộ pipeline (nạp dữ liệu, EDA, dự báo, mô phỏng tồn kho FIFO có hạn sử dụng, khoảng tin cậy bootstrap) nằm trong **một file**:
`code/perishable_inventory_stress_test.py`.

## Dữ liệu

Kaggle "Store Sales - Time Series Forecasting": https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Tải về và đặt các file CSV (`train.csv`, `stores.csv`, `holidays_events.csv`, `oil.csv`, `transactions.csv`, ...) vào thư mục `Dataset/`
(cùng cấp với `code/`). Dữ liệu không được đưa lên repo vì `train.csv` nặng khoảng 120 MB.

## Cách chạy

```
pip install -r requirements.txt
python code/perishable_inventory_stress_test.py
```

Hoặc mở file trong PyCharm và chạy (chuột phải > Run). Yêu cầu Python 3.12.

- Lần chạy đầu mất khoảng 6-10 phút (huấn luyện LightGBM walk-forward); kết quả trung gian được lưu ở `outputs/cache`,
  các lần sau chỉ còn vài phút nếu code và cấu hình không đổi. Đặt `USE_CACHE = False` để chạy lại từ đầu.
- Kết quả (bảng CSV, biểu đồ PNG, `run_log.txt`) ghi vào `outputs/` (thư mục này nằm trong `.gitignore`).
- Mọi số ngẫu nhiên dùng `SEED = 42`.

## Các giả định cần lưu ý

Favorita không có dữ liệu lead time, hạn sử dụng, giá hay chi phí, nên các đại lượng sau là **giả định có phân tích độ nhạy**, không phải số đo:

- Lead time: kịch bản trung bình 1-4 ngày, lệch ±1 ngày với xác suất 0 / 15% / 30%.
- Hạn sử dụng: lấy từ "The Food Keeper" (FMI & Cornell) là hạn bảo quản tại nhà, thử thêm hệ số 0,5 / 1 / 1,5.
- Biên lợi nhuận 30% (lưới 20/30/45%), chi phí giữ hàng 25%/năm (lưới 10/25/30%), giá trị thu hồi 0.
- Dữ liệu ở cấp nhóm hàng x cửa hàng (không phải SKU); doanh số quan sát được coi là cầu (không có cờ hết hàng).
