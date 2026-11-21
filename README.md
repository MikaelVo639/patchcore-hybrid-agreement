# PatchCore Hybrid Agreement

Mô hình của nhóm kết hợp đặc trưng WideResNet50-2 và DINOv2-S/14 để phát hiện và định vị bất thường công nghiệp. Repo chứa bank seed 0 cho 15 lớp MVTec AD và hai encoder FP16 để suy luận bằng CUDA.

## Kết quả

Trung bình đều 15 lớp và 5 seed trên cùng 1.725 ảnh test MVTec AD, mô hình đạt AU-PRO@5% lỗi nhỏ 78,03%, AU-PRO@30% 95,00% và Image AUROC 99,38%. Bank trung bình 655,49 KiB/lớp, peak VRAM 114,47 MiB và peak RAM 2.041,70 MiB theo giao thức trong báo cáo. Hai encoder đóng gói dùng chung có tổng kích thước khoảng 50 MiB. Bank được đưa vào repo là seed 0, điểm tổng hợp bao gồm các seed 0–4.

Các cấu hình đã được so sánh và lựa chọn trên cùng test MVTec AD. Kết quả mang tính thăm dò. Báo cáo ghi phạm vi đo của từng cấu hình, gồm cả các lượt đo từ feature cache.

[Bảng so sánh đầy đủ](reports/BAO_CAO_TONG_HOP.md) · [CSV tổng hợp](reports/results/all-configurations.csv) · [Điểm từng lớp và seed](reports/results/all-quality.csv) · [Đo tài nguyên](reports/results/all-resources.csv)

## Cài đặt

Dùng Python 3.12 và GPU NVIDIA có CUDA. Môi trường benchmark dùng PyTorch 2.7.1+cu128 và RTX 5070 Laptop 8 GB.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Suy luận

Clone repo để lấy các bank và encoder, rồi chạy từ thư mục repo.

```powershell
.venv/Scripts/python.exe predict.py --category screw --image "D:/datasets/mvtec_ad/screw/test/scratch_head/000.png" --output outputs/screw.npy
```

Lệnh xuất điểm bất thường cấp ảnh ra JSON và heatmap float32 224×224 ra file NumPy. Điểm lớn hơn tương ứng mức bất thường cao hơn. Nhãn lỗi cần ngưỡng được hiệu chỉnh từ dữ liệu triển khai. Ảnh đầu vào được resize rồi crop trung tâm theo tỉ lệ 7/8.

Các lớp hỗ trợ gồm bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood và zipper. Dùng bank đúng lớp của ảnh.

## Cấu hình và checkpoint

`config.json` ghi tham số của cấu hình đã huấn luyện. Recipe thực thi tương ứng nằm trong `src/advanced/advanced.py`. Thay đổi recipe cần đánh giá lại và có thể cần dựng lại bank hoặc calibration.

- Map kết hợp fine/mid/deep theo 0,45/0,25/0,30.
- Điểm ảnh kết hợp coarse/mid/deep theo 0,25/0,25/0,50.
- Bank fine dùng đặc trưng WRN layer1+layer2 với UINT4. Bank coarse dùng WRN layer2, UINT4 và calibration normal.
- Bank DINOv2 mid/deep dùng INT4, cosine exact và trộn token block 6/12.
- `models/manifest.json` lưu SHA-256 của từng bank và encoder. CLI xác minh bank và hai encoder trước khi suy luận.

`src/core`, `src/comparison` và `src/advanced` giữ các module phụ thuộc của pipeline thực nghiệm. CLI `predict.py` nạp checkpoint cục bộ. Các đường dẫn dataset và cache mặc định trong những script nghiên cứu cần được thiết lập lại nếu dùng để huấn luyện hoặc tái chạy benchmark.

## Nguồn và điều kiện sử dụng

Mã của nhóm theo MIT. Encoder pretrained và bank tạo từ MVTec AD có nguồn và điều kiện sử dụng ghi trong [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
