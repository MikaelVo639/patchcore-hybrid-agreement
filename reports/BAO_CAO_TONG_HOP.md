# Benchmark PatchCore trên MVTec AD

## Giao thức và giới hạn

Các lần fit mới lấy 2.903 ảnh build và 726 ảnh calibration từ manifest của nhóm. Mọi kết quả tổng hợp dùng cùng 1.725 ảnh test, mask 224×224 và evaluator PRO. CUDA chạy trên RTX 5070 Laptop GPU. Peak VRAM là PyTorch allocated. Peak RAM là RSS tiến trình. Profile end-to-end mới dùng 20 ảnh mỗi lớp sau warm-up 5 ảnh. Các lượt profile feature cache dùng ảnh đã trích đặc trưng. Độ trễ giữa các đợt có thể khác theo tải máy.

Quét tham số một seed, thử nghiệm nhiều seed và ablation dùng cùng tập test. Xếp hạng mang tính thăm dò. Các biến thể runtime dùng chung số đo chất lượng khi báo cáo trước đã kiểm tra parity. Bank Unified là tổng cho một model dùng chung. Các bank còn lại là trung bình một lớp.

Các dòng **Allocation v1/v2** và **Baseline PCA** mở rộng hai pilot cũ từ các lớp đã khảo sát sang manifest chung. Allocation dùng selector gốc và ResNet18. Baseline PCA dùng bộ trích PatchCore Amazon, covariance PCA-128, coreset và bốn trần dung lượng của pilot. Các lần mở rộng dùng seed 0 cho split và thuật toán, chạy tìm 1-NN chính xác bằng CUDA và đo batch 1 từ feature cache.

## Chênh lệch theo cùng seed 0 so với PatchCore Original

| Cấu hình | PRO@5 lỗi nhỏ | PRO@5 toàn bộ | PRO@30 toàn bộ | Image AUROC | Bank |
|---|---:|---:|---:|---:|---:|
| INT4 Robust · 11×11, top-16 | +4,45 đpt | -2,49 đpt | -4,52 đpt | -2,04 đpt | 0,31% của gốc |
| hybrid-agreement | +15,00 đpt | +9,43 đpt | +2,20 đpt | +0,49 đpt | 1,08% của gốc |
| hybrid-compact-mid | +15,55 đpt | +9,06 đpt | +1,02 đpt | -0,51 đpt | 0,55% của gốc |
| hybrid-wide | +14,62 đpt | +8,72 đpt | +1,70 đpt | +0,52 đpt | 0,82% của gốc |

## Kết quả

Dòng đầu là mã Amazon PatchCore Original với coreset 10%. Sau các đối chứng là các vòng thực nghiệm của nhóm theo thứ tự.

Điểm dùng chung manifest 15 lớp và 1.725 ảnh test. Mỗi lớp có trọng số bằng nhau. Chỉ tổng hợp seed có đủ 15 lớp. AU-PRO lỗi nhỏ tính trên các vùng thuộc tứ phân vị diện tích thấp nhất của mỗi lớp, dùng kết nối 8 hướng và giới hạn FPR 5%. Bank là kích thước tệp mô hình ngoài trọng số encoder. Lite Unified dùng một bank cho cả 15 lớp. Số RAM/VRAM là trung vị các đỉnh trong phạm vi ghi trên dòng. Dấu — biểu thị ô thiếu số đo.

| # | Vòng | Cấu hình | Số seed | AU-PRO@5% lỗi nhỏ | AU-PRO@5% | AU-PRO@30% | Image AUROC % | Bank KiB | Peak RAM MiB | Peak VRAM MiB | p50 ms | Phạm vi đo / trạng thái |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Đối chứng | PatchCore Original · Amazon FP32 10% | 1 | 62,82 | 70,53 | 92,87 | 98,88 | 60.690,67 | 1.146,10 | 484,93 | 592,97 | Batch 1 (15 lớp) |
| 2 | Đối chứng | PatchCore Original · Amazon FP32 1% | 1 | 62,39 | 69,70 | 91,77 | 98,93 | 6.067,73 | 1.090,80 | 484,93 | 94,69 | Batch 1 (15 lớp) |
| 3 | Đối chứng | Anomalib PatchCore 2.7.0 | 1 | 52,97 | 64,48 | 90,93 | 98,55 | 91.037,06 | 1.565,07 | 312,96 | 52,52 | Batch 1 (15 lớp) |
| 4 | Đối chứng | Lite PatchCore tác giả · Unified PCA32 | 1 | 60,79 | 67,63 | 92,29 | 98,52 | 28.606,73 | 1.346,14 | 485,18 | 658,40 | Batch 1 (15 lớp) |
| 5 | Đối chứng | Mahalanobis PatchCore tác giả · Streaming | 1 | 62,44 | 69,14 | 91,98 | 98,74 | 8.083,18 | 950,19 | 0,00 | 407,99 | Batch 1 (15 lớp) |
| 6 | Đối chứng | SoftPatch tác giả · LOF, ảnh build sạch | 1 | 63,31 | 70,19 | 92,81 | 98,06 | 60.750,06 | 1.221,08 | 484,93 | 191,72 | Batch 1 (15 lớp) |
| 7 | Đối chứng | Lightweight PatchCore tác giả · MobileNetV3-Small | 1 | 54,14 | 62,54 | 89,42 | 94,86 | 7.586,58 | 1.138,50 | 20,50 | 265,04 | Batch 1 (15 lớp) |
| 8 | Đối chứng | AnomalyDINO · Full-shot S/14 | 1 | 68,54 | 73,23 | 92,70 | 99,20 | 227.596,66 | 2.594,54 | 411,58 | 285,83 | Batch 1 (15 lớp) |
| 9 | Đối chứng | ASO PatchCore · công thức bài báo | 1 | 48,19 | 60,87 | 88,44 | 96,64 | 2.130,41 | 1.490,05 | 157,53 | 83,36 | Batch 1 (15 lớp) |
| 10 | Đối chứng | PatchCore-Lite · Product Quantization | 1 | 43,23 | 47,63 | 77,05 | 90,68 | 317,86 | 3.253,43 | 1.066,73 | 477,84 | Batch 1 (15 lớp) |
| 11 | Allocation v1 | raw · 512 KiB | 1 | 53,24 | 59,20 | 87,77 | 91,98 | 511,00 | 2.405,69 | 71,58 | 19,74 | Batch 1 từ feature cache (15 lớp) |
| 12 | Allocation v1 | fixed · 512 KiB | 1 | 51,41 | 56,39 | 85,81 | 86,30 | 512,00 | 2.420,33 | 82,29 | 10,91 | Batch 1 từ feature cache (15 lớp) |
| 13 | Allocation v1 | variance · 512 KiB | 1 | 53,56 | 58,26 | 87,05 | 88,40 | 510,15 | 2.405,69 | 76,47 | 16,71 | Batch 1 từ feature cache (15 lớp) |
| 14 | Allocation v1 | joint · 512 KiB | 1 | 49,18 | 53,18 | 82,87 | 80,05 | 389,26 | 2.420,33 | 85,47 | 10,91 | Batch 1 từ feature cache (15 lớp) |
| 15 | Allocation v1 | reconstruction · 512 KiB | 1 | 53,76 | 59,20 | 87,75 | 89,79 | 512,00 | 2.416,86 | 71,59 | 25,73 | Batch 1 từ feature cache (15 lớp) |
| 16 | Allocation v1 | coverage · 512 KiB | 1 | 41,89 | 46,91 | 78,90 | 75,72 | 154,50 | 2.405,69 | 83,58 | 18,34 | Batch 1 từ feature cache (15 lớp) |
| 17 | Allocation v1 | whitened_joint · 512 KiB | 1 | 43,22 | 47,12 | 78,87 | 77,66 | 324,33 | 2.405,34 | 83,28 | 17,63 | Batch 1 từ feature cache (15 lớp) |
| 18 | Allocation v1 | raw · 1024 KiB | 1 | 53,66 | 59,62 | 88,47 | 92,28 | 1.024,00 | 2.406,70 | 73,27 | 9,13 | Batch 1 từ feature cache (15 lớp) |
| 19 | Allocation v1 | fixed · 1024 KiB | 1 | 51,29 | 56,29 | 85,73 | 86,20 | 610,50 | 2.405,34 | 84,17 | 11,67 | Batch 1 từ feature cache (15 lớp) |
| 20 | Allocation v1 | variance · 1024 KiB | 1 | 52,89 | 57,79 | 86,84 | 88,10 | 879,68 | 2.406,69 | 84,04 | 26,23 | Batch 1 từ feature cache (15 lớp) |
| 21 | Allocation v1 | joint · 1024 KiB | 1 | 49,33 | 53,39 | 82,90 | 80,14 | 475,93 | 2.405,34 | 84,17 | 12,40 | Batch 1 từ feature cache (15 lớp) |
| 22 | Allocation v1 | reconstruction · 1024 KiB | 1 | 53,84 | 59,01 | 87,42 | 89,22 | 1.024,00 | 2.381,50 | 80,78 | 9,70 | Batch 1 từ feature cache (15 lớp) |
| 23 | Allocation v1 | coverage · 1024 KiB | 1 | 41,89 | 46,91 | 78,90 | 75,72 | 154,50 | 2.405,69 | 83,58 | 18,34 | Batch 1 từ feature cache (15 lớp) |
| 24 | Allocation v1 | whitened_joint · 1024 KiB | 1 | 42,87 | 46,67 | 78,67 | 77,43 | 377,43 | 2.393,96 | 83,28 | 23,42 | Batch 1 từ feature cache (15 lớp) |
| 25 | Allocation v1 | raw · 4096 KiB | 1 | 53,30 | 59,09 | 87,98 | 91,71 | 3.073,00 | 2.407,25 | 87,03 | 8,85 | Batch 1 từ feature cache (15 lớp) |
| 26 | Allocation v1 | fixed · 4096 KiB | 1 | 51,29 | 56,29 | 85,73 | 86,20 | 610,50 | 2.405,34 | 84,17 | 11,67 | Batch 1 từ feature cache (15 lớp) |
| 27 | Allocation v1 | variance · 4096 KiB | 1 | 52,84 | 57,77 | 86,77 | 88,03 | 1.001,26 | 2.405,20 | 84,04 | 14,21 | Batch 1 từ feature cache (15 lớp) |
| 28 | Allocation v1 | joint · 4096 KiB | 1 | 50,43 | 54,31 | 83,40 | 80,58 | 610,50 | 2.373,13 | 83,67 | 11,94 | Batch 1 từ feature cache (15 lớp) |
| 29 | Allocation v1 | reconstruction · 4096 KiB | 1 | 53,76 | 58,92 | 87,35 | 89,11 | 1.218,50 | 2.381,67 | 84,46 | 11,43 | Batch 1 từ feature cache (15 lớp) |
| 30 | Allocation v1 | coverage · 4096 KiB | 1 | 41,89 | 46,91 | 78,90 | 75,72 | 154,50 | 2.405,69 | 83,58 | 18,34 | Batch 1 từ feature cache (15 lớp) |
| 31 | Allocation v1 | whitened_joint · 4096 KiB | 1 | 42,87 | 46,67 | 78,67 | 77,43 | 377,43 | 2.393,96 | 83,28 | 23,42 | Batch 1 từ feature cache (15 lớp) |
| 32 | Allocation v2 | raw · 256 KiB | 1 | 51,22 | 57,31 | 84,79 | 92,61 | 256,00 | 2.407,81 | 68,57 | 6,50 | Batch 1 từ feature cache (15 lớp) |
| 33 | Allocation v2 | fixed · 256 KiB | 1 | 52,36 | 58,63 | 86,31 | 91,42 | 256,00 | 2.402,45 | 71,32 | 8,92 | Batch 1 từ feature cache (15 lớp) |
| 34 | Allocation v2 | variance · 256 KiB | 1 | 49,31 | 55,48 | 82,26 | 91,56 | 255,85 | 2.410,49 | 67,92 | 9,63 | Batch 1 từ feature cache (15 lớp) |
| 35 | Allocation v2 | joint · 256 KiB | 1 | 48,21 | 54,37 | 84,47 | 87,84 | 256,00 | 2.402,45 | 86,91 | 8,46 | Batch 1 từ feature cache (15 lớp) |
| 36 | Allocation v2 | reconstruction · 256 KiB | 1 | 47,79 | 53,92 | 80,16 | 90,81 | 256,00 | 2.410,49 | 67,93 | 8,63 | Batch 1 từ feature cache (15 lớp) |
| 37 | Allocation v2 | coverage · 256 KiB | 1 | 44,23 | 51,63 | 83,26 | 86,70 | 256,00 | 2.402,45 | 98,26 | 13,12 | Batch 1 từ feature cache (15 lớp) |
| 38 | Allocation v2 | whitened_joint · 256 KiB | 1 | 44,55 | 49,87 | 81,26 | 84,13 | 256,00 | 2.410,49 | 88,85 | 8,37 | Batch 1 từ feature cache (15 lớp) |
| 39 | Allocation v2 | raw · 512 KiB | 1 | 54,54 | 60,08 | 87,75 | 95,27 | 511,00 | 2.402,45 | 70,51 | 7,11 | Batch 1 từ feature cache (15 lớp) |
| 40 | Allocation v2 | fixed · 512 KiB | 1 | 53,85 | 60,46 | 88,57 | 92,92 | 512,00 | 2.410,49 | 80,54 | 9,80 | Batch 1 từ feature cache (15 lớp) |
| 41 | Allocation v2 | variance · 512 KiB | 1 | 53,44 | 60,69 | 88,38 | 93,80 | 511,81 | 2.402,45 | 70,04 | 11,23 | Batch 1 từ feature cache (15 lớp) |
| 42 | Allocation v2 | joint · 512 KiB | 1 | 49,27 | 55,77 | 85,53 | 88,60 | 512,00 | 2.412,60 | 97,77 | 16,23 | Batch 1 từ feature cache (15 lớp) |
| 43 | Allocation v2 | reconstruction · 512 KiB | 1 | 54,92 | 61,41 | 87,95 | 94,10 | 512,00 | 2.416,83 | 70,61 | 9,53 | Batch 1 từ feature cache (15 lớp) |
| 44 | Allocation v2 | coverage · 512 KiB | 1 | 44,41 | 51,68 | 83,10 | 86,51 | 512,00 | 2.438,84 | 134,30 | 14,79 | Batch 1 từ feature cache (15 lớp) |
| 45 | Allocation v2 | whitened_joint · 512 KiB | 1 | 44,86 | 50,59 | 81,79 | 84,40 | 512,00 | 2.416,83 | 98,74 | 13,27 | Batch 1 từ feature cache (15 lớp) |
| 46 | Baseline PCA | raw · 1 MiB | 1 | 58,40 | 65,00 | 88,11 | 96,66 | 1.020,71 | 2.957,37 | 336,21 | 3,42 | Batch 1 từ feature cache (15 lớp) |
| 47 | Baseline PCA | raw · 2 MiB | 1 | 59,78 | 67,57 | 90,75 | 97,62 | 2.044,71 | 2.990,06 | 339,44 | 3,54 | Batch 1 từ feature cache (15 lớp) |
| 48 | Baseline PCA | raw · 4 MiB | 1 | 61,33 | 69,04 | 91,45 | 98,70 | 4.092,71 | 2.963,37 | 343,81 | 3,61 | Batch 1 từ feature cache (15 lớp) |
| 49 | Baseline PCA | raw · 8 MiB | 1 | 62,57 | 70,03 | 92,22 | 98,93 | 8.188,71 | 3.002,06 | 356,98 | 3,51 | Batch 1 từ feature cache (15 lớp) |
| 50 | Baseline PCA | pca128 · 1 MiB | 1 | 58,67 | 67,04 | 89,89 | 97,59 | 1.023,73 | 2.955,37 | 339,07 | 4,34 | Batch 1 từ feature cache (15 lớp) |
| 51 | Baseline PCA | pca128 · 2 MiB | 1 | 61,60 | 69,38 | 91,72 | 98,71 | 2.047,73 | 2.989,05 | 355,80 | 3,55 | Batch 1 từ feature cache (15 lớp) |
| 52 | Baseline PCA | pca128 · 4 MiB | 1 | 62,09 | 69,69 | 92,24 | 98,64 | 4.095,73 | 2.962,37 | 394,81 | 3,62 | Batch 1 từ feature cache (15 lớp) |
| 53 | Baseline PCA | pca128 · 8 MiB | 1 | 62,75 | 70,11 | 92,50 | 98,56 | 8.191,73 | 3.001,06 | 472,38 | 3,62 | Batch 1 từ feature cache (15 lớp) |
| 54 | Allocation v2 | Ứng viên PCA theo ngưỡng phương sai 95% · 256 KiB | 1 | 49,31 | 55,48 | 82,26 | 91,56 | 255,85 | 2.410,49 | 66,99 | 9,67 | Batch 1 từ feature cache (15 lớp) |
| 55 | Allocation v2 | Ứng viên PCA theo ngưỡng phương sai 95% · 512 KiB | 1 | 53,44 | 60,69 | 88,38 | 93,80 | 511,81 | 2.402,45 | 70,04 | 11,23 | Batch 1 từ feature cache (15 lớp) |
| 56 | Allocation v2 | Ứng viên pca_d128__b262144 | 1 | 47,79 | 53,92 | 80,16 | 90,81 | 256,00 | 2.410,49 | 66,96 | 9,03 | Batch 1 từ feature cache (15 lớp) |
| 57 | Allocation v2 | Ứng viên pca_d128__b524288 | 1 | 54,92 | 61,41 | 87,95 | 94,10 | 512,00 | 2.416,83 | 70,61 | 9,53 | Batch 1 từ feature cache (15 lớp) |
| 58 | Allocation v2 | Ứng viên pca_d16__b262144 | 1 | 44,23 | 51,63 | 83,26 | 86,70 | 256,00 | 2.402,45 | 97,29 | 13,12 | Batch 1 từ feature cache (15 lớp) |
| 59 | Allocation v2 | Ứng viên pca_d16__b524288 | 1 | 44,41 | 51,68 | 83,10 | 86,51 | 512,00 | 2.438,84 | 134,30 | 14,79 | Batch 1 từ feature cache (15 lớp) |
| 60 | Allocation v2 | Ứng viên pca_d32__b262144 | 1 | 49,01 | 55,78 | 85,55 | 90,44 | 256,00 | 2.402,45 | 79,16 | 8,46 | Batch 1 từ feature cache (15 lớp) |
| 61 | Allocation v2 | Ứng viên pca_d32__b524288 | 1 | 49,61 | 56,60 | 86,50 | 90,21 | 512,00 | 2.412,60 | 97,77 | 11,61 | Batch 1 từ feature cache (15 lớp) |
| 62 | Allocation v2 | Ứng viên pca_d64__b262144 | 1 | 52,36 | 58,63 | 86,31 | 91,42 | 256,00 | 2.402,45 | 70,13 | 10,66 | Batch 1 từ feature cache (15 lớp) |
| 63 | Allocation v2 | Ứng viên pca_d64__b524288 | 1 | 53,85 | 60,46 | 88,57 | 92,92 | 512,00 | 2.410,49 | 79,57 | 9,80 | Batch 1 từ feature cache (15 lớp) |
| 64 | Allocation v2 | Ứng viên raw_d384__b262144 | 1 | 51,22 | 57,31 | 84,79 | 92,61 | 256,00 | 2.407,81 | 67,60 | 6,78 | Batch 1 từ feature cache (15 lớp) |
| 65 | Allocation v2 | Ứng viên raw_d384__b524288 | 1 | 54,54 | 60,08 | 87,75 | 95,27 | 511,00 | 2.402,45 | 69,48 | 7,07 | Batch 1 từ feature cache (15 lớp) |
| 66 | Allocation v2 | Ứng viên whitened_d128__b262144 | 1 | 54,18 | 59,26 | 87,14 | 94,31 | 256,00 | 2.438,84 | 66,96 | 9,66 | Batch 1 từ feature cache (15 lớp) |
| 67 | Allocation v2 | Ứng viên whitened_d128__b524288 | 1 | 56,25 | 62,31 | 89,51 | 95,30 | 512,00 | 2.416,83 | 70,61 | 8,39 | Batch 1 từ feature cache (15 lớp) |
| 68 | Allocation v2 | Ứng viên whitened_d16__b262144 | 1 | 41,08 | 47,95 | 80,55 | 84,00 | 256,00 | 2.411,11 | 97,29 | 9,12 | Batch 1 từ feature cache (15 lớp) |
| 69 | Allocation v2 | Ứng viên whitened_d16__b524288 | 1 | 41,18 | 47,88 | 80,25 | 83,24 | 512,00 | 2.416,83 | 134,30 | 10,67 | Batch 1 từ feature cache (15 lớp) |
| 70 | Allocation v2 | Ứng viên whitened_d32__b262144 | 1 | 47,69 | 54,09 | 84,52 | 89,15 | 256,00 | 2.438,84 | 79,16 | 11,69 | Batch 1 từ feature cache (15 lớp) |
| 71 | Allocation v2 | Ứng viên whitened_d32__b524288 | 1 | 48,14 | 54,39 | 85,24 | 88,92 | 512,00 | 2.416,83 | 97,77 | 11,02 | Batch 1 từ feature cache (15 lớp) |
| 72 | Allocation v2 | Ứng viên whitened_d64__b262144 | 1 | 53,63 | 59,38 | 87,12 | 92,84 | 256,00 | 2.416,83 | 70,13 | 10,81 | Batch 1 từ feature cache (15 lớp) |
| 73 | Allocation v2 | Ứng viên whitened_d64__b524288 | 1 | 55,38 | 60,77 | 89,08 | 93,13 | 512,00 | 2.438,84 | 79,57 | 8,25 | Batch 1 từ feature cache (15 lớp) |
| 74 | Nén bank | fp32-10pct | 5 | 50,37 | 61,93 | 89,75 | 96,50 | 4.915,21 | 2.018,43 | 174,65 | 25,74 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 75 | Nén bank | fp32-budget · trần 512 KiB | 5 | 43,96 | 55,30 | 82,81 | 93,66 | 511,21 | 2.013,62 | 170,35 | 26,85 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 76 | Nén bank | fp16-full · trần 512 KiB | 1 | 46,85 | 58,36 | 85,75 | 95,09 | 511,21 | 2.013,62 | 170,35 | 28,28 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 77 | Nén bank | int8-full · trần 512 KiB | 5 | 48,11 | 60,21 | 87,79 | 96,24 | 511,21 | 1.798,29 | 170,35 | 20,96 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 78 | Nén bank | pca64-int8 · trần 512 KiB | 1 | 46,26 | 57,85 | 87,47 | 92,86 | 511,96 | 1.798,29 | 170,35 | 24,35 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 79 | Nén bank | pca128-int8 · trần 512 KiB | 5 | 48,16 | 59,66 | 87,69 | 94,49 | 511,96 | 1.798,29 | 170,35 | 31,52 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 80 | Nén bank | pca64-int8-residual · trần 512 KiB | 5 | 49,41 | 60,43 | 89,29 | 94,94 | 512,00 | 1.798,29 | 170,35 | 31,06 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 81 | Nén bank | pca128-int8-residual · trần 512 KiB | 5 | 48,95 | 60,49 | 88,50 | 95,54 | 511,94 | 1.798,29 | 170,35 | 23,31 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 82 | Nén bank | fp32-budget · trần 1024 KiB | 5 | 47,14 | 58,53 | 86,01 | 95,47 | 1.021,21 | 1.799,29 | 170,85 | 21,03 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 83 | Nén bank | fp16-full · trần 1024 KiB | 1 | 47,54 | 60,02 | 87,66 | 96,26 | 1.021,21 | 1.799,29 | 170,85 | 21,90 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 84 | Nén bank | int8-full · trần 1024 KiB | 5 | 49,98 | 61,60 | 89,40 | 96,47 | 1.022,71 | 1.799,29 | 170,85 | 25,64 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 85 | Nén bank | pca64-int8 · trần 1024 KiB | 1 | 46,30 | 57,82 | 87,41 | 92,73 | 711,46 | 1.798,29 | 170,55 | 35,11 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 86 | Nén bank | pca128-int8 · trần 1024 KiB | 5 | 49,96 | 61,01 | 89,17 | 94,28 | 1.023,96 | 1.799,29 | 170,85 | 27,84 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 87 | Nén bank | pca64-int8-residual · trần 1024 KiB | 5 | 49,46 | 60,52 | 89,28 | 94,87 | 727,46 | 1.798,29 | 170,56 | 29,62 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 88 | Nén bank | pca128-int8-residual · trần 1024 KiB | 5 | 50,30 | 61,62 | 89,77 | 95,46 | 1.023,94 | 1.799,29 | 170,85 | 30,41 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 89 | Nén bank | int4-full · trần 512 KiB | 5 | 50,10 | 61,65 | 89,44 | 96,48 | 511,96 | 1.798,29 | 170,35 | 26,50 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 90 | Nén bank | int4-full · trần 1024 KiB | 5 | 50,92 | 62,44 | 90,29 | 96,48 | 1.023,46 | 1.799,29 | 170,85 | 25,82 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 91 | Nén bank | uint4-full · trần 512 KiB | 5 | 50,00 | 61,60 | 89,41 | 96,45 | 511,46 | 1.798,29 | 170,35 | 27,00 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 92 | Nén bank | uint4-full · trần 1024 KiB | 5 | 50,91 | 62,44 | 90,28 | 96,48 | 1.023,71 | 1.799,29 | 170,85 | 25,94 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 93 | Điểm patch | baseline | 5 | 50,92 | 62,44 | 90,29 | 96,48 | 1.023,46 | 1.799,29 | 170,85 | 26,68 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 94 | Điểm patch | position | 1 | 50,36 | 61,23 | 89,72 | 97,08 | 1.023,56 | 1.799,29 | 170,86 | 31,44 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 95 | Điểm patch | position-center | 1 | 55,97 | 64,30 | 89,86 | 96,95 | 1.023,58 | 1.799,29 | 170,86 | 23,55 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 96 | Điểm patch | position-mean | 1 | 50,60 | 61,41 | 89,84 | 96,91 | 1.023,58 | 1.799,29 | 170,86 | 21,71 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 97 | Điểm patch | image-center | 1 | 55,58 | 64,87 | 90,04 | 96,19 | 1.023,51 | 1.799,29 | 170,85 | 23,05 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 98 | Điểm patch | position-k3 | 1 | 50,73 | 60,81 | 89,92 | 97,10 | 1.023,57 | 1.799,29 | 170,86 | 23,64 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 99 | Điểm patch | position-sigma2 | 1 | 51,22 | 61,74 | 89,74 | 97,08 | 1.023,58 | 1.799,29 | 170,86 | 21,71 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 100 | Điểm patch | position-layer2 | 5 | 62,57 | 67,56 | 90,50 | 95,99 | 344,58 | 2.012,20 | 161,95 | 14,68 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 101 | Điểm patch | layer2 | 1 | 62,64 | 68,44 | 91,21 | 95,47 | 1.023,49 | 1.799,29 | 170,85 | 23,76 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 102 | Điểm patch | position-layer-mix | 1 | 58,51 | 66,10 | 91,11 | 96,86 | 1.023,60 | 1.799,29 | 170,86 | 21,03 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 103 | Điểm patch | position-cosine | 1 | 45,83 | 57,95 | 86,39 | 96,64 | 1.023,58 | 1.828,57 | 170,86 | 25,05 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 104 | Điểm patch | cosine | 1 | 41,03 | 52,08 | 81,41 | 85,40 | 1.023,49 | 1.829,56 | 170,85 | 24,54 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 105 | Điểm patch | local-contrast | 5 | 63,53 | 68,98 | 90,62 | 96,48 | 1.023,52 | 1.829,59 | 170,85 | 24,83 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 106 | Điểm patch | position-contrast | 5 | 62,93 | 67,84 | 90,04 | 96,47 | 1.023,59 | 1.829,65 | 170,86 | 23,03 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 107 | Điểm patch | layer2-contrast | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 342,52 | 2.012,20 | 161,95 | 21,21 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 108 | Ablation INT4 ban đầu | fp32-reference | 5 | 50,37 | 61,93 | 89,75 | 96,50 | 4.915,21 | 1.833,45 | 174,65 | 20,79 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 109 | Ablation INT4 ban đầu | fp32-budget | 5 | 47,14 | 58,53 | 86,01 | 95,47 | 1.021,21 | 1.829,65 | 170,85 | 23,86 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 110 | Ablation INT4 ban đầu | fp32-matched | 5 | 50,90 | 62,43 | 90,28 | 96,48 | 8.125,21 | 1.836,58 | 177,79 | 24,51 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 111 | Ablation INT4 ban đầu | int4-full | 5 | 50,91 | 62,44 | 90,29 | 96,48 | 1.022,71 | 1.829,65 | 170,85 | 25,33 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 112 | Ablation INT4 ban đầu | int4-full-contrast | 5 | 63,53 | 68,98 | 90,62 | 96,48 | 1.022,71 | 1.829,65 | 170,85 | 30,11 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 113 | Ablation INT4 ban đầu | int4-layer2 | 5 | 62,77 | 68,56 | 91,11 | 95,50 | 341,71 | 2.012,20 | 161,95 | 14,22 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 114 | Ablation INT4 ban đầu | int4-layer2-contrast | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 341,71 | 2.012,20 | 161,95 | 21,07 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 115 | Codec lossless | original | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 342,52 | 2.012,20 | 161,95 | 18,36 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 116 | Codec lossless | deflate-int4 | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 222,02 | 2.012,20 | 161,95 | 18,14 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 117 | Codec lossless | lzma-int4 | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 199,09 | 2.012,20 | 161,95 | 21,96 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 118 | Codec lossless | deflate-planes3 | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 202,02 | 2.012,20 | 161,95 | 16,18 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 119 | Codec lossless | lzma-planes3 | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 2.012,20 | 161,95 | 14,41 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 120 | INT4 Fast | Layer2 + contrast 7×7 | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 1.782,12 | 167,02 | 16,95 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 121 | INT4 Fast | Layer2 7×7 · sequential | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 1.332,95 | 66,94 | — | Batch 1 · thông lượng pipeline (3 lớp) |
| 122 | INT4 Fast | Layer2 7×7 · cached | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 1.331,20 | 69,26 | — | Batch 1 · thông lượng pipeline (3 lớp) |
| 123 | INT4 Fast | Layer2 7×7 · prefetch2 | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 1.334,33 | 69,26 | — | Batch 1 · thông lượng pipeline (3 lớp) |
| 124 | INT4 Fast | Layer2 7×7 · prefetch2-norms | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 1.333,89 | 66,95 | — | Batch 1 · thông lượng pipeline (3 lớp) |
| 125 | INT4 Fast | Scorer · tiled | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 1.802,68 | 164,70 | 22,68 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 126 | INT4 Fast | Scorer · previous | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 1.801,30 | 164,70 | 31,20 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 127 | INT4 Fast | Scorer · norms | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 1.802,66 | 164,70 | 29,15 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 128 | INT4 Fast | Scorer · decoded | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 1.802,66 | 167,02 | 17,74 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 129 | INT4 Balanced | w=7 strength=0.75 σ=4 | 5 | 66,29 | 66,37 | 86,93 | 95,50 | 190,66 | 2.012,21 | 161,95 | 13,15 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 130 | INT4 Balanced | w=7 strength=0 σ=4 | 1 | 62,64 | 68,44 | 91,21 | 95,47 | 190,63 | 2.012,21 | 161,95 | 14,57 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 131 | INT4 Balanced | w=7 strength=0.4 σ=4 | 1 | 65,83 | 69,05 | 90,55 | 95,47 | 190,63 | 2.012,21 | 161,95 | 17,15 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 132 | INT4 Balanced | w=7 strength=0.55 σ=4 | 1 | 66,68 | 68,52 | 89,68 | 95,47 | 190,63 | 2.012,21 | 161,95 | 16,82 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 133 | INT4 Balanced | w=7 strength=0.65 σ=4 | 1 | 66,87 | 67,70 | 88,66 | 95,47 | 190,63 | 2.012,20 | 161,95 | 20,17 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 134 | INT4 Balanced | w=5 strength=0.55 σ=2 | 1 | 64,69 | 66,32 | 88,37 | 95,47 | 190,63 | 2.012,20 | 161,95 | 15,69 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 135 | INT4 Balanced | w=7 strength=0.55 σ=2 | 1 | 65,85 | 67,09 | 88,56 | 95,47 | 190,63 | 2.012,20 | 161,95 | 21,31 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 136 | INT4 Balanced | w=9 strength=0.55 σ=2 | 1 | 66,28 | 67,70 | 88,89 | 95,47 | 190,63 | 2.012,20 | 161,95 | 14,65 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 137 | INT4 Balanced | w=11 strength=0.55 σ=4 | 1 | 66,79 | 69,09 | 90,01 | 95,47 | 190,63 | 2.012,20 | 161,95 | 19,55 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 138 | INT4 Balanced | w=11 strength=0.65 σ=2 | 1 | 66,52 | 67,56 | 88,39 | 95,47 | 190,63 | 2.012,20 | 161,95 | 15,88 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 139 | INT4 Balanced | w=11 strength=0.75 σ=4 | 5 | 67,01 | 68,09 | 88,21 | 95,50 | 190,66 | 2.012,21 | 161,95 | 17,98 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 140 | INT4 Balanced | w=15 strength=0.75 σ=2 | 1 | 66,61 | 67,98 | 88,55 | 95,47 | 190,63 | 2.012,21 | 161,95 | 15,08 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 141 | INT4 Balanced | w=15 strength=0.75 σ=4 | 5 | 66,95 | 68,95 | 89,19 | 95,50 | 190,66 | 2.012,21 | 161,95 | 16,43 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 142 | INT4 Robust | standard:11:0.0 · top-1 | 5 | 67,01 | 68,09 | 88,21 | 95,50 | 190,66 | 2.012,21 | 161,95 | 20,99 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 143 | INT4 Robust | standard:11:0.0 · top-4 | 5 | 67,01 | 68,09 | 88,21 | 96,26 | 190,66 | 2.012,21 | 161,95 | 14,03 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 144 | INT4 Robust | standard:11:0.0 · top-16 | 5 | 67,01 | 68,09 | 88,21 | 96,48 | 190,66 | 2.012,21 | 161,95 | 16,09 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 145 | INT4 Robust | standard:11:0.0 · top-32 | 5 | 67,01 | 68,09 | 88,21 | 96,09 | 190,66 | 2.012,21 | 161,95 | 21,85 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 146 | INT4 Robust | standard:15:0.0 · top-1 | 5 | 66,95 | 68,95 | 89,19 | 95,50 | 190,66 | 2.012,21 | 161,95 | 15,71 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 147 | INT4 Robust | standard:15:0.0 · top-4 | 5 | 66,95 | 68,95 | 89,19 | 96,26 | 190,66 | 2.012,21 | 161,95 | 19,36 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 148 | INT4 Robust | standard:15:0.0 · top-16 | 5 | 66,95 | 68,95 | 89,19 | 96,48 | 190,66 | 2.012,21 | 161,95 | 18,59 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 149 | INT4 Robust | standard:15:0.0 · top-32 | 5 | 66,95 | 68,95 | 89,19 | 96,09 | 190,66 | 2.012,21 | 161,95 | 16,05 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 150 | INT4 Robust | standard:19:0.0 · top-1 | 5 | 66,64 | 69,41 | 89,86 | 95,50 | 190,66 | 2.012,21 | 161,95 | 14,93 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 151 | INT4 Robust | standard:19:0.0 · top-4 | 5 | 66,64 | 69,41 | 89,86 | 96,26 | 190,66 | 2.012,21 | 161,95 | 22,31 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 152 | INT4 Robust | standard:19:0.0 · top-16 | 5 | 66,64 | 69,41 | 89,86 | 96,48 | 190,66 | 2.012,20 | 161,95 | 16,13 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 153 | INT4 Robust | standard:19:0.0 · top-32 | 5 | 66,64 | 69,41 | 89,86 | 96,09 | 190,66 | 2.012,19 | 161,95 | 24,48 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 154 | INT4 Robust | standard:23:0.0 · top-1 | 1 | 66,16 | 69,53 | 90,43 | 95,47 | 190,63 | 2.012,20 | 161,95 | 20,79 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 155 | INT4 Robust | standard:23:0.0 · top-4 | 1 | 66,16 | 69,53 | 90,43 | 96,51 | 190,63 | 2.012,20 | 161,95 | 16,62 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 156 | INT4 Robust | standard:23:0.0 · top-16 | 1 | 66,16 | 69,53 | 90,43 | 96,84 | 190,63 | 2.012,20 | 161,95 | 16,80 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 157 | INT4 Robust | standard:23:0.0 · top-32 | 1 | 66,16 | 69,53 | 90,43 | 96,47 | 190,63 | 2.012,20 | 161,95 | 18,03 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 158 | INT4 Robust | before:11:0.85 · top-1 | 1 | 63,62 | 67,98 | 90,51 | 95,47 | 190,63 | 2.012,20 | 161,95 | 23,78 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 159 | INT4 Robust | before:11:0.85 · top-4 | 1 | 63,62 | 67,98 | 90,51 | 96,51 | 190,63 | 2.012,20 | 161,95 | 23,81 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 160 | INT4 Robust | before:11:0.85 · top-16 | 1 | 63,62 | 67,98 | 90,51 | 96,84 | 190,63 | 2.012,20 | 161,95 | 20,79 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 161 | INT4 Robust | before:11:0.85 · top-32 | 1 | 63,62 | 67,98 | 90,51 | 96,47 | 190,63 | 2.012,20 | 161,95 | 18,20 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 162 | INT4 Robust | before:11:0.95 · top-1 | 1 | 64,16 | 67,88 | 90,05 | 95,47 | 190,63 | 2.012,20 | 161,95 | 17,97 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 163 | INT4 Robust | before:11:0.95 · top-4 | 1 | 64,16 | 67,88 | 90,05 | 96,51 | 190,63 | 2.012,20 | 161,95 | 13,55 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 164 | INT4 Robust | before:11:0.95 · top-16 | 1 | 64,16 | 67,88 | 90,05 | 96,84 | 190,63 | 2.012,20 | 161,95 | 19,48 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 165 | INT4 Robust | before:11:0.95 · top-32 | 1 | 64,16 | 67,88 | 90,05 | 96,47 | 190,63 | 2.012,20 | 161,95 | 18,31 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 166 | INT4 Robust | before:11:0.99 · top-1 | 1 | 64,79 | 67,72 | 89,50 | 95,47 | 190,63 | 2.012,20 | 161,95 | 25,32 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 167 | INT4 Robust | before:11:0.99 · top-4 | 1 | 64,79 | 67,72 | 89,50 | 96,51 | 190,63 | 2.012,19 | 161,95 | 18,85 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 168 | INT4 Robust | before:11:0.99 · top-16 | 1 | 64,79 | 67,72 | 89,50 | 96,84 | 190,63 | 2.012,20 | 161,95 | 16,90 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 169 | INT4 Robust | before:11:0.99 · top-32 | 1 | 64,79 | 67,72 | 89,50 | 96,47 | 190,63 | 2.012,20 | 161,95 | 21,13 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 170 | INT4 Robust | after:11:0.85 · top-1 | 1 | 63,35 | 68,06 | 89,90 | 95,47 | 190,63 | 2.012,20 | 161,95 | 14,52 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 171 | INT4 Robust | after:11:0.85 · top-4 | 1 | 63,35 | 68,06 | 89,90 | 96,51 | 190,63 | 2.012,20 | 161,95 | 16,88 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 172 | INT4 Robust | after:11:0.85 · top-16 | 1 | 63,35 | 68,06 | 89,90 | 96,84 | 190,63 | 2.012,20 | 161,95 | 22,26 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 173 | INT4 Robust | after:11:0.85 · top-32 | 1 | 63,35 | 68,06 | 89,90 | 96,47 | 190,63 | 2.012,20 | 161,95 | 21,86 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 174 | INT4 Robust | after:11:0.95 · top-1 | 1 | 64,60 | 68,01 | 89,28 | 95,47 | 190,63 | 2.012,20 | 161,95 | 18,03 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 175 | INT4 Robust | after:11:0.95 · top-4 | 1 | 64,60 | 68,01 | 89,28 | 96,51 | 190,63 | 2.012,20 | 161,95 | 18,16 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 176 | INT4 Robust | after:11:0.95 · top-16 | 1 | 64,60 | 68,01 | 89,28 | 96,84 | 190,63 | 2.012,20 | 161,95 | 18,31 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 177 | INT4 Robust | after:11:0.95 · top-32 | 1 | 64,60 | 68,01 | 89,28 | 96,47 | 190,63 | 2.012,20 | 161,95 | 16,78 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 178 | INT4 Robust | after:11:0.99 · top-1 | 1 | 65,79 | 68,10 | 88,89 | 95,47 | 190,63 | 2.012,20 | 161,95 | 19,97 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 179 | INT4 Robust | after:11:0.99 · top-4 | 1 | 65,79 | 68,10 | 88,89 | 96,51 | 190,63 | 2.012,20 | 161,95 | 20,01 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 180 | INT4 Robust | after:11:0.99 · top-16 | 1 | 65,79 | 68,10 | 88,89 | 96,84 | 190,63 | 2.012,20 | 161,95 | 16,59 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 181 | INT4 Robust | after:11:0.99 · top-32 | 1 | 65,79 | 68,10 | 88,89 | 96,47 | 190,63 | 2.012,20 | 161,95 | 19,94 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 182 | INT4 Robust | before:15:0.95 · top-1 | 1 | 64,30 | 68,28 | 90,30 | 95,47 | 190,63 | 2.012,20 | 161,95 | 19,66 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 183 | INT4 Robust | before:15:0.95 · top-4 | 1 | 64,30 | 68,28 | 90,30 | 96,51 | 190,63 | 2.012,20 | 161,95 | 20,08 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 184 | INT4 Robust | before:15:0.95 · top-16 | 1 | 64,30 | 68,28 | 90,30 | 96,84 | 190,63 | 2.012,20 | 161,95 | 13,55 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 185 | INT4 Robust | before:15:0.95 · top-32 | 1 | 64,30 | 68,28 | 90,30 | 96,47 | 190,63 | 2.012,20 | 161,95 | 24,32 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 186 | INT4 Robust | after:15:0.95 · top-1 | 1 | 65,43 | 68,78 | 89,75 | 95,47 | 190,63 | 2.012,20 | 161,95 | 16,72 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 187 | INT4 Robust | after:15:0.95 · top-4 | 1 | 65,43 | 68,78 | 89,75 | 96,51 | 190,63 | 2.012,20 | 161,95 | 22,76 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 188 | INT4 Robust | after:15:0.95 · top-16 | 1 | 65,43 | 68,78 | 89,75 | 96,84 | 190,63 | 2.012,20 | 161,95 | 16,85 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 189 | INT4 Robust | after:15:0.95 · top-32 | 1 | 65,43 | 68,78 | 89,75 | 96,47 | 190,63 | 2.012,20 | 161,95 | 22,28 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 190 | INT4 Robust | after:19:0.99 · top-1 | 1 | 66,43 | 69,37 | 90,04 | 95,47 | 190,63 | 2.012,20 | 161,95 | 17,16 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 191 | INT4 Robust | after:19:0.99 · top-4 | 1 | 66,43 | 69,37 | 90,04 | 96,51 | 190,63 | 2.012,20 | 161,95 | 16,73 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 192 | INT4 Robust | after:19:0.99 · top-16 | 1 | 66,43 | 69,37 | 90,04 | 96,84 | 190,63 | 2.012,20 | 161,95 | 21,76 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 193 | INT4 Robust | after:19:0.99 · top-32 | 1 | 66,43 | 69,37 | 90,04 | 96,47 | 190,63 | 2.012,20 | 161,95 | 20,72 | Batch 1 · bắt đầu từ ảnh đã tiền xử lý (3 lớp) |
| 194 | Benchmark rộng | aso-fp32 | 1 | 46,28 | 57,83 | 85,38 | 94,98 | 788,41 | 1.403,91 | 155,84 | 37,98 | Batch 1 (15 lớp) |
| 195 | Benchmark rộng | dual-dense | 5 | 70,60 | 70,09 | 88,29 | 98,24 | 330,42 | 1.922,96 | 67,86 | 45,36 | Batch 1 (15 lớp) |
| 196 | Benchmark rộng | dual-fusion25 | 5 | 72,80 | 73,22 | 90,25 | 98,24 | 330,42 | 1.831,48 | 67,86 | 46,23 | Batch 1 (15 lớp) |
| 197 | Benchmark rộng | dual-fusion50 | 5 | 72,42 | 73,82 | 90,82 | 98,24 | 330,42 | 1.831,51 | 67,86 | 46,22 | Batch 1 (15 lớp) |
| 198 | Benchmark rộng | fp32-10pct | 5 | 50,37 | 61,93 | 89,75 | 96,50 | 4.915,21 | 1.406,81 | 160,50 | 41,90 | Batch 1 (15 lớp) |
| 199 | Benchmark rộng | fusion-centroids | 1 | 59,92 | 61,87 | 84,79 | 94,21 | 222,59 | 1.501,88 | 155,65 | 41,46 | Batch 1 (15 lớp) |
| 200 | Benchmark rộng | fusion-pca64-wide | 1 | 57,39 | 62,58 | 84,27 | 94,92 | 222,59 | 1.502,43 | 155,65 | 40,58 | Batch 1 (15 lớp) |
| 201 | Benchmark rộng | fusion-pca64 | 1 | 56,71 | 60,56 | 82,16 | 94,92 | 222,59 | 1.503,07 | 155,65 | 42,13 | Batch 1 (15 lớp) |
| 202 | Benchmark rộng | fusion-spatial | 1 | 58,76 | 60,84 | 83,31 | 95,82 | 223,58 | 1.644,39 | 156,23 | 40,01 | Batch 1 (15 lớp) |
| 203 | Benchmark rộng | layer1-dense | 1 | 65,54 | 63,11 | 83,15 | 92,03 | 94,96 | 1.587,59 | 44,90 | 36,85 | Batch 1 (15 lớp) |
| 204 | Benchmark rộng | layer1-layer2-wide | 5 | 70,60 | 70,09 | 88,29 | 93,75 | 188,46 | 1.575,96 | 78,43 | 39,44 | Batch 1 (15 lớp) |
| 205 | Benchmark rộng | layer1-layer2 | 1 | 69,45 | 67,69 | 85,92 | 93,94 | 188,46 | 1.576,06 | 77,85 | 43,70 | Batch 1 (15 lớp) |
| 206 | Benchmark rộng | layer2-pca96 | 1 | 60,99 | 63,57 | 84,75 | 95,56 | 134,28 | 1.576,37 | 66,92 | 34,33 | Batch 1 (15 lớp) |
| 207 | Benchmark rộng | lite-pca32 | 1 | 46,45 | 56,61 | 83,25 | 91,45 | 149,46 | 1.680,71 | 155,39 | 38,56 | Batch 1 (15 lớp) |
| 208 | Benchmark rộng | mh-pca64 | 1 | 56,17 | 62,39 | 87,60 | 94,93 | 245,71 | 1.466,99 | 155,67 | 38,76 | Batch 1 (15 lớp) |
| 209 | Benchmark rộng | robust-11 | 5 | 67,01 | 68,09 | 88,21 | 96,48 | 190,66 | 1.578,37 | 66,94 | 33,35 | Batch 1 (15 lớp) |
| 210 | Benchmark rộng | robust-19 | 5 | 66,64 | 69,41 | 89,86 | 96,48 | 190,66 | 1.577,45 | 66,94 | 37,91 | Batch 1 (15 lớp) |
| 211 | Benchmark rộng | robust-cal | 1 | 67,89 | 69,29 | 89,26 | 96,84 | 204,93 | 1.628,72 | 66,94 | 34,81 | Batch 1 (15 lớp) |
| 212 | Benchmark rộng | robust-cal11 | 5 | 68,07 | 68,94 | 88,52 | 96,48 | 204,93 | 1.627,30 | 66,94 | 38,56 | Batch 1 (15 lớp) |
| 213 | Benchmark rộng | robust-cal19 | 5 | 67,68 | 70,25 | 90,29 | 96,48 | 204,93 | 1.627,53 | 66,94 | 35,56 | Batch 1 (15 lớp) |
| 214 | Benchmark rộng | robust-cal21 | 1 | 67,14 | 70,06 | 90,51 | 96,84 | 204,93 | 1.627,46 | 66,94 | 38,70 | Batch 1 (15 lớp) |
| 215 | Benchmark rộng | robust-cal27 | 1 | 66,28 | 70,14 | 91,03 | 96,84 | 204,93 | 1.770,15 | 66,94 | 36,29 | Batch 1 (15 lớp) |
| 216 | Benchmark rộng | robust-coarse11 | 1 | 67,51 | 68,01 | 87,88 | 96,84 | 191,62 | 1.627,59 | 66,94 | 38,71 | Batch 1 (15 lớp) |
| 217 | Benchmark rộng | robust-coarse15-k24 | 1 | 67,57 | 68,98 | 89,01 | 96,69 | 191,62 | 1.627,90 | 66,94 | 42,12 | Batch 1 (15 lớp) |
| 218 | Benchmark rộng | robust-coarse15-k32 | 1 | 67,57 | 68,98 | 89,01 | 96,47 | 191,62 | 1.628,77 | 66,94 | 37,63 | Batch 1 (15 lớp) |
| 219 | Benchmark rộng | robust-coarse15 | 5 | 67,40 | 69,03 | 88,88 | 96,48 | 191,64 | 1.628,34 | 66,94 | 35,16 | Batch 1 (15 lớp) |
| 220 | Benchmark rộng | robust-coarse19-k24 | 1 | 67,26 | 69,61 | 89,92 | 96,69 | 191,62 | 1.627,84 | 66,94 | 36,73 | Batch 1 (15 lớp) |
| 221 | Benchmark rộng | robust-coarse19-k32 | 1 | 67,26 | 69,61 | 89,92 | 96,47 | 191,62 | 1.628,07 | 66,94 | 36,10 | Batch 1 (15 lớp) |
| 222 | Benchmark rộng | robust-coarse19 | 5 | 67,24 | 69,69 | 89,80 | 96,48 | 191,64 | 1.628,44 | 66,94 | 37,45 | Batch 1 (15 lớp) |
| 223 | Benchmark rộng | robust-sharp | 1 | 66,61 | 67,98 | 88,55 | 96,84 | 190,63 | 1.578,61 | 66,94 | 36,48 | Batch 1 (15 lớp) |
| 224 | Benchmark rộng | robust-strong | 1 | 67,00 | 68,22 | 88,36 | 96,84 | 190,63 | 1.719,67 | 66,94 | 35,54 | Batch 1 (15 lớp) |
| 225 | Benchmark rộng | soft-int4 | 1 | 64,73 | 66,13 | 87,42 | 94,61 | 189,91 | 1.588,70 | 66,79 | 34,72 | Batch 1 (15 lớp) |
| 226 | Benchmark rộng | uint4-32k-cal | 5 | 65,71 | 69,38 | 88,93 | 98,24 | 158,08 | 1.626,23 | 66,79 | 38,52 | Batch 1 (15 lớp) |
| 227 | Benchmark rộng | uint4-32k | 1 | 64,88 | 69,16 | 88,69 | 98,15 | 188,46 | 1.576,56 | 66,78 | 39,53 | Batch 1 (15 lớp) |
| 228 | Benchmark rộng | uint4-384 | 1 | 62,67 | 65,55 | 85,86 | 96,75 | 99,46 | 1.549,92 | 66,69 | 34,22 | Batch 1 (15 lớp) |
| 229 | Benchmark rộng | uint4-cal11 | 1 | 66,66 | 67,77 | 87,38 | 97,14 | 159,99 | 1.625,73 | 66,79 | 38,55 | Batch 1 (15 lớp) |
| 230 | Benchmark rộng | uint4-cal19 | 5 | 66,75 | 69,57 | 89,41 | 96,74 | 160,01 | 1.625,94 | 66,79 | 38,54 | Batch 1 (15 lớp) |
| 231 | Benchmark rộng | uint4-cal27 | 5 | 65,75 | 69,95 | 90,48 | 96,74 | 160,01 | 1.626,21 | 66,79 | 36,36 | Batch 1 (15 lớp) |
| 232 | Benchmark rộng | uint4-pos | 1 | 64,93 | 68,52 | 89,15 | 97,20 | 189,45 | 1.717,44 | 66,79 | 41,13 | Batch 1 (15 lớp) |
| 233 | Benchmark rộng | uint4-strong | 1 | 66,39 | 69,26 | 89,52 | 97,14 | 188,46 | 1.575,46 | 66,78 | 35,19 | Batch 1 (15 lớp) |
| 234 | Benchmark rộng | uint4 | 1 | 66,54 | 68,33 | 88,37 | 97,14 | 188,46 | 1.576,16 | 66,78 | 37,91 | Batch 1 (15 lớp) |
| 235 | Backend | robust-coarse15 · fp16-packed | 5 | 67,40 | 69,03 | 88,88 | 96,48 | 191,64 | 1.975,24 | 45,86 | 41,48 | Batch 1 · parity backend (15 lớp) |
| 236 | Backend | robust-coarse19 · fp16-packed | 5 | 67,24 | 69,69 | 89,80 | 96,48 | 191,64 | 1.974,61 | 45,86 | 44,34 | Batch 1 · parity backend (15 lớp) |
| 237 | Backend | uint4-32k-cal · fp16-packed | 5 | 65,71 | 69,39 | 88,93 | 98,24 | 158,08 | 1.972,83 | 45,02 | 42,31 | Batch 1 · parity backend (15 lớp) |
| 238 | Backend | uint4-cal19 · fp16-packed | 5 | 66,75 | 69,57 | 89,41 | 96,75 | 160,01 | 1.973,86 | 45,02 | 38,38 | Batch 1 · parity backend (15 lớp) |
| 239 | Backend | robust-coarse15 · fp16-decoded | 5 | 67,40 | 69,03 | 88,88 | 96,48 | 191,64 | 1.996,64 | 47,82 | 40,96 | Batch 1 · parity backend (15 lớp) |
| 240 | Backend | robust-coarse15 · fp32-decoded | 5 | 67,40 | 69,03 | 88,88 | 96,48 | 191,64 | 1.792,98 | 69,59 | 40,31 | Batch 1 · parity backend (15 lớp) |
| 241 | Backend | robust-coarse15 · fp32-packed | 5 | 67,40 | 69,03 | 88,88 | 96,48 | 191,64 | 1.771,15 | 66,94 | 40,06 | Batch 1 · parity backend (15 lớp) |
| 242 | Backend | uint4-cal19 · fp16-decoded | 5 | 66,75 | 69,57 | 89,41 | 96,75 | 160,01 | 1.994,86 | 46,46 | 39,03 | Batch 1 · parity backend (15 lớp) |
| 243 | Backend | uint4-cal19 · fp32-decoded | 5 | 66,75 | 69,57 | 89,41 | 96,74 | 160,01 | 1.790,27 | 68,23 | 36,78 | Batch 1 · parity backend (15 lớp) |
| 244 | Backend | uint4-cal19 · fp32-packed | 5 | 66,75 | 69,57 | 89,41 | 96,74 | 160,01 | 1.768,76 | 66,79 | 38,56 | Batch 1 · parity backend (15 lớp) |
| 245 | Hybrid DINOv2 | dino-deep | 5 | 70,66 | 74,58 | 93,29 | 98,97 | 164,25 | 1.593,75 | 84,87 | 38,60 | Batch 1 · encoder đóng gói (15 lớp) |
| 246 | Hybrid DINOv2 | dino-mid | 5 | 71,71 | 76,40 | 93,32 | 98,36 | 159,15 | 1.591,46 | 84,87 | 43,00 | Batch 1 (15 lớp) |
| 247 | Hybrid DINOv2 | dino-two | 5 | 73,34 | 77,39 | 94,45 | 99,04 | 323,54 | 1.591,95 | 87,26 | 45,18 | Batch 1 (15 lớp) |
| 248 | Hybrid DINOv2 | dual-fusion25 | 5 | 72,80 | 73,22 | 90,25 | 98,24 | 330,42 | 1.971,51 | 67,86 | 45,43 | Batch 1 (15 lớp) |
| 249 | Hybrid DINOv2 | hybrid-agreement | 5 | 78,03 | 79,89 | 95,00 | 99,38 | 655,49 | 2.041,70 | 114,47 | 51,12 | Batch 1 (15 lớp) |
| 250 | Hybrid DINOv2 | hybrid-compact-deep | 5 | 78,08 | 79,27 | 94,24 | 98,97 | 337,83 | 1.770,88 | 112,17 | 62,47 | Batch 1 · encoder đóng gói (15 lớp) |
| 251 | Hybrid DINOv2 | hybrid-compact-mid | 5 | 78,44 | 79,46 | 94,00 | 98,36 | 332,73 | 1.771,90 | 112,17 | 47,99 | Batch 1 · encoder đóng gói (15 lớp) |
| 252 | Hybrid DINOv2 | hybrid-mid45 | 5 | 78,11 | 79,92 | 94,55 | 99,08 | 492,77 | 2.069,50 | 112,38 | 62,51 | Batch 1 (15 lớp) |
| 253 | Hybrid DINOv2 | hybrid-wide | 5 | 77,72 | 79,24 | 94,49 | 99,41 | 497,88 | 2.069,65 | 112,38 | 50,83 | Batch 1 (15 lớp) |
| 254 | Hybrid DINOv2 | hybrid25 | 5 | 77,93 | 79,06 | 93,94 | 99,41 | 497,86 | 1.797,70 | 112,38 | 56,21 | Batch 1 · encoder đóng gói (15 lớp) |
| 255 | Hybrid DINOv2 | hybrid45 | 5 | 77,19 | 79,06 | 94,62 | 99,41 | 497,86 | 1.797,02 | 112,38 | 64,19 | Batch 1 (15 lớp) |
| 256 | Hybrid DINOv2 | wrn-gpu-replay | 5 | 72,80 | 73,22 | 90,25 | 98,24 | 330,42 | 2.041,48 | 67,86 | 38,88 | Batch 1 (15 lớp) |
| 257 | Hybrid DINOv2 | wrn-spatial-wide | 5 | 73,18 | 74,28 | 91,01 | 98,21 | 333,88 | 2.041,18 | 67,90 | 46,50 | Batch 1 (15 lớp) |
| 258 | Hybrid DINOv2 | wrn-spatial | 5 | 73,40 | 73,77 | 90,43 | 98,24 | 333,47 | 2.040,67 | 67,89 | 44,42 | Batch 1 (15 lớp) |
| 259 | Hybrid DINOv2 | normal-adaptive | 5 | 75,33 | 77,15 | 93,69 | 99,32 | 337,28 | 1.592,50 | 87,26 | 44,72 | Batch 1 (15 lớp) |
| 260 | Ablation Hybrid Mid | block12-only | 5 | 77,13 | 78,50 | 93,79 | 98,85 | 338,15 | 1.768,41 | 112,17 | 56,21 | Batch 1 · encoder đóng gói (15 lớp) |
| 261 | Ablation Hybrid Mid | block6-only | 5 | 77,34 | 78,40 | 93,40 | 97,18 | 329,23 | 1.769,85 | 112,17 | 64,02 | Batch 1 · encoder đóng gói (15 lớp) |
| 262 | Ablation Hybrid Mid | fp16-bank | 5 | 78,44 | 79,48 | 94,00 | 98,36 | 877,65 | 1.772,09 | 112,17 | 62,66 | Batch 1 · encoder đóng gói (15 lớp) |
| 263 | Ablation Hybrid Mid | hybrid-compact-mid | 5 | 78,44 | 79,46 | 94,00 | 98,36 | 332,73 | 1.769,12 | 112,17 | 49,93 | Batch 1 · encoder đóng gói (15 lớp) |
| 264 | Ablation Hybrid Mid | without-dino-contrast | 5 | 78,08 | 79,57 | 94,31 | 98,36 | 332,73 | 1.770,34 | 112,17 | 58,20 | Batch 1 · encoder đóng gói (15 lớp) |
| 265 | Ablation Hybrid Mid | without-dino-map | 5 | 71,82 | 71,14 | 88,59 | 98,36 | 331,29 | 1.769,72 | 112,16 | 58,69 | Batch 1 · encoder đóng gói (15 lớp) |
| 266 | Ablation Hybrid Mid | without-dino | 5 | 71,82 | 71,14 | 88,59 | 93,19 | 173,83 | 1.883,34 | 68,29 | 40,68 | Batch 1 · encoder đóng gói (15 lớp) |
| 267 | Ablation Hybrid Mid | without-gaussian | 5 | 78,43 | 78,36 | 93,05 | 98,36 | 332,73 | 1.698,75 | 112,17 | 57,27 | Batch 1 · encoder đóng gói (15 lớp) |
| 268 | Ablation Hybrid Mid | without-spatial-spread | 5 | 78,39 | 79,40 | 94,00 | 98,36 | 331,30 | 1.770,15 | 112,16 | 59,03 | Batch 1 · encoder đóng gói (15 lớp) |
| 269 | Ablation Hybrid Mid | without-spatial | 5 | 78,42 | 79,40 | 93,90 | 98,36 | 330,22 | 1.763,95 | 112,14 | 62,05 | Batch 1 · encoder đóng gói (15 lớp) |
| 270 | Ablation Hybrid Mid | without-wrn-contrast | 5 | 77,42 | 79,51 | 94,60 | 98,36 | 332,69 | 1.770,60 | 112,17 | 58,99 | Batch 1 · encoder đóng gói (15 lớp) |
| 271 | Ablation Hybrid Mid | without-wrn | 5 | 71,65 | 76,38 | 93,36 | 98,36 | 159,16 | 1.593,34 | 84,87 | 36,51 | Batch 1 · encoder đóng gói (15 lớp) |

## References

- **PatchCore Original** dùng mã Amazon commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`, WideResNet50-2 V1, layer2+layer3, embedding 1.024 chiều, approximate greedy coreset và FAISS CPU. Hai tỷ lệ 10% và 1% tính trên mọi patch của tập build đã khóa. FP32 10% trong nhóm nén bank lấy 819 vector từ pool 8.192 và dùng embedding 1.536 chiều. [PatchCore Original](https://github.com/amazon-science/patchcore-inspection)
- **Lite PatchCore** dùng mã tác giả Unified PCA32, coreset 10% từng lớp và một bank chung. Adapter gọi từng bước trích đặc trưng, lấy coreset, PCA và FAISS của tác giả, kể cả khi batch cuối có một ảnh. [Lite PatchCore](https://doi.org/10.5220/0014253200004084)
- **Anomalib** dùng model PatchCore 2.7.0, WideResNet50-2 V1, 9 láng giềng để tính điểm ảnh và coreset 10%. Adapter giữ embedding gốc trên RAM, chiếu từng khối lên CUDA và dùng bình phương L2 trong k-center. Kiểm tra phép tính k-center trên CPU với 700 patch cho dãy coreset trùng với mã gốc. Tiền xử lý và mask lấy từ giao thức nhóm. [Anomalib PatchCore](https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/models/image/patchcore.html)
- **Mahalanobis Streaming** dùng checkpoint seed 42 của tác giả, với IPCA 99% phương sai, shrinkage 0,07, k-center merge-reduce và bank 1.000. Đã đối chiếu SHA-256 của 5.354 ảnh giữa hai bản dataset, sau đó chạy lại suy luận và đo bộ nhớ trên CPU theo manifest chung. Checkpoint được fit trên toàn bộ train normal gốc. Các lần fit mới dùng tập build 80%. [Mahalanobis PatchCore](https://arxiv.org/abs/2605.27748)
- **SoftPatch** dùng mã tác giả, LOF, ngưỡng 0,15 và soft weight, chạy với ảnh build sạch. **Lightweight PatchCore** gọi mã tác giả với MobileNetV3-Small, features.3 và features.8, D=128 và coreset 10%. [SoftPatch](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch), [Lightweight PatchCore](https://github.com/augustinus2000/lightweight-patchcore)
- **AnomalyDINO** dùng full-shot, DINOv2-S/14 block cuối, cosine exact, hàm top-1% và hàm tạo heatmap của tác giả. Adapter dùng checkpoint timm cục bộ, resize 448 rồi crop trung tâm 392, tra cứu theo block CUDA và tắt foreground masking. Evaluator nhận heatmap 224×224. [AnomalyDINO](https://github.com/dammsi/AnomalyDINO)
- **ASO PatchCore** triển khai quy tắc chuẩn hóa và argmin của bài báo trên thứ tự coreset FP32 của nhóm, WideResNet50-2 và pool 8.192 patch. **PatchCore-Lite PQ** dùng MoViAD, MobileNetV2, bank 10.000 vector và shortlist 1.000. Lượt chấm điểm tái sử dụng bank đã giải mã, với sai số trong ngưỡng `1e-5` trên ảnh kiểm tra so với đường gốc. Lượt đo tài nguyên gọi đường suy luận gốc. [ASO PatchCore](https://www.scitepress.org/Papers/2026/144451/144451.pdf), [PatchCore-Lite](https://arxiv.org/abs/2603.20288), [MoViAD](https://github.com/AMCO-UniPD/moviad)

