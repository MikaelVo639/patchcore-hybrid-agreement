# Phụ lục cấu hình benchmark PatchCore

Số thứ tự khớp 271 dòng trong [báo cáo tổng hợp](BAO_CAO_TONG_HOP.md). Seed trong bảng là các seed có đủ 15 lớp. Cột lượt chấm tính từng cặp lớp–seed đã lưu, kể cả seed chỉ chạy một phần. Cột lượt đo đếm các dòng tài nguyên của cùng cấu hình trong CSV.

Dùng `group` và `id` để ghép bảng tổng hợp. Trong chi tiết chất lượng, `id` tương ứng với `method`. Chi tiết tài nguyên có `group` và `config` dùng mã cấu hình. `source` trong chi tiết chất lượng tính từ thư mục `patchcore-compression`.

[Chỉ mục cấu hình](results/all-configurations.csv) · [Lượt chấm từng lớp và seed](results/all-quality.csv) · [Lượt đo tài nguyên](results/all-resources.csv)

## Thiết lập chung

MVTec AD dùng split seed 0, 2.903 ảnh build, 726 ảnh calibration và 1.725 ảnh test trên 15 lớp. Input WRN thường resize cạnh ngắn 256 rồi center crop 224, chuẩn hóa ImageNet. DINOv2 dùng resize 448 rồi crop 392. Evaluator dùng mask 224×224, vùng liên thông 8 hướng và cùng danh sách ảnh test. Cột `memory_scope` của CSV tài nguyên ghi phạm vi từng lượt đo. Profile tham chiếu mới dùng batch 1, 20 ảnh mỗi lớp sau warm-up 5 ảnh. Peak VRAM lấy từ PyTorch allocated, peak RAM là RSS tiến trình.

## Tham số của từng cấu hình

### Đối chứng

upstream_baseline.py (nguồn tham chiếu trong workspace thực nghiệm) · reference_runs.py (nguồn tham chiếu trong workspace thực nghiệm) · [mvtec_streaming_mh_patchcore.yaml](../../mahalanobis-patchcore/configs/mvtec/baselines/mvtec_streaming_mh_patchcore.yaml)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 1 | `PatchCore Original · Amazon FP32 10%`<br>Mã `amazon-fp32-10pct` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, FAISS CPU 1-NN (4 luồng), approximate greedy 10% toàn bộ patch build, max patch, σ=4 |
| 2 | `PatchCore Original · Amazon FP32 1%`<br>Mã `amazon-fp32-1pct` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, FAISS CPU 1-NN (4 luồng), approximate greedy 1% toàn bộ patch build, max patch, σ=4 |
| 3 | `Anomalib PatchCore 2.7.0`<br>Mã `anomalib-original` | 0 | 15 | 15 | Anomalib 2.7.0, WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, k-center 10%, 1-NN Euclid cấp patch, 9 láng giềng để tính điểm ảnh, σ=4 |
| 4 | `Lite PatchCore tác giả · Unified PCA32`<br>Mã `lite-pca32-original` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, FAISS CPU 1-NN (4 luồng), coreset 10% mỗi lớp, PCA32 và một bank Unified cho 15 lớp |
| 5 | `Mahalanobis PatchCore tác giả · Streaming`<br>Mã `mh-streaming-original` | 42 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, IPCA giữ 99% phương sai, Cholesky shrinkage 0,07, merge-reduce k-center 1000, chunk coreset 256, k=1, checkpoint seed 42 fit toàn bộ train normal, suy luận CPU |
| 6 | `SoftPatch tác giả · LOF, ảnh build sạch`<br>Mã `softpatch-original` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, FAISS CPU 1-NN (4 luồng), coreset 10% sau lọc LOF k=6 ngưỡng 0,15, soft weight, build normal sạch |
| 7 | `Lightweight PatchCore tác giả · MobileNetV3-Small`<br>Mã `mobilenet-original` | 0 | 15 | 15 | MobileNetV3-Small V1, features.3 + features.8, embedding D=128, patch 3, approximate greedy 10%, FAISS CPU 1-NN (4 luồng) |
| 8 | `AnomalyDINO · Full-shot S/14`<br>Mã `anomalydino-original` | 0 | 15 | 15 | DINOv2-S/14 FP32 block cuối, resize 448 rồi crop 392, full-shot toàn bộ build, cosine exact, mean top-1%, foreground mask tắt, heatmap tác giả |
| 9 | `ASO PatchCore · công thức bài báo`<br>Mã `aso-paper` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, FP32, pool 8192, chọn prefix từ coreset 10% theo argmin hai trục chuẩn hóa, 1-NN L2, max patch, σ=4 |
| 10 | `PatchCore-Lite · Product Quantization`<br>Mã `pq-paper` | 0 | 15 | 15 | MoViAD, MobileNetV2 V1 features.4/7/10, bank 10000, product quantization, shortlist 1000, 1-NN sau giải mã, max patch. Chấm điểm cache bank đã giải mã, profile dùng đường gốc |

### Allocation v1

[config.json](../../../PatchCore/results/bottle_resnet18_pilot_seed0/config.json) · pilots.py (nguồn tham chiếu trong workspace thực nghiệm) · [selection.py](../../../PatchCore/src/patchcore_budget/selection.py)

ResNet18 V1, layer2+layer3, D=384, lưới 28×28, L2 1-NN và max patch. `dimensions=[16, 32, 64, 128]`, `fixed_dimension=64`, `variance_threshold=0.95`, `reconstruction_weight=0.5`, `whitening_regularization=0.01`, `max_fit_patches=4096`, `max_bank_pool=2048`, `max_calibration_patches=4096`, `sampler=coreset`, `seed=0`, `map_smoothing_sigma=4.0`.

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 11 | `raw · 512 KiB` | 0 | 15 | 15 | Selector raw, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 12 | `fixed · 512 KiB` | 0 | 15 | 15 | Selector fixed, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 13 | `variance · 512 KiB` | 0 | 15 | 15 | Selector variance, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 14 | `joint · 512 KiB` | 0 | 15 | 15 | Selector joint, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 15 | `reconstruction · 512 KiB` | 0 | 15 | 15 | Selector reconstruction, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 16 | `coverage · 512 KiB` | 0 | 15 | 15 | Selector coverage, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 17 | `whitened_joint · 512 KiB` | 0 | 15 | 15 | Selector whitened_joint, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 18 | `raw · 1024 KiB` | 0 | 15 | 15 | Selector raw, trần 1024 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 19 | `fixed · 1024 KiB` | 0 | 15 | 15 | Selector fixed, trần 1024 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 20 | `variance · 1024 KiB` | 0 | 15 | 15 | Selector variance, trần 1024 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 21 | `joint · 1024 KiB` | 0 | 15 | 15 | Selector joint, trần 1024 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 22 | `reconstruction · 1024 KiB` | 0 | 15 | 15 | Selector reconstruction, trần 1024 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 23 | `coverage · 1024 KiB` | 0 | 15 | 15 | Selector coverage, trần 1024 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 24 | `whitened_joint · 1024 KiB` | 0 | 15 | 15 | Selector whitened_joint, trần 1024 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 25 | `raw · 4096 KiB` | 0 | 15 | 15 | Selector raw, trần 4096 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 26 | `fixed · 4096 KiB` | 0 | 15 | 15 | Selector fixed, trần 4096 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 27 | `variance · 4096 KiB` | 0 | 15 | 15 | Selector variance, trần 4096 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 28 | `joint · 4096 KiB` | 0 | 15 | 15 | Selector joint, trần 4096 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 29 | `reconstruction · 4096 KiB` | 0 | 15 | 15 | Selector reconstruction, trần 4096 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 30 | `coverage · 4096 KiB` | 0 | 15 | 15 | Selector coverage, trần 4096 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 31 | `whitened_joint · 4096 KiB` | 0 | 15 | 15 | Selector whitened_joint, trần 4096 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |

### Allocation v2

[mvtec_resnet18_pilot_v2.json](../../../PatchCore/configs/mvtec_resnet18_pilot_v2.json) · pilots.py (nguồn tham chiếu trong workspace thực nghiệm) · [selection.py](../../../PatchCore/src/patchcore_budget/selection.py)

ResNet18 V1, layer2+layer3, D=384, lưới 28×28, L2 1-NN và max patch. `dimensions=[16, 32, 64, 128]`, `fixed_dimension=64`, `variance_threshold=0.95`, `reconstruction_weight=0.5`, `whitening_regularization=0.01`, `max_fit_patches=4096`, `max_bank_pool=8192`, `max_calibration_patches=4096`, `sampler=coreset`, `seed=0`, `map_smoothing_sigma=4.0`, `include_raw_in_joint=False`, `tie_tolerance=0.0`.

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 32 | `raw · 256 KiB` | 0 | 17 | 17 | Selector raw, trần 256 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 33 | `fixed · 256 KiB` | 0 | 17 | 17 | Selector fixed, trần 256 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 34 | `variance · 256 KiB` | 0 | 17 | 17 | Selector variance, trần 256 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 35 | `joint · 256 KiB` | 0 | 17 | 17 | Selector joint, trần 256 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 36 | `reconstruction · 256 KiB` | 0 | 17 | 17 | Selector reconstruction, trần 256 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 37 | `coverage · 256 KiB` | 0 | 17 | 17 | Selector coverage, trần 256 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 38 | `whitened_joint · 256 KiB` | 0 | 17 | 17 | Selector whitened_joint, trần 256 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 39 | `raw · 512 KiB` | 0 | 17 | 17 | Selector raw, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 40 | `fixed · 512 KiB` | 0 | 17 | 17 | Selector fixed, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 41 | `variance · 512 KiB` | 0 | 17 | 17 | Selector variance, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 42 | `joint · 512 KiB` | 0 | 17 | 17 | Selector joint, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 43 | `reconstruction · 512 KiB` | 0 | 17 | 17 | Selector reconstruction, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 44 | `coverage · 512 KiB` | 0 | 17 | 17 | Selector coverage, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 45 | `whitened_joint · 512 KiB` | 0 | 17 | 17 | Selector whitened_joint, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |

### Baseline PCA

pilots.py (nguồn tham chiếu trong workspace thực nghiệm) · [run_pilot.py](../../../Baseline/scripts/run_pilot.py)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 46 | `raw · 1 MiB` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, D=1024, không PCA, bank FP32, approximate greedy theo trần 1 MiB, squared L2 1-NN, max patch, σ=4 |
| 47 | `raw · 2 MiB` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, D=1024, không PCA, bank FP32, approximate greedy theo trần 2 MiB, squared L2 1-NN, max patch, σ=4 |
| 48 | `raw · 4 MiB` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, D=1024, không PCA, bank FP32, approximate greedy theo trần 4 MiB, squared L2 1-NN, max patch, σ=4 |
| 49 | `raw · 8 MiB` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, D=1024, không PCA, bank FP32, approximate greedy theo trần 8 MiB, squared L2 1-NN, max patch, σ=4 |
| 50 | `pca128 · 1 MiB` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, PCA128 covariance float64 trên build, chiếu FP32, bank FP32, approximate greedy theo trần 1 MiB, squared L2 1-NN, max patch, σ=4 |
| 51 | `pca128 · 2 MiB` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, PCA128 covariance float64 trên build, chiếu FP32, bank FP32, approximate greedy theo trần 2 MiB, squared L2 1-NN, max patch, σ=4 |
| 52 | `pca128 · 4 MiB` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, PCA128 covariance float64 trên build, chiếu FP32, bank FP32, approximate greedy theo trần 4 MiB, squared L2 1-NN, max patch, σ=4 |
| 53 | `pca128 · 8 MiB` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, embedding D=1024, patch 3, PCA128 covariance float64 trên build, chiếu FP32, bank FP32, approximate greedy theo trần 8 MiB, squared L2 1-NN, max patch, σ=4 |

### Allocation v2

[mvtec_resnet18_pilot_v2.json](../../../PatchCore/configs/mvtec_resnet18_pilot_v2.json) · pilots.py (nguồn tham chiếu trong workspace thực nghiệm) · [selection.py](../../../PatchCore/src/patchcore_budget/selection.py)

ResNet18 V1, layer2+layer3, D=384, lưới 28×28, L2 1-NN và max patch. `dimensions=[16, 32, 64, 128]`, `fixed_dimension=64`, `variance_threshold=0.95`, `reconstruction_weight=0.5`, `whitening_regularization=0.01`, `max_fit_patches=4096`, `max_bank_pool=8192`, `max_calibration_patches=4096`, `sampler=coreset`, `seed=0`, `map_smoothing_sigma=4.0`, `include_raw_in_joint=False`, `tie_tolerance=0.0`.

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 54 | `Ứng viên PCA theo ngưỡng phương sai 95% · 256 KiB` | 0 | 17 | 17 | Selector Ứng viên PCA theo ngưỡng phương sai 95%, trần 256 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 55 | `Ứng viên PCA theo ngưỡng phương sai 95% · 512 KiB` | 0 | 17 | 17 | Selector Ứng viên PCA theo ngưỡng phương sai 95%, trần 512 KiB. Xem D và số vector thực tế theo lớp ở phần dưới |
| 56 | `Ứng viên pca_d128__b262144` | 0 | 17 | 17 | Ứng viên cố định pca, D=128, trần 256 KiB |
| 57 | `Ứng viên pca_d128__b524288` | 0 | 17 | 17 | Ứng viên cố định pca, D=128, trần 512 KiB |
| 58 | `Ứng viên pca_d16__b262144` | 0 | 17 | 17 | Ứng viên cố định pca, D=16, trần 256 KiB |
| 59 | `Ứng viên pca_d16__b524288` | 0 | 17 | 17 | Ứng viên cố định pca, D=16, trần 512 KiB |
| 60 | `Ứng viên pca_d32__b262144` | 0 | 17 | 17 | Ứng viên cố định pca, D=32, trần 256 KiB |
| 61 | `Ứng viên pca_d32__b524288` | 0 | 17 | 17 | Ứng viên cố định pca, D=32, trần 512 KiB |
| 62 | `Ứng viên pca_d64__b262144` | 0 | 17 | 17 | Ứng viên cố định pca, D=64, trần 256 KiB |
| 63 | `Ứng viên pca_d64__b524288` | 0 | 17 | 17 | Ứng viên cố định pca, D=64, trần 512 KiB |
| 64 | `Ứng viên raw_d384__b262144` | 0 | 17 | 17 | Ứng viên cố định raw, D=384, trần 256 KiB |
| 65 | `Ứng viên raw_d384__b524288` | 0 | 17 | 17 | Ứng viên cố định raw, D=384, trần 512 KiB |
| 66 | `Ứng viên whitened_d128__b262144` | 0 | 17 | 17 | Ứng viên cố định whitened, D=128, trần 256 KiB |
| 67 | `Ứng viên whitened_d128__b524288` | 0 | 17 | 17 | Ứng viên cố định whitened, D=128, trần 512 KiB |
| 68 | `Ứng viên whitened_d16__b262144` | 0 | 17 | 17 | Ứng viên cố định whitened, D=16, trần 256 KiB |
| 69 | `Ứng viên whitened_d16__b524288` | 0 | 17 | 17 | Ứng viên cố định whitened, D=16, trần 512 KiB |
| 70 | `Ứng viên whitened_d32__b262144` | 0 | 17 | 17 | Ứng viên cố định whitened, D=32, trần 256 KiB |
| 71 | `Ứng viên whitened_d32__b524288` | 0 | 17 | 17 | Ứng viên cố định whitened, D=32, trần 512 KiB |
| 72 | `Ứng viên whitened_d64__b262144` | 0 | 17 | 17 | Ứng viên cố định whitened, D=64, trần 256 KiB |
| 73 | `Ứng viên whitened_d64__b524288` | 0 | 17 | 17 | Ứng viên cố định whitened, D=64, trần 512 KiB |

### Nén bank

[benchmark.py](../src/core/benchmark.py) · [compression.py](../src/core/compression.py) · extraction.json (nguồn tham chiếu trong workspace thực nghiệm)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 74 | `fp32-10pct` | 0 1 2 3 4 | 75 | 99 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo 10% pool (819 vector), FP32, L2 1-NN, max patch, σ=4 |
| 75 | `fp32-budget · trần 512 KiB` | 0 1 2 3 4 | 75 | 99 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 512 KiB, FP32, L2 1-NN, max patch, σ=4 |
| 76 | `fp16-full · trần 512 KiB` | 0 | 15 | 18 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 512 KiB, FP16, L2 1-NN, max patch, σ=4 |
| 77 | `int8-full · trần 512 KiB` | 0 1 2 3 4 | 75 | 93 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 512 KiB, INT8, L2 1-NN, max patch, σ=4 |
| 78 | `pca64-int8 · trần 512 KiB` | 0 | 15 | 18 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 512 KiB, PCA64, INT8, L2 1-NN, max patch, σ=4 |
| 79 | `pca128-int8 · trần 512 KiB` | 0 1 2 3 4 | 75 | 93 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 512 KiB, PCA128, INT8, L2 1-NN, max patch, σ=4 |
| 80 | `pca64-int8-residual · trần 512 KiB` | 0 1 2 3 4 | 75 | 93 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 512 KiB, PCA64, INT8, giữ norm phần dư, L2 1-NN, max patch, σ=4 |
| 81 | `pca128-int8-residual · trần 512 KiB` | 0 1 2 3 4 | 75 | 93 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 512 KiB, PCA128, INT8, giữ norm phần dư, L2 1-NN, max patch, σ=4 |
| 82 | `fp32-budget · trần 1024 KiB` | 0 1 2 3 4 | 75 | 99 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 1024 KiB, FP32, L2 1-NN, max patch, σ=4 |
| 83 | `fp16-full · trần 1024 KiB` | 0 | 15 | 18 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 1024 KiB, FP16, L2 1-NN, max patch, σ=4 |
| 84 | `int8-full · trần 1024 KiB` | 0 1 2 3 4 | 75 | 93 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 1024 KiB, INT8, L2 1-NN, max patch, σ=4 |
| 85 | `pca64-int8 · trần 1024 KiB` | 0 | 15 | 18 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 1024 KiB, PCA64, INT8, L2 1-NN, max patch, σ=4 |
| 86 | `pca128-int8 · trần 1024 KiB` | 0 1 2 3 4 | 75 | 93 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 1024 KiB, PCA128, INT8, L2 1-NN, max patch, σ=4 |
| 87 | `pca64-int8-residual · trần 1024 KiB` | 0 1 2 3 4 | 75 | 93 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 1024 KiB, PCA64, INT8, giữ norm phần dư, L2 1-NN, max patch, σ=4 |
| 88 | `pca128-int8-residual · trần 1024 KiB` | 0 1 2 3 4 | 75 | 93 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 1024 KiB, PCA128, INT8, giữ norm phần dư, L2 1-NN, max patch, σ=4 |
| 89 | `int4-full · trần 512 KiB` | 0 1 2 3 4 | 75 | 84 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 512 KiB, INT4, L2 1-NN, max patch, σ=4 |
| 90 | `int4-full · trần 1024 KiB` | 0 1 2 3 4 | 75 | 84 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 1024 KiB, INT4, L2 1-NN, max patch, σ=4 |
| 91 | `uint4-full · trần 512 KiB` | 0 1 2 3 4 | 75 | 78 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 512 KiB, UINT4, L2 1-NN, max patch, σ=4 |
| 92 | `uint4-full · trần 1024 KiB` | 0 1 2 3 4 | 75 | 78 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, greedy theo trần 1024 KiB, UINT4, L2 1-NN, max patch, σ=4 |

### Điểm patch

[score_tuning.py](../src/core/score_tuning.py)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 93 | `baseline` | 0 1 2 3 4 | 75 | 93 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric l2, k=1, chuẩn hóa none, σ=4, max patch cấp ảnh |
| 94 | `position` | 0 | 15 | 18 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric l2, k=1, chuẩn hóa position, σ=4, max patch cấp ảnh |
| 95 | `position-center` | 0 | 15 | 18 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric l2, k=1, chuẩn hóa position-center, σ=4, max patch cấp ảnh |
| 96 | `position-mean` | 0 | 15 | 18 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric l2, k=1, chuẩn hóa position-mean, σ=4, max patch cấp ảnh |
| 97 | `image-center` | 0 | 15 | 18 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric l2, k=1, chuẩn hóa image-center, σ=4, max patch cấp ảnh |
| 98 | `position-k3` | 0 | 15 | 18 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric l2, k=3, chuẩn hóa position, σ=4, max patch cấp ảnh |
| 99 | `position-sigma2` | 0 | 15 | 18 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric l2, k=1, chuẩn hóa position, σ=2, max patch cấp ảnh |
| 100 | `position-layer2` | 0 1 2 3 4 | 75 | 93 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric layer2, k=1, chuẩn hóa position, σ=4, max patch cấp ảnh |
| 101 | `layer2` | 0 | 15 | 18 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric layer2, k=1, chuẩn hóa none, σ=4, max patch cấp ảnh |
| 102 | `position-layer-mix` | 0 | 15 | 18 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric layer-mix, k=1, chuẩn hóa position, σ=4, max patch cấp ảnh |
| 103 | `position-cosine` | 0 | 15 | 18 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric cosine, k=1, chuẩn hóa position, σ=4, max patch cấp ảnh |
| 104 | `cosine` | 0 | 15 | 18 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric cosine, k=1, chuẩn hóa none, σ=4, max patch cấp ảnh |
| 105 | `local-contrast` | 0 1 2 3 4 | 75 | 93 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric l2, k=1, chuẩn hóa local-contrast, σ=4, max patch cấp ảnh, lấy điểm trước contrast |
| 106 | `position-contrast` | 0 1 2 3 4 | 75 | 93 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric l2, k=1, chuẩn hóa position-contrast, σ=4, max patch cấp ảnh, lấy điểm trước contrast |
| 107 | `layer2-contrast` | 0 1 2 3 4 | 75 | 93 | Bank INT4 từ pool WRN 8192, trần 1 MiB, metric layer2, k=1, chuẩn hóa local-contrast, σ=4, max patch cấp ảnh, lấy điểm trước contrast |

### Ablation INT4 ban đầu

run_ablation.py (nguồn tham chiếu trong workspace thực nghiệm)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 108 | `fp32-reference` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, FP32, 819 vector, L2 1-NN, max patch, σ=4, contrast tắt |
| 109 | `fp32-budget` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, FP32, trần 1 MiB, L2 1-NN, max patch, σ=4, contrast tắt |
| 110 | `fp32-matched` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, FP32, khớp vector INT4 full, L2 1-NN, max patch, σ=4, contrast tắt |
| 111 | `int4-full` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, INT4 có dấu, cùng chỉ số coreset gốc, L2 1-NN, max patch, σ=4, contrast tắt |
| 112 | `int4-full-contrast` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, INT4 có dấu, cùng chỉ số coreset gốc, L2 1-NN, max patch, σ=4, contrast 7×7 @ 0,75 |
| 113 | `int4-layer2` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, cùng chỉ số coreset gốc, L2 1-NN, max patch, σ=4, contrast tắt |
| 114 | `int4-layer2-contrast` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, cùng chỉ số coreset gốc, L2 1-NN, max patch, σ=4, contrast 7×7 @ 0,75 |

### Codec lossless

[lossless_bank.py](../src/core/lossless_bank.py) · compression.json (nguồn tham chiếu trong workspace thực nghiệm)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 115 | `original` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec original, L2 1-NN, max patch, contrast 7×7 @ 0,75, σ=4 |
| 116 | `deflate-int4` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec deflate-int4, L2 1-NN, max patch, contrast 7×7 @ 0,75, σ=4 |
| 117 | `lzma-int4` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-int4, L2 1-NN, max patch, contrast 7×7 @ 0,75, σ=4 |
| 118 | `deflate-planes3` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec deflate-planes3, L2 1-NN, max patch, contrast 7×7 @ 0,75, σ=4 |
| 119 | `lzma-planes3` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 1-NN, max patch, contrast 7×7 @ 0,75, σ=4 |

### INT4 Fast

[inference_experiment.py](../src/core/inference_experiment.py) · [fast_inference.py](../src/core/fast_inference.py) · scoring.json (nguồn tham chiếu trong workspace thực nghiệm) · legacy_tiled.py (nguồn tham chiếu trong workspace thực nghiệm)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 120 | `Layer2 + contrast 7×7` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0,75, σ=4, pipeline Fast, scorer decoded |
| 121 | `Layer2 7×7 · sequential` | 0 1 2 3 4 | 75 | 9 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0,75, σ=4, scorer previous, 0 worker prefetch |
| 122 | `Layer2 7×7 · cached` | 0 1 2 3 4 | 75 | 9 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0,75, σ=4, scorer decoded, 0 worker prefetch |
| 123 | `Layer2 7×7 · prefetch2` | 0 1 2 3 4 | 75 | 9 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0,75, σ=4, scorer decoded, 2 worker prefetch |
| 124 | `Layer2 7×7 · prefetch2-norms` | 0 1 2 3 4 | 75 | 9 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0,75, σ=4, scorer norms, 2 worker prefetch |
| 125 | `Scorer · tiled` | 0 1 2 3 4 | 75 | 6 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0,75, σ=4, scorer tiled |
| 126 | `Scorer · previous` | 0 1 2 3 4 | 75 | 6 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0,75, σ=4, scorer previous |
| 127 | `Scorer · norms` | 0 1 2 3 4 | 75 | 6 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0,75, σ=4, scorer norms |
| 128 | `Scorer · decoded` | 0 1 2 3 4 | 75 | 6 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0,75, σ=4, scorer decoded |

### INT4 Balanced

search_balanced_maps.py (nguồn tham chiếu trong workspace thực nghiệm) · map_config.json (nguồn tham chiếu trong workspace thực nghiệm)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 129 | `w=7 strength=0.75 σ=4` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0.75, σ=4 |
| 130 | `w=7 strength=0 σ=4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0, σ=4 |
| 131 | `w=7 strength=0.4 σ=4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0.4, σ=4 |
| 132 | `w=7 strength=0.55 σ=4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0.55, σ=4 |
| 133 | `w=7 strength=0.65 σ=4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0.65, σ=4 |
| 134 | `w=5 strength=0.55 σ=2` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 5×5 @ 0.55, σ=2 |
| 135 | `w=7 strength=0.55 σ=2` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 7×7 @ 0.55, σ=2 |
| 136 | `w=9 strength=0.55 σ=2` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 9×9 @ 0.55, σ=2 |
| 137 | `w=11 strength=0.55 σ=4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 11×11 @ 0.55, σ=4 |
| 138 | `w=11 strength=0.65 σ=2` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 11×11 @ 0.65, σ=2 |
| 139 | `w=11 strength=0.75 σ=4` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 11×11 @ 0.75, σ=4 |
| 140 | `w=15 strength=0.75 σ=2` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 15×15 @ 0.75, σ=2 |
| 141 | `w=15 strength=0.75 σ=4` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, max patch, contrast 15×15 @ 0.75, σ=4 |

### INT4 Robust

search_robust_maps.py (nguồn tham chiếu trong workspace thực nghiệm) · [robust_maps.py](../src/core/robust_maps.py) · map_config.json (nguồn tham chiếu trong workspace thực nghiệm)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 142 | `standard:11:0.0 · top-1` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, không clip, điểm ảnh top-1 |
| 143 | `standard:11:0.0 · top-4` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, không clip, điểm ảnh top-4 |
| 144 | `standard:11:0.0 · top-16` | 0 1 2 3 4 | 75 | 12 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, không clip, điểm ảnh top-16 |
| 145 | `standard:11:0.0 · top-32` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, không clip, điểm ảnh top-32 |
| 146 | `standard:15:0.0 · top-1` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, không clip, điểm ảnh top-1 |
| 147 | `standard:15:0.0 · top-4` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, không clip, điểm ảnh top-4 |
| 148 | `standard:15:0.0 · top-16` | 0 1 2 3 4 | 75 | 12 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, không clip, điểm ảnh top-16 |
| 149 | `standard:15:0.0 · top-32` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, không clip, điểm ảnh top-32 |
| 150 | `standard:19:0.0 · top-1` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0,75, σ=4, không clip, điểm ảnh top-1 |
| 151 | `standard:19:0.0 · top-4` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0,75, σ=4, không clip, điểm ảnh top-4 |
| 152 | `standard:19:0.0 · top-16` | 0 1 2 3 4 | 75 | 12 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0,75, σ=4, không clip, điểm ảnh top-16 |
| 153 | `standard:19:0.0 · top-32` | 0 1 2 3 4 | 75 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0,75, σ=4, không clip, điểm ảnh top-32 |
| 154 | `standard:23:0.0 · top-1` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 23×23 @ 0,75, σ=4, không clip, điểm ảnh top-1 |
| 155 | `standard:23:0.0 · top-4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 23×23 @ 0,75, σ=4, không clip, điểm ảnh top-4 |
| 156 | `standard:23:0.0 · top-16` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 23×23 @ 0,75, σ=4, không clip, điểm ảnh top-16 |
| 157 | `standard:23:0.0 · top-32` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 23×23 @ 0,75, σ=4, không clip, điểm ảnh top-32 |
| 158 | `before:11:0.85 · top-1` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.85 từ normal calibration, điểm ảnh top-1 |
| 159 | `before:11:0.85 · top-4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.85 từ normal calibration, điểm ảnh top-4 |
| 160 | `before:11:0.85 · top-16` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.85 từ normal calibration, điểm ảnh top-16 |
| 161 | `before:11:0.85 · top-32` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.85 từ normal calibration, điểm ảnh top-32 |
| 162 | `before:11:0.95 · top-1` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.95 từ normal calibration, điểm ảnh top-1 |
| 163 | `before:11:0.95 · top-4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.95 từ normal calibration, điểm ảnh top-4 |
| 164 | `before:11:0.95 · top-16` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.95 từ normal calibration, điểm ảnh top-16 |
| 165 | `before:11:0.95 · top-32` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.95 từ normal calibration, điểm ảnh top-32 |
| 166 | `before:11:0.99 · top-1` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.99 từ normal calibration, điểm ảnh top-1 |
| 167 | `before:11:0.99 · top-4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.99 từ normal calibration, điểm ảnh top-4 |
| 168 | `before:11:0.99 · top-16` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.99 từ normal calibration, điểm ảnh top-16 |
| 169 | `before:11:0.99 · top-32` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip before, q=0.99 từ normal calibration, điểm ảnh top-32 |
| 170 | `after:11:0.85 · top-1` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.85 từ normal calibration, điểm ảnh top-1 |
| 171 | `after:11:0.85 · top-4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.85 từ normal calibration, điểm ảnh top-4 |
| 172 | `after:11:0.85 · top-16` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.85 từ normal calibration, điểm ảnh top-16 |
| 173 | `after:11:0.85 · top-32` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.85 từ normal calibration, điểm ảnh top-32 |
| 174 | `after:11:0.95 · top-1` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.95 từ normal calibration, điểm ảnh top-1 |
| 175 | `after:11:0.95 · top-4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.95 từ normal calibration, điểm ảnh top-4 |
| 176 | `after:11:0.95 · top-16` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.95 từ normal calibration, điểm ảnh top-16 |
| 177 | `after:11:0.95 · top-32` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.95 từ normal calibration, điểm ảnh top-32 |
| 178 | `after:11:0.99 · top-1` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.99 từ normal calibration, điểm ảnh top-1 |
| 179 | `after:11:0.99 · top-4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.99 từ normal calibration, điểm ảnh top-4 |
| 180 | `after:11:0.99 · top-16` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.99 từ normal calibration, điểm ảnh top-16 |
| 181 | `after:11:0.99 · top-32` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0,75, σ=4, clip after, q=0.99 từ normal calibration, điểm ảnh top-32 |
| 182 | `before:15:0.95 · top-1` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, clip before, q=0.95 từ normal calibration, điểm ảnh top-1 |
| 183 | `before:15:0.95 · top-4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, clip before, q=0.95 từ normal calibration, điểm ảnh top-4 |
| 184 | `before:15:0.95 · top-16` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, clip before, q=0.95 từ normal calibration, điểm ảnh top-16 |
| 185 | `before:15:0.95 · top-32` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, clip before, q=0.95 từ normal calibration, điểm ảnh top-32 |
| 186 | `after:15:0.95 · top-1` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, clip after, q=0.95 từ normal calibration, điểm ảnh top-1 |
| 187 | `after:15:0.95 · top-4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, clip after, q=0.95 từ normal calibration, điểm ảnh top-4 |
| 188 | `after:15:0.95 · top-16` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, clip after, q=0.95 từ normal calibration, điểm ảnh top-16 |
| 189 | `after:15:0.95 · top-32` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0,75, σ=4, clip after, q=0.95 từ normal calibration, điểm ảnh top-32 |
| 190 | `after:19:0.99 · top-1` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0,75, σ=4, clip after, q=0.99 từ normal calibration, điểm ảnh top-1 |
| 191 | `after:19:0.99 · top-4` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0,75, σ=4, clip after, q=0.99 từ normal calibration, điểm ảnh top-4 |
| 192 | `after:19:0.99 · top-16` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0,75, σ=4, clip after, q=0.99 từ normal calibration, điểm ảnh top-16 |
| 193 | `after:19:0.99 · top-32` | 0 | 15 | 3 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0,75, σ=4, clip after, q=0.99 từ normal calibration, điểm ảnh top-32 |

### Benchmark rộng

[experiment.py](../src/comparison/experiment.py) · early_features.py (nguồn tham chiếu trong workspace thực nghiệm) · [dual_head.py](../src/comparison/dual_head.py) · [deployment.py](../src/comparison/deployment.py)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 194 | `aso-fp32` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, pool 8192, bank theo elbow, 32-bit, contrast tắt, σ=4, positional calibration tắt, điểm ảnh top-1 |
| 195 | `dual-dense` | 0 1 2 3 4 | 75 | 15 | WRN hai nhánh layer1-layer2-wide (map) + uint4-32k-cal (điểm ảnh), trọng số map coarse=0 sau chuẩn hóa bằng độ lệch chuẩn normal, điểm ảnh top-16 nhánh coarse |
| 196 | `dual-fusion25` | 0 1 2 3 4 | 75 | 15 | WRN hai nhánh layer1-layer2-wide (map) + uint4-32k-cal (điểm ảnh), trọng số map coarse=0.25 sau chuẩn hóa bằng độ lệch chuẩn normal, điểm ảnh top-16 nhánh coarse |
| 197 | `dual-fusion50` | 0 1 2 3 4 | 75 | 15 | WRN hai nhánh layer1-layer2-wide (map) + uint4-32k-cal (điểm ảnh), trọng số map coarse=0.5 sau chuẩn hóa bằng độ lệch chuẩn normal, điểm ảnh top-16 nhánh coarse |
| 198 | `fp32-10pct` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28, FP32, 819 vector từ pool 8192, contrast tắt, σ=4, positional calibration tắt, điểm ảnh top-1 |
| 199 | `fusion-centroids` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28 (layer3 ×0,35), pool 8192, bank 740, tinh chỉnh centroid 8 vòng, 4-bit, PCA64, covariance power=0, contrast 15×15 @ 0.75, σ=2, positional calibration tắt, điểm ảnh top-16 |
| 200 | `fusion-pca64-wide` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28 (layer3 ×0,35), pool 8192, bank 740, 4-bit, PCA64, covariance power=0, contrast 21×21 @ 0.75, σ=3, positional calibration tắt, điểm ảnh top-16 |
| 201 | `fusion-pca64` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28 (layer3 ×0,35), pool 8192, bank 740, 4-bit, PCA64, covariance power=0, contrast 15×15 @ 0.75, σ=2, positional calibration tắt, điểm ảnh top-16 |
| 202 | `fusion-spatial` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28 (layer3 ×0,35), pool 8192, bank 740, 4-bit, PCA64, covariance power=0.25, thêm 2 tọa độ patch, contrast 15×15 @ 0.75, σ=2, positional calibration tắt, điểm ảnh top-16 |
| 203 | `layer1-dense` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer1, D=256, lưới 56×56, pool 8192, greedy 740, UINT4, contrast 27×27 @ 0.75, σ=2, positional calibration tắt, điểm ảnh top-32 |
| 204 | `layer1-layer2-wide` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer1+layer2 (gộp cặp kênh layer2 ×0,75), D=512, lưới 56×56, pool 8192, greedy 740, UINT4, contrast 37×37 @ 0.75, σ=3, positional calibration tắt, điểm ảnh top-32 |
| 205 | `layer1-layer2` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer1+layer2 (gộp cặp kênh layer2 ×0,75), D=512, lưới 56×56, pool 8192, greedy 740, UINT4, contrast 27×27 @ 0.75, σ=2, positional calibration tắt, điểm ảnh top-32 |
| 206 | `layer2-pca96` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, 4-bit, PCA96, covariance power=0, contrast 15×15 @ 0.75, σ=2, positional calibration tắt, điểm ảnh top-16 |
| 207 | `lite-pca32` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28 (layer3 ×0,35), pool 8192, bank 740, 16-bit, PCA32, covariance power=0, contrast tắt, σ=4, positional calibration tắt, điểm ảnh top-16 |
| 208 | `mh-pca64` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2+layer3, D=1536, lưới 28×28 (layer3 ×0,35), pool 8192, bank 740, 8-bit, PCA64, covariance power=0.5, contrast tắt, σ=4, positional calibration tắt, điểm ảnh top-16 |
| 209 | `robust-11` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0.75, σ=4, positional calibration tắt, điểm ảnh top-16 |
| 210 | `robust-19` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0.75, σ=4, positional calibration tắt, điểm ảnh top-16 |
| 211 | `robust-cal` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0.75, σ=2, positional calibration 0.5 trên lưới 28×28, điểm ảnh top-16 |
| 212 | `robust-cal11` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0.75, σ=3, positional calibration 0.5 trên lưới 28×28, điểm ảnh top-16 |
| 213 | `robust-cal19` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 28×28, điểm ảnh top-16 |
| 214 | `robust-cal21` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 21×21 @ 0.75, σ=4, positional calibration 0.75 trên lưới 28×28, điểm ảnh top-16 |
| 215 | `robust-cal27` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 27×27 @ 0.75, σ=4, positional calibration 0.25 trên lưới 28×28, điểm ảnh top-16 |
| 216 | `robust-coarse11` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 11×11 @ 0.75, σ=3, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16 |
| 217 | `robust-coarse15-k24` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0.75, σ=3, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-24 |
| 218 | `robust-coarse15-k32` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0.75, σ=3, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-32 |
| 219 | `robust-coarse15` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0.75, σ=3, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16 |
| 220 | `robust-coarse19-k24` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-24 |
| 221 | `robust-coarse19-k32` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-32 |
| 222 | `robust-coarse19` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16 |
| 223 | `robust-sharp` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0.75, σ=2, positional calibration tắt, điểm ảnh top-16 |
| 224 | `robust-strong` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 17×17 @ 0.9, σ=3, positional calibration tắt, điểm ảnh top-16 |
| 225 | `soft-int4` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740 sau lọc điểm yếu theo vị trí q95, 4-bit, contrast 15×15 @ 0.75, σ=2, positional calibration tắt, điểm ảnh top-16 |
| 226 | `uint4-32k-cal` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 32768, bank 740, UINT4, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16 |
| 227 | `uint4-32k` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 32768, bank 740, UINT4, contrast 19×19 @ 0.75, σ=3, positional calibration tắt, điểm ảnh top-16 |
| 228 | `uint4-384` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 384, UINT4, contrast 15×15 @ 0.75, σ=2, positional calibration tắt, điểm ảnh top-16 |
| 229 | `uint4-cal11` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, UINT4, contrast 11×11 @ 0.75, σ=3, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16 |
| 230 | `uint4-cal19` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, UINT4, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16 |
| 231 | `uint4-cal27` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, UINT4, contrast 27×27 @ 0.75, σ=4, positional calibration 0.25 trên lưới 14×14, điểm ảnh top-16 |
| 232 | `uint4-pos` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, UINT4, thêm 2 tọa độ patch, contrast 19×19 @ 0.75, σ=3, positional calibration tắt, điểm ảnh top-16 |
| 233 | `uint4-strong` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, UINT4, contrast 19×19 @ 0.75, σ=3, positional calibration tắt, điểm ảnh top-16 |
| 234 | `uint4` | 0 | 15 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, UINT4, contrast 15×15 @ 0.75, σ=2, positional calibration tắt, điểm ảnh top-16 |

### Backend

[deployment.py](../src/comparison/deployment.py) · [experiment.py](../src/comparison/experiment.py)

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 235 | `robust-coarse15 · fp16-packed` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0.75, σ=3, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16. Encoder FP16, bank runtime packed |
| 236 | `robust-coarse19 · fp16-packed` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16. Encoder FP16, bank runtime packed |
| 237 | `uint4-32k-cal · fp16-packed` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 32768, bank 740, UINT4, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16. Encoder FP16, bank runtime packed |
| 238 | `uint4-cal19 · fp16-packed` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, UINT4, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16. Encoder FP16, bank runtime packed |
| 239 | `robust-coarse15 · fp16-decoded` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0.75, σ=3, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16. Encoder FP16, bank runtime decoded |
| 240 | `robust-coarse15 · fp32-decoded` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0.75, σ=3, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16. Encoder FP32, bank runtime decoded |
| 241 | `robust-coarse15 · fp32-packed` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, INT4 có dấu, codec lzma-planes3, L2 exact, contrast 15×15 @ 0.75, σ=3, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16. Encoder FP32, bank runtime packed |
| 242 | `uint4-cal19 · fp16-decoded` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, UINT4, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16. Encoder FP16, bank runtime decoded |
| 243 | `uint4-cal19 · fp32-decoded` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, UINT4, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16. Encoder FP32, bank runtime decoded |
| 244 | `uint4-cal19 · fp32-packed` | 0 1 2 3 4 | 75 | 15 | WideResNet50-2 ImageNet V1 layer2, D=512, lưới 28×28, pool 8192, bank 740, UINT4, contrast 19×19 @ 0.75, σ=3.5, positional calibration 0.5 trên lưới 14×14, điểm ảnh top-16. Encoder FP32, bank runtime packed |

### Hybrid DINOv2

[advanced.py](../src/advanced/advanced.py) · [raw_scores.py](../src/advanced/raw_scores.py) · select_on_normal.py (nguồn tham chiếu trong workspace thực nghiệm) · normal-selection.json (nguồn tham chiếu trong workspace thực nghiệm)

Nhánh fine dùng `layer1-layer2-wide` và lưới 56×56. Nhánh coarse dùng `uint4-32k-cal` và lưới 28×28. Hai nhánh WRN dùng encoder FP16 trong lượt trích raw score và lấy trung bình top-16 cấp ảnh. Hai nhánh DINOv2-S/14 dùng encoder FP16, embedding D=384, pool 16384, farthest-first 1024, bank INT4 LZMA và cosine exact. `mid` trộn block 6/12 theo 0,70/0,30, `deep` theo 0,25/0,75. Cả hai lấy trung bình top-10 cấp ảnh. Gaussian σ lần lượt là 3/3,5/4/4 cho fine/coarse/mid/deep. Calibration normal dùng lưới 14×14 từ q50 và q90, nội suy về lưới từng nhánh. Các ablation ghi giá trị sau khi áp dụng thay đổi, kể cả trường hợp chỉ thay bank DINO.

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 245 | `dino-deep` | 0 1 2 3 4 | 75 | 30 | map deep×1, ảnh deep×1, spatial=0.3, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 246 | `dino-mid` | 0 1 2 3 4 | 75 | 15 | map mid×1, ảnh mid×1, spatial=0.3, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 247 | `dino-two` | 0 1 2 3 4 | 75 | 15 | map mid×0.6 + deep×0.4, ảnh mid×0.35 + deep×0.65, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 248 | `dual-fusion25` | 0 1 2 3 4 | 75 | 15 | WRN hai nhánh layer1-layer2-wide (map) + uint4-32k-cal (điểm ảnh), trọng số map coarse=0.25 sau chuẩn hóa bằng độ lệch chuẩn normal, điểm ảnh top-16 nhánh coarse |
| 249 | `hybrid-agreement` | 0 1 2 3 4 | 75 | 15 | map fine×0.45 + mid×0.25 + deep×0.3, ảnh coarse×0.25 + mid×0.25 + deep×0.5, spatial=0.6, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 250 | `hybrid-compact-deep` | 0 1 2 3 4 | 75 | 30 | map fine×0.6 + deep×0.4, ảnh deep×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 251 | `hybrid-compact-mid` | 0 1 2 3 4 | 75 | 30 | map fine×0.6 + mid×0.4, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 252 | `hybrid-mid45` | 0 1 2 3 4 | 75 | 15 | map fine×0.4 + coarse×0.15 + mid×0.45, ảnh coarse×0.35 + mid×0.65, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 253 | `hybrid-wide` | 0 1 2 3 4 | 75 | 15 | map fine×0.4 + coarse×0.2 + deep×0.4, ảnh coarse×0.35 + deep×0.65, spatial=0.65, fine contrast 47×47 @ 0.75, DINO contrast 19×19 @ 0.55, spatial spread bật, Gaussian bật |
| 254 | `hybrid25` | 0 1 2 3 4 | 75 | 30 | map fine×0.55 + coarse×0.2 + deep×0.25, ảnh coarse×0.35 + deep×0.65, spatial=0.25, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 255 | `hybrid45` | 0 1 2 3 4 | 75 | 15 | map fine×0.4 + coarse×0.15 + deep×0.45, ảnh coarse×0.35 + deep×0.65, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 256 | `wrn-gpu-replay` | 0 1 2 3 4 | 75 | 15 | WRN hai nhánh layer1-layer2-wide (map) + uint4-32k-cal (điểm ảnh), trọng số map coarse=0.25 sau chuẩn hóa bằng độ lệch chuẩn normal, điểm ảnh top-16 nhánh coarse, map replay CUDA |
| 257 | `wrn-spatial-wide` | 0 1 2 3 4 | 75 | 15 | map fine×0.65 + coarse×0.35, ảnh coarse×0.9 + fine×0.1, spatial=0.7, fine contrast 47×47 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 258 | `wrn-spatial` | 0 1 2 3 4 | 75 | 15 | map fine×0.75 + coarse×0.25, ảnh coarse×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 259 | `normal-adaptive` | 0 1 2 3 4 | 75 | 15 | Chọn recipe theo lớp bằng tối đa 12 ảnh calibration normal, proxy seed 203 với scratch/texture/color, objective 0,55/0,25/0,20 cho PRO nhỏ/PRO@30/I-AUROC, ưu tiên bank nhỏ khi hòa |

### Ablation Hybrid Mid

ablation.py (nguồn tham chiếu trong workspace thực nghiệm) · ablation_banks.py (nguồn tham chiếu trong workspace thực nghiệm) · [advanced.py](../src/advanced/advanced.py)

Nhánh fine dùng `layer1-layer2-wide` và lưới 56×56. Nhánh coarse dùng `uint4-32k-cal` và lưới 28×28. Hai nhánh WRN dùng encoder FP16 trong lượt trích raw score và lấy trung bình top-16 cấp ảnh. Hai nhánh DINOv2-S/14 dùng encoder FP16, embedding D=384, pool 16384, farthest-first 1024, bank INT4 LZMA và cosine exact. `mid` trộn block 6/12 theo 0,70/0,30, `deep` theo 0,25/0,75. Cả hai lấy trung bình top-10 cấp ảnh. Gaussian σ lần lượt là 3/3,5/4/4 cho fine/coarse/mid/deep. Calibration normal dùng lưới 14×14 từ q50 và q90, nội suy về lưới từng nhánh. Các ablation ghi giá trị sau khi áp dụng thay đổi, kể cả trường hợp chỉ thay bank DINO.

| # | Cấu hình | Seed đầy đủ | Lượt chấm | Lượt đo | Thiết lập |
|---:|---|---|---:|---:|---|
| 260 | `block12-only` | 0 1 2 3 4 | 75 | 15 | map fine×0.6 + mid×0.4, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật, token DINO block 12 |
| 261 | `block6-only` | 0 1 2 3 4 | 75 | 15 | map fine×0.6 + mid×0.4, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật, token DINO block 6 |
| 262 | `fp16-bank` | 0 1 2 3 4 | 75 | 15 | map fine×0.6 + mid×0.4, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật, DINO bank FP16 |
| 263 | `hybrid-compact-mid` | 0 1 2 3 4 | 75 | 15 | map fine×0.6 + mid×0.4, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 264 | `without-dino-contrast` | 0 1 2 3 4 | 75 | 15 | map fine×0.6 + mid×0.4, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0, spatial spread bật, Gaussian bật |
| 265 | `without-dino-map` | 0 1 2 3 4 | 75 | 15 | map fine×1, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 266 | `without-dino` | 0 1 2 3 4 | 75 | 15 | map fine×1, ảnh fine×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 267 | `without-gaussian` | 0 1 2 3 4 | 75 | 15 | map fine×0.6 + mid×0.4, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian tắt |
| 268 | `without-spatial-spread` | 0 1 2 3 4 | 75 | 15 | map fine×0.6 + mid×0.4, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread tắt, Gaussian bật |
| 269 | `without-spatial` | 0 1 2 3 4 | 75 | 15 | map fine×0.6 + mid×0.4, ảnh mid×1, spatial=0, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 270 | `without-wrn-contrast` | 0 1 2 3 4 | 75 | 15 | map fine×0.6 + mid×0.4, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |
| 271 | `without-wrn` | 0 1 2 3 4 | 75 | 15 | map mid×1, ảnh mid×1, spatial=0.4, fine contrast 37×37 @ 0.75, DINO contrast 19×19 @ 0.3, spatial spread bật, Gaussian bật |

## Recipe đã chọn của `normal-adaptive`

| Lớp | Recipe |
|---|---|
| bottle | `dino-two` |
| cable | `dino-deep` |
| capsule | `hybrid-agreement` |
| carpet | `dino-mid` |
| grid | `dino-two` |
| hazelnut | `dino-deep` |
| leather | `dino-mid` |
| metal_nut | `dino-two` |
| pill | `dino-two` |
| screw | `hybrid45` |
| tile | `dino-two` |
| toothbrush | `dino-two` |
| transistor | `hybrid25` |
| wood | `hybrid-agreement` |
| zipper | `dino-deep` |

## Kích thước bank thực tế của từng lượt pilot

Mỗi ô ghi `lớp/seed = D × số vector`. Các giá trị lấy từ file bank hoặc metadata triển khai đã lưu.

| # | Cấu hình | D × số vector theo lớp và seed |
|---:|---|---|
| 11 | `raw · 512 KiB` | bottle/0 = 384×340<br>cable/0 = 384×340<br>capsule/0 = 384×340<br>carpet/0 = 384×340<br>grid/0 = 384×340<br>hazelnut/0 = 384×340<br>leather/0 = 384×340<br>metal_nut/0 = 384×340<br>pill/0 = 384×340<br>screw/0 = 384×340<br>tile/0 = 384×340<br>toothbrush/0 = 384×340<br>transistor/0 = 384×340<br>wood/0 = 384×340<br>zipper/0 = 384×340 |
| 12 | `fixed · 512 KiB` | bottle/0 = 64×1654<br>cable/0 = 64×1654<br>capsule/0 = 64×1654<br>carpet/0 = 64×1654<br>grid/0 = 64×1654<br>hazelnut/0 = 64×1654<br>leather/0 = 64×1654<br>metal_nut/0 = 64×1654<br>pill/0 = 64×1654<br>screw/0 = 64×1654<br>tile/0 = 64×1654<br>toothbrush/0 = 64×1654<br>transistor/0 = 64×1654<br>wood/0 = 64×1654<br>zipper/0 = 64×1654 |
| 13 | `variance · 512 KiB` | bottle/0 = 51×2048<br>cable/0 = 142×534<br>capsule/0 = 56×1945<br>carpet/0 = 113×770<br>grid/0 = 82×1206<br>hazelnut/0 = 137×568<br>leather/0 = 103×882<br>metal_nut/0 = 124×667<br>pill/0 = 94×1003<br>screw/0 = 117×730<br>tile/0 = 153×468<br>toothbrush/0 = 74×1378<br>transistor/0 = 125×659<br>wood/0 = 141×541<br>zipper/0 = 65×1622 |
| 14 | `joint · 512 KiB` | bottle/0 = 64×1654<br>cable/0 = 64×1654<br>capsule/0 = 64×1654<br>carpet/0 = 16×2048<br>grid/0 = 32×2048<br>hazelnut/0 = 64×1654<br>leather/0 = 16×2048<br>metal_nut/0 = 64×1654<br>pill/0 = 64×1654<br>screw/0 = 64×1654<br>tile/0 = 16×2048<br>toothbrush/0 = 64×1654<br>transistor/0 = 64×1654<br>wood/0 = 16×2048<br>zipper/0 = 32×2048 |
| 15 | `reconstruction · 512 KiB` | bottle/0 = 128×635<br>cable/0 = 128×635<br>capsule/0 = 128×635<br>carpet/0 = 128×635<br>grid/0 = 128×635<br>hazelnut/0 = 128×635<br>leather/0 = 128×635<br>metal_nut/0 = 128×635<br>pill/0 = 128×635<br>screw/0 = 128×635<br>tile/0 = 128×635<br>toothbrush/0 = 128×635<br>transistor/0 = 128×635<br>wood/0 = 128×635<br>zipper/0 = 128×635 |
| 16 | `coverage · 512 KiB` | bottle/0 = 16×2048<br>cable/0 = 16×2048<br>capsule/0 = 16×2048<br>carpet/0 = 16×2048<br>grid/0 = 16×2048<br>hazelnut/0 = 16×2048<br>leather/0 = 16×2048<br>metal_nut/0 = 16×2048<br>pill/0 = 16×2048<br>screw/0 = 16×2048<br>tile/0 = 16×2048<br>toothbrush/0 = 16×2048<br>transistor/0 = 16×2048<br>wood/0 = 16×2048<br>zipper/0 = 16×2048 |
| 17 | `whitened_joint · 512 KiB` | bottle/0 = 64×1654<br>cable/0 = 32×2048<br>capsule/0 = 64×1654<br>carpet/0 = 16×2048<br>grid/0 = 16×2048<br>hazelnut/0 = 32×2048<br>leather/0 = 16×2048<br>metal_nut/0 = 64×1654<br>pill/0 = 32×2048<br>screw/0 = 64×1654<br>tile/0 = 16×2048<br>toothbrush/0 = 64×1654<br>transistor/0 = 32×2048<br>wood/0 = 16×2048<br>zipper/0 = 32×2048 |
| 18 | `raw · 1024 KiB` | bottle/0 = 384×682<br>cable/0 = 384×682<br>capsule/0 = 384×682<br>carpet/0 = 384×682<br>grid/0 = 384×682<br>hazelnut/0 = 384×682<br>leather/0 = 384×682<br>metal_nut/0 = 384×682<br>pill/0 = 384×682<br>screw/0 = 384×682<br>tile/0 = 384×682<br>toothbrush/0 = 384×682<br>transistor/0 = 384×682<br>wood/0 = 384×682<br>zipper/0 = 384×682 |
| 19 | `fixed · 1024 KiB` | bottle/0 = 64×2048<br>cable/0 = 64×2048<br>capsule/0 = 64×2048<br>carpet/0 = 64×2048<br>grid/0 = 64×2048<br>hazelnut/0 = 64×2048<br>leather/0 = 64×2048<br>metal_nut/0 = 64×2048<br>pill/0 = 64×2048<br>screw/0 = 64×2048<br>tile/0 = 64×2048<br>toothbrush/0 = 64×2048<br>transistor/0 = 64×2048<br>wood/0 = 64×2048<br>zipper/0 = 64×2048 |
| 20 | `variance · 1024 KiB` | bottle/0 = 51×2048<br>cable/0 = 142×1457<br>capsule/0 = 56×2048<br>carpet/0 = 113×1930<br>grid/0 = 82×2048<br>hazelnut/0 = 137×1524<br>leather/0 = 103×2048<br>metal_nut/0 = 124×1724<br>pill/0 = 94×2048<br>screw/0 = 117×1851<br>tile/0 = 153×1325<br>toothbrush/0 = 74×2048<br>transistor/0 = 125×1708<br>wood/0 = 141×1470<br>zipper/0 = 65×2048 |
| 21 | `joint · 1024 KiB` | bottle/0 = 64×2048<br>cable/0 = 64×2048<br>capsule/0 = 128×1659<br>carpet/0 = 16×2048<br>grid/0 = 32×2048<br>hazelnut/0 = 64×2048<br>leather/0 = 16×2048<br>metal_nut/0 = 64×2048<br>pill/0 = 64×2048<br>screw/0 = 64×2048<br>tile/0 = 16×2048<br>toothbrush/0 = 64×2048<br>transistor/0 = 64×2048<br>wood/0 = 16×2048<br>zipper/0 = 32×2048 |
| 22 | `reconstruction · 1024 KiB` | bottle/0 = 128×1659<br>cable/0 = 128×1659<br>capsule/0 = 128×1659<br>carpet/0 = 128×1659<br>grid/0 = 128×1659<br>hazelnut/0 = 128×1659<br>leather/0 = 128×1659<br>metal_nut/0 = 128×1659<br>pill/0 = 128×1659<br>screw/0 = 128×1659<br>tile/0 = 128×1659<br>toothbrush/0 = 128×1659<br>transistor/0 = 128×1659<br>wood/0 = 128×1659<br>zipper/0 = 128×1659 |
| 23 | `coverage · 1024 KiB` | bottle/0 = 16×2048<br>cable/0 = 16×2048<br>capsule/0 = 16×2048<br>carpet/0 = 16×2048<br>grid/0 = 16×2048<br>hazelnut/0 = 16×2048<br>leather/0 = 16×2048<br>metal_nut/0 = 16×2048<br>pill/0 = 16×2048<br>screw/0 = 16×2048<br>tile/0 = 16×2048<br>toothbrush/0 = 16×2048<br>transistor/0 = 16×2048<br>wood/0 = 16×2048<br>zipper/0 = 16×2048 |
| 24 | `whitened_joint · 1024 KiB` | bottle/0 = 64×2048<br>cable/0 = 32×2048<br>capsule/0 = 64×2048<br>carpet/0 = 16×2048<br>grid/0 = 16×2048<br>hazelnut/0 = 32×2048<br>leather/0 = 16×2048<br>metal_nut/0 = 64×2048<br>pill/0 = 32×2048<br>screw/0 = 64×2048<br>tile/0 = 16×2048<br>toothbrush/0 = 64×2048<br>transistor/0 = 64×2048<br>wood/0 = 16×2048<br>zipper/0 = 32×2048 |
| 25 | `raw · 4096 KiB` | bottle/0 = 384×2048<br>cable/0 = 384×2048<br>capsule/0 = 384×2048<br>carpet/0 = 384×2048<br>grid/0 = 384×2048<br>hazelnut/0 = 384×2048<br>leather/0 = 384×2048<br>metal_nut/0 = 384×2048<br>pill/0 = 384×2048<br>screw/0 = 384×2048<br>tile/0 = 384×2048<br>toothbrush/0 = 384×2048<br>transistor/0 = 384×2048<br>wood/0 = 384×2048<br>zipper/0 = 384×2048 |
| 26 | `fixed · 4096 KiB` | bottle/0 = 64×2048<br>cable/0 = 64×2048<br>capsule/0 = 64×2048<br>carpet/0 = 64×2048<br>grid/0 = 64×2048<br>hazelnut/0 = 64×2048<br>leather/0 = 64×2048<br>metal_nut/0 = 64×2048<br>pill/0 = 64×2048<br>screw/0 = 64×2048<br>tile/0 = 64×2048<br>toothbrush/0 = 64×2048<br>transistor/0 = 64×2048<br>wood/0 = 64×2048<br>zipper/0 = 64×2048 |
| 27 | `variance · 4096 KiB` | bottle/0 = 51×2048<br>cable/0 = 142×2048<br>capsule/0 = 56×2048<br>carpet/0 = 113×2048<br>grid/0 = 82×2048<br>hazelnut/0 = 137×2048<br>leather/0 = 103×2048<br>metal_nut/0 = 124×2048<br>pill/0 = 94×2048<br>screw/0 = 117×2048<br>tile/0 = 153×2048<br>toothbrush/0 = 74×2048<br>transistor/0 = 125×2048<br>wood/0 = 141×2048<br>zipper/0 = 65×2048 |
| 28 | `joint · 4096 KiB` | bottle/0 = 128×2048<br>cable/0 = 64×2048<br>capsule/0 = 128×2048<br>carpet/0 = 16×2048<br>grid/0 = 32×2048<br>hazelnut/0 = 64×2048<br>leather/0 = 16×2048<br>metal_nut/0 = 128×2048<br>pill/0 = 64×2048<br>screw/0 = 128×2048<br>tile/0 = 16×2048<br>toothbrush/0 = 64×2048<br>transistor/0 = 64×2048<br>wood/0 = 16×2048<br>zipper/0 = 32×2048 |
| 29 | `reconstruction · 4096 KiB` | bottle/0 = 128×2048<br>cable/0 = 128×2048<br>capsule/0 = 128×2048<br>carpet/0 = 128×2048<br>grid/0 = 128×2048<br>hazelnut/0 = 128×2048<br>leather/0 = 128×2048<br>metal_nut/0 = 128×2048<br>pill/0 = 128×2048<br>screw/0 = 128×2048<br>tile/0 = 128×2048<br>toothbrush/0 = 128×2048<br>transistor/0 = 128×2048<br>wood/0 = 128×2048<br>zipper/0 = 128×2048 |
| 30 | `coverage · 4096 KiB` | bottle/0 = 16×2048<br>cable/0 = 16×2048<br>capsule/0 = 16×2048<br>carpet/0 = 16×2048<br>grid/0 = 16×2048<br>hazelnut/0 = 16×2048<br>leather/0 = 16×2048<br>metal_nut/0 = 16×2048<br>pill/0 = 16×2048<br>screw/0 = 16×2048<br>tile/0 = 16×2048<br>toothbrush/0 = 16×2048<br>transistor/0 = 16×2048<br>wood/0 = 16×2048<br>zipper/0 = 16×2048 |
| 31 | `whitened_joint · 4096 KiB` | bottle/0 = 64×2048<br>cable/0 = 32×2048<br>capsule/0 = 64×2048<br>carpet/0 = 16×2048<br>grid/0 = 16×2048<br>hazelnut/0 = 32×2048<br>leather/0 = 16×2048<br>metal_nut/0 = 64×2048<br>pill/0 = 32×2048<br>screw/0 = 64×2048<br>tile/0 = 16×2048<br>toothbrush/0 = 64×2048<br>transistor/0 = 64×2048<br>wood/0 = 16×2048<br>zipper/0 = 32×2048 |
| 32 | `raw · 256 KiB` | bottle/0 = 384×170<br>cable/0 = 384×170<br>capsule/0 = 384×170<br>carpet/0 = 384×170<br>grid/0 = 384×170<br>hazelnut/0 = 384×170<br>leather/0 = 384×170<br>metal_nut/0 = 384×170<br>pill/0 = 384×170<br>screw/0 = 384×170<br>tile/0 = 384×170<br>toothbrush/0 = 384×170<br>transistor/0 = 384×170<br>wood/0 = 384×170<br>zipper/0 = 384×170<br>bottle/1 = 384×170<br>bottle/2 = 384×170 |
| 33 | `fixed · 256 KiB` | bottle/0 = 64×630<br>cable/0 = 64×630<br>capsule/0 = 64×630<br>carpet/0 = 64×630<br>grid/0 = 64×630<br>hazelnut/0 = 64×630<br>leather/0 = 64×630<br>metal_nut/0 = 64×630<br>pill/0 = 64×630<br>screw/0 = 64×630<br>tile/0 = 64×630<br>toothbrush/0 = 64×630<br>transistor/0 = 64×630<br>wood/0 = 64×630<br>zipper/0 = 64×630<br>bottle/1 = 64×630<br>bottle/2 = 64×630 |
| 34 | `variance · 256 KiB` | bottle/0 = 51×888<br>cable/0 = 142×73<br>capsule/0 = 56×774<br>carpet/0 = 113×190<br>grid/0 = 82×407<br>hazelnut/0 = 137×89<br>leather/0 = 103×246<br>metal_nut/0 = 124×139<br>pill/0 = 94×306<br>screw/0 = 117×170<br>tile/0 = 153×40<br>toothbrush/0 = 74×492<br>transistor/0 = 125×135<br>wood/0 = 141×76<br>zipper/0 = 65×614<br>bottle/1 = 51×888<br>bottle/2 = 50×913 |
| 35 | `joint · 256 KiB` | bottle/0 = 32×1644<br>cable/0 = 32×1644<br>capsule/0 = 32×1644<br>carpet/0 = 16×3672<br>grid/0 = 16×3672<br>hazelnut/0 = 32×1644<br>leather/0 = 16×3672<br>metal_nut/0 = 32×1644<br>pill/0 = 32×1644<br>screw/0 = 32×1644<br>tile/0 = 16×3672<br>toothbrush/0 = 32×1644<br>transistor/0 = 32×1644<br>wood/0 = 16×3672<br>zipper/0 = 32×1644<br>bottle/1 = 32×1644<br>bottle/2 = 32×1644 |
| 36 | `reconstruction · 256 KiB` | bottle/0 = 128×123<br>cable/0 = 128×123<br>capsule/0 = 128×123<br>carpet/0 = 128×123<br>grid/0 = 128×123<br>hazelnut/0 = 128×123<br>leather/0 = 128×123<br>metal_nut/0 = 128×123<br>pill/0 = 128×123<br>screw/0 = 128×123<br>tile/0 = 128×123<br>toothbrush/0 = 128×123<br>transistor/0 = 128×123<br>wood/0 = 128×123<br>zipper/0 = 128×123<br>bottle/1 = 128×123<br>bottle/2 = 128×123 |
| 37 | `coverage · 256 KiB` | bottle/0 = 16×3672<br>cable/0 = 16×3672<br>capsule/0 = 16×3672<br>carpet/0 = 16×3672<br>grid/0 = 16×3672<br>hazelnut/0 = 16×3672<br>leather/0 = 16×3672<br>metal_nut/0 = 16×3672<br>pill/0 = 16×3672<br>screw/0 = 16×3672<br>tile/0 = 16×3672<br>toothbrush/0 = 16×3672<br>transistor/0 = 16×3672<br>wood/0 = 16×3672<br>zipper/0 = 16×3672<br>bottle/1 = 16×3672<br>bottle/2 = 16×3672 |
| 38 | `whitened_joint · 256 KiB` | bottle/0 = 32×1644<br>cable/0 = 32×1644<br>capsule/0 = 32×1644<br>carpet/0 = 16×3672<br>grid/0 = 16×3672<br>hazelnut/0 = 16×3672<br>leather/0 = 16×3672<br>metal_nut/0 = 32×1644<br>pill/0 = 32×1644<br>screw/0 = 32×1644<br>tile/0 = 16×3672<br>toothbrush/0 = 32×1644<br>transistor/0 = 32×1644<br>wood/0 = 16×3672<br>zipper/0 = 16×3672<br>bottle/1 = 32×1644<br>bottle/2 = 32×1644 |
| 39 | `raw · 512 KiB` | bottle/0 = 384×340<br>cable/0 = 384×340<br>capsule/0 = 384×340<br>carpet/0 = 384×340<br>grid/0 = 384×340<br>hazelnut/0 = 384×340<br>leather/0 = 384×340<br>metal_nut/0 = 384×340<br>pill/0 = 384×340<br>screw/0 = 384×340<br>tile/0 = 384×340<br>toothbrush/0 = 384×340<br>transistor/0 = 384×340<br>wood/0 = 384×340<br>zipper/0 = 384×340<br>bottle/1 = 384×340<br>bottle/2 = 384×340 |
| 40 | `fixed · 512 KiB` | bottle/0 = 64×1654<br>cable/0 = 64×1654<br>capsule/0 = 64×1654<br>carpet/0 = 64×1654<br>grid/0 = 64×1654<br>hazelnut/0 = 64×1654<br>leather/0 = 64×1654<br>metal_nut/0 = 64×1654<br>pill/0 = 64×1654<br>screw/0 = 64×1654<br>tile/0 = 64×1654<br>toothbrush/0 = 64×1654<br>transistor/0 = 64×1654<br>wood/0 = 64×1654<br>zipper/0 = 64×1654<br>bottle/1 = 64×1654<br>bottle/2 = 64×1654 |
| 41 | `variance · 512 KiB` | bottle/0 = 51×2173<br>cable/0 = 142×534<br>capsule/0 = 56×1945<br>carpet/0 = 113×770<br>grid/0 = 82×1206<br>hazelnut/0 = 137×568<br>leather/0 = 103×882<br>metal_nut/0 = 124×667<br>pill/0 = 94×1003<br>screw/0 = 117×730<br>tile/0 = 153×468<br>toothbrush/0 = 74×1378<br>transistor/0 = 125×659<br>wood/0 = 141×541<br>zipper/0 = 65×1622<br>bottle/1 = 51×2173<br>bottle/2 = 50×2224 |
| 42 | `joint · 512 KiB` | bottle/0 = 64×1654<br>cable/0 = 32×3692<br>capsule/0 = 64×1654<br>carpet/0 = 16×7768<br>grid/0 = 32×3692<br>hazelnut/0 = 32×3692<br>leather/0 = 16×7768<br>metal_nut/0 = 32×3692<br>pill/0 = 32×3692<br>screw/0 = 32×3692<br>tile/0 = 16×7768<br>toothbrush/0 = 32×3692<br>transistor/0 = 32×3692<br>wood/0 = 16×7768<br>zipper/0 = 32×3692<br>bottle/1 = 64×1654<br>bottle/2 = 64×1654 |
| 43 | `reconstruction · 512 KiB` | bottle/0 = 128×635<br>cable/0 = 128×635<br>capsule/0 = 128×635<br>carpet/0 = 128×635<br>grid/0 = 128×635<br>hazelnut/0 = 128×635<br>leather/0 = 128×635<br>metal_nut/0 = 128×635<br>pill/0 = 128×635<br>screw/0 = 128×635<br>tile/0 = 128×635<br>toothbrush/0 = 128×635<br>transistor/0 = 128×635<br>wood/0 = 128×635<br>zipper/0 = 128×635<br>bottle/1 = 128×635<br>bottle/2 = 128×635 |
| 44 | `coverage · 512 KiB` | bottle/0 = 16×7768<br>cable/0 = 16×7768<br>capsule/0 = 16×7768<br>carpet/0 = 16×7768<br>grid/0 = 16×7768<br>hazelnut/0 = 16×7768<br>leather/0 = 16×7768<br>metal_nut/0 = 16×7768<br>pill/0 = 16×7768<br>screw/0 = 16×7768<br>tile/0 = 16×7768<br>toothbrush/0 = 16×7768<br>transistor/0 = 16×7768<br>wood/0 = 16×7768<br>zipper/0 = 16×7768<br>bottle/1 = 16×7768<br>bottle/2 = 16×7768 |
| 45 | `whitened_joint · 512 KiB` | bottle/0 = 32×3692<br>cable/0 = 32×3692<br>capsule/0 = 64×1654<br>carpet/0 = 16×7768<br>grid/0 = 16×7768<br>hazelnut/0 = 32×3692<br>leather/0 = 16×7768<br>metal_nut/0 = 32×3692<br>pill/0 = 32×3692<br>screw/0 = 32×3692<br>tile/0 = 16×7768<br>toothbrush/0 = 32×3692<br>transistor/0 = 32×3692<br>wood/0 = 16×7768<br>zipper/0 = 32×3692<br>bottle/1 = 32×3692<br>bottle/2 = 32×3692 |
| 46 | `raw · 1 MiB` | bottle/0 = 1024×255<br>cable/0 = 1024×255<br>capsule/0 = 1024×255<br>carpet/0 = 1024×255<br>grid/0 = 1024×255<br>hazelnut/0 = 1024×255<br>leather/0 = 1024×255<br>metal_nut/0 = 1024×255<br>pill/0 = 1024×255<br>screw/0 = 1024×255<br>tile/0 = 1024×255<br>toothbrush/0 = 1024×255<br>transistor/0 = 1024×255<br>wood/0 = 1024×255<br>zipper/0 = 1024×255 |
| 47 | `raw · 2 MiB` | bottle/0 = 1024×511<br>cable/0 = 1024×511<br>capsule/0 = 1024×511<br>carpet/0 = 1024×511<br>grid/0 = 1024×511<br>hazelnut/0 = 1024×511<br>leather/0 = 1024×511<br>metal_nut/0 = 1024×511<br>pill/0 = 1024×511<br>screw/0 = 1024×511<br>tile/0 = 1024×511<br>toothbrush/0 = 1024×511<br>transistor/0 = 1024×511<br>wood/0 = 1024×511<br>zipper/0 = 1024×511 |
| 48 | `raw · 4 MiB` | bottle/0 = 1024×1023<br>cable/0 = 1024×1023<br>capsule/0 = 1024×1023<br>carpet/0 = 1024×1023<br>grid/0 = 1024×1023<br>hazelnut/0 = 1024×1023<br>leather/0 = 1024×1023<br>metal_nut/0 = 1024×1023<br>pill/0 = 1024×1023<br>screw/0 = 1024×1023<br>tile/0 = 1024×1023<br>toothbrush/0 = 1024×1023<br>transistor/0 = 1024×1023<br>wood/0 = 1024×1023<br>zipper/0 = 1024×1023 |
| 49 | `raw · 8 MiB` | bottle/0 = 1024×2047<br>cable/0 = 1024×2047<br>capsule/0 = 1024×2047<br>carpet/0 = 1024×2047<br>grid/0 = 1024×2047<br>hazelnut/0 = 1024×2047<br>leather/0 = 1024×2047<br>metal_nut/0 = 1024×2047<br>pill/0 = 1024×2047<br>screw/0 = 1024×2047<br>tile/0 = 1024×2047<br>toothbrush/0 = 1024×2047<br>transistor/0 = 1024×2047<br>wood/0 = 1024×2047<br>zipper/0 = 1024×2047 |
| 50 | `pca128 · 1 MiB` | bottle/0 = 128×1013<br>cable/0 = 128×1013<br>capsule/0 = 128×1013<br>carpet/0 = 128×1013<br>grid/0 = 128×1013<br>hazelnut/0 = 128×1013<br>leather/0 = 128×1013<br>metal_nut/0 = 128×1013<br>pill/0 = 128×1013<br>screw/0 = 128×1013<br>tile/0 = 128×1013<br>toothbrush/0 = 128×1013<br>transistor/0 = 128×1013<br>wood/0 = 128×1013<br>zipper/0 = 128×1013 |
| 51 | `pca128 · 2 MiB` | bottle/0 = 128×3061<br>cable/0 = 128×3061<br>capsule/0 = 128×3061<br>carpet/0 = 128×3061<br>grid/0 = 128×3061<br>hazelnut/0 = 128×3061<br>leather/0 = 128×3061<br>metal_nut/0 = 128×3061<br>pill/0 = 128×3061<br>screw/0 = 128×3061<br>tile/0 = 128×3061<br>toothbrush/0 = 128×3061<br>transistor/0 = 128×3061<br>wood/0 = 128×3061<br>zipper/0 = 128×3061 |
| 52 | `pca128 · 4 MiB` | bottle/0 = 128×7157<br>cable/0 = 128×7157<br>capsule/0 = 128×7157<br>carpet/0 = 128×7157<br>grid/0 = 128×7157<br>hazelnut/0 = 128×7157<br>leather/0 = 128×7157<br>metal_nut/0 = 128×7157<br>pill/0 = 128×7157<br>screw/0 = 128×7157<br>tile/0 = 128×7157<br>toothbrush/0 = 128×7157<br>transistor/0 = 128×7157<br>wood/0 = 128×7157<br>zipper/0 = 128×7157 |
| 53 | `pca128 · 8 MiB` | bottle/0 = 128×15349<br>cable/0 = 128×15349<br>capsule/0 = 128×15349<br>carpet/0 = 128×15349<br>grid/0 = 128×15349<br>hazelnut/0 = 128×15349<br>leather/0 = 128×15349<br>metal_nut/0 = 128×15349<br>pill/0 = 128×15349<br>screw/0 = 128×15349<br>tile/0 = 128×15349<br>toothbrush/0 = 128×15349<br>transistor/0 = 128×15349<br>wood/0 = 128×15349<br>zipper/0 = 128×15349 |
| 54 | `Ứng viên PCA theo ngưỡng phương sai 95% · 256 KiB` | bottle/0 = 51×888<br>cable/0 = 142×73<br>capsule/0 = 56×774<br>carpet/0 = 113×190<br>grid/0 = 82×407<br>hazelnut/0 = 137×89<br>leather/0 = 103×246<br>metal_nut/0 = 124×139<br>pill/0 = 94×306<br>screw/0 = 117×170<br>tile/0 = 153×40<br>toothbrush/0 = 74×492<br>transistor/0 = 125×135<br>wood/0 = 141×76<br>zipper/0 = 65×614<br>bottle/1 = 51×888<br>bottle/2 = 50×913 |
| 55 | `Ứng viên PCA theo ngưỡng phương sai 95% · 512 KiB` | bottle/0 = 51×2173<br>cable/0 = 142×534<br>capsule/0 = 56×1945<br>carpet/0 = 113×770<br>grid/0 = 82×1206<br>hazelnut/0 = 137×568<br>leather/0 = 103×882<br>metal_nut/0 = 124×667<br>pill/0 = 94×1003<br>screw/0 = 117×730<br>tile/0 = 153×468<br>toothbrush/0 = 74×1378<br>transistor/0 = 125×659<br>wood/0 = 141×541<br>zipper/0 = 65×1622<br>bottle/1 = 51×2173<br>bottle/2 = 50×2224 |
| 56 | `Ứng viên pca_d128__b262144` | bottle/0 = 128×123<br>cable/0 = 128×123<br>capsule/0 = 128×123<br>carpet/0 = 128×123<br>grid/0 = 128×123<br>hazelnut/0 = 128×123<br>leather/0 = 128×123<br>metal_nut/0 = 128×123<br>pill/0 = 128×123<br>screw/0 = 128×123<br>tile/0 = 128×123<br>toothbrush/0 = 128×123<br>transistor/0 = 128×123<br>wood/0 = 128×123<br>zipper/0 = 128×123<br>bottle/1 = 128×123<br>bottle/2 = 128×123 |
| 57 | `Ứng viên pca_d128__b524288` | bottle/0 = 128×635<br>cable/0 = 128×635<br>capsule/0 = 128×635<br>carpet/0 = 128×635<br>grid/0 = 128×635<br>hazelnut/0 = 128×635<br>leather/0 = 128×635<br>metal_nut/0 = 128×635<br>pill/0 = 128×635<br>screw/0 = 128×635<br>tile/0 = 128×635<br>toothbrush/0 = 128×635<br>transistor/0 = 128×635<br>wood/0 = 128×635<br>zipper/0 = 128×635<br>bottle/1 = 128×635<br>bottle/2 = 128×635 |
| 58 | `Ứng viên pca_d16__b262144` | bottle/0 = 16×3672<br>cable/0 = 16×3672<br>capsule/0 = 16×3672<br>carpet/0 = 16×3672<br>grid/0 = 16×3672<br>hazelnut/0 = 16×3672<br>leather/0 = 16×3672<br>metal_nut/0 = 16×3672<br>pill/0 = 16×3672<br>screw/0 = 16×3672<br>tile/0 = 16×3672<br>toothbrush/0 = 16×3672<br>transistor/0 = 16×3672<br>wood/0 = 16×3672<br>zipper/0 = 16×3672<br>bottle/1 = 16×3672<br>bottle/2 = 16×3672 |
| 59 | `Ứng viên pca_d16__b524288` | bottle/0 = 16×7768<br>cable/0 = 16×7768<br>capsule/0 = 16×7768<br>carpet/0 = 16×7768<br>grid/0 = 16×7768<br>hazelnut/0 = 16×7768<br>leather/0 = 16×7768<br>metal_nut/0 = 16×7768<br>pill/0 = 16×7768<br>screw/0 = 16×7768<br>tile/0 = 16×7768<br>toothbrush/0 = 16×7768<br>transistor/0 = 16×7768<br>wood/0 = 16×7768<br>zipper/0 = 16×7768<br>bottle/1 = 16×7768<br>bottle/2 = 16×7768 |
| 60 | `Ứng viên pca_d32__b262144` | bottle/0 = 32×1644<br>cable/0 = 32×1644<br>capsule/0 = 32×1644<br>carpet/0 = 32×1644<br>grid/0 = 32×1644<br>hazelnut/0 = 32×1644<br>leather/0 = 32×1644<br>metal_nut/0 = 32×1644<br>pill/0 = 32×1644<br>screw/0 = 32×1644<br>tile/0 = 32×1644<br>toothbrush/0 = 32×1644<br>transistor/0 = 32×1644<br>wood/0 = 32×1644<br>zipper/0 = 32×1644<br>bottle/1 = 32×1644<br>bottle/2 = 32×1644 |
| 61 | `Ứng viên pca_d32__b524288` | bottle/0 = 32×3692<br>cable/0 = 32×3692<br>capsule/0 = 32×3692<br>carpet/0 = 32×3692<br>grid/0 = 32×3692<br>hazelnut/0 = 32×3692<br>leather/0 = 32×3692<br>metal_nut/0 = 32×3692<br>pill/0 = 32×3692<br>screw/0 = 32×3692<br>tile/0 = 32×3692<br>toothbrush/0 = 32×3692<br>transistor/0 = 32×3692<br>wood/0 = 32×3692<br>zipper/0 = 32×3692<br>bottle/1 = 32×3692<br>bottle/2 = 32×3692 |
| 62 | `Ứng viên pca_d64__b262144` | bottle/0 = 64×630<br>cable/0 = 64×630<br>capsule/0 = 64×630<br>carpet/0 = 64×630<br>grid/0 = 64×630<br>hazelnut/0 = 64×630<br>leather/0 = 64×630<br>metal_nut/0 = 64×630<br>pill/0 = 64×630<br>screw/0 = 64×630<br>tile/0 = 64×630<br>toothbrush/0 = 64×630<br>transistor/0 = 64×630<br>wood/0 = 64×630<br>zipper/0 = 64×630<br>bottle/1 = 64×630<br>bottle/2 = 64×630 |
| 63 | `Ứng viên pca_d64__b524288` | bottle/0 = 64×1654<br>cable/0 = 64×1654<br>capsule/0 = 64×1654<br>carpet/0 = 64×1654<br>grid/0 = 64×1654<br>hazelnut/0 = 64×1654<br>leather/0 = 64×1654<br>metal_nut/0 = 64×1654<br>pill/0 = 64×1654<br>screw/0 = 64×1654<br>tile/0 = 64×1654<br>toothbrush/0 = 64×1654<br>transistor/0 = 64×1654<br>wood/0 = 64×1654<br>zipper/0 = 64×1654<br>bottle/1 = 64×1654<br>bottle/2 = 64×1654 |
| 64 | `Ứng viên raw_d384__b262144` | bottle/0 = 384×170<br>cable/0 = 384×170<br>capsule/0 = 384×170<br>carpet/0 = 384×170<br>grid/0 = 384×170<br>hazelnut/0 = 384×170<br>leather/0 = 384×170<br>metal_nut/0 = 384×170<br>pill/0 = 384×170<br>screw/0 = 384×170<br>tile/0 = 384×170<br>toothbrush/0 = 384×170<br>transistor/0 = 384×170<br>wood/0 = 384×170<br>zipper/0 = 384×170<br>bottle/1 = 384×170<br>bottle/2 = 384×170 |
| 65 | `Ứng viên raw_d384__b524288` | bottle/0 = 384×340<br>cable/0 = 384×340<br>capsule/0 = 384×340<br>carpet/0 = 384×340<br>grid/0 = 384×340<br>hazelnut/0 = 384×340<br>leather/0 = 384×340<br>metal_nut/0 = 384×340<br>pill/0 = 384×340<br>screw/0 = 384×340<br>tile/0 = 384×340<br>toothbrush/0 = 384×340<br>transistor/0 = 384×340<br>wood/0 = 384×340<br>zipper/0 = 384×340<br>bottle/1 = 384×340<br>bottle/2 = 384×340 |
| 66 | `Ứng viên whitened_d128__b262144` | bottle/0 = 128×123<br>cable/0 = 128×123<br>capsule/0 = 128×123<br>carpet/0 = 128×123<br>grid/0 = 128×123<br>hazelnut/0 = 128×123<br>leather/0 = 128×123<br>metal_nut/0 = 128×123<br>pill/0 = 128×123<br>screw/0 = 128×123<br>tile/0 = 128×123<br>toothbrush/0 = 128×123<br>transistor/0 = 128×123<br>wood/0 = 128×123<br>zipper/0 = 128×123<br>bottle/1 = 128×123<br>bottle/2 = 128×123 |
| 67 | `Ứng viên whitened_d128__b524288` | bottle/0 = 128×635<br>cable/0 = 128×635<br>capsule/0 = 128×635<br>carpet/0 = 128×635<br>grid/0 = 128×635<br>hazelnut/0 = 128×635<br>leather/0 = 128×635<br>metal_nut/0 = 128×635<br>pill/0 = 128×635<br>screw/0 = 128×635<br>tile/0 = 128×635<br>toothbrush/0 = 128×635<br>transistor/0 = 128×635<br>wood/0 = 128×635<br>zipper/0 = 128×635<br>bottle/1 = 128×635<br>bottle/2 = 128×635 |
| 68 | `Ứng viên whitened_d16__b262144` | bottle/0 = 16×3672<br>cable/0 = 16×3672<br>capsule/0 = 16×3672<br>carpet/0 = 16×3672<br>grid/0 = 16×3672<br>hazelnut/0 = 16×3672<br>leather/0 = 16×3672<br>metal_nut/0 = 16×3672<br>pill/0 = 16×3672<br>screw/0 = 16×3672<br>tile/0 = 16×3672<br>toothbrush/0 = 16×3672<br>transistor/0 = 16×3672<br>wood/0 = 16×3672<br>zipper/0 = 16×3672<br>bottle/1 = 16×3672<br>bottle/2 = 16×3672 |
| 69 | `Ứng viên whitened_d16__b524288` | bottle/0 = 16×7768<br>cable/0 = 16×7768<br>capsule/0 = 16×7768<br>carpet/0 = 16×7768<br>grid/0 = 16×7768<br>hazelnut/0 = 16×7768<br>leather/0 = 16×7768<br>metal_nut/0 = 16×7768<br>pill/0 = 16×7768<br>screw/0 = 16×7768<br>tile/0 = 16×7768<br>toothbrush/0 = 16×7768<br>transistor/0 = 16×7768<br>wood/0 = 16×7768<br>zipper/0 = 16×7768<br>bottle/1 = 16×7768<br>bottle/2 = 16×7768 |
| 70 | `Ứng viên whitened_d32__b262144` | bottle/0 = 32×1644<br>cable/0 = 32×1644<br>capsule/0 = 32×1644<br>carpet/0 = 32×1644<br>grid/0 = 32×1644<br>hazelnut/0 = 32×1644<br>leather/0 = 32×1644<br>metal_nut/0 = 32×1644<br>pill/0 = 32×1644<br>screw/0 = 32×1644<br>tile/0 = 32×1644<br>toothbrush/0 = 32×1644<br>transistor/0 = 32×1644<br>wood/0 = 32×1644<br>zipper/0 = 32×1644<br>bottle/1 = 32×1644<br>bottle/2 = 32×1644 |
| 71 | `Ứng viên whitened_d32__b524288` | bottle/0 = 32×3692<br>cable/0 = 32×3692<br>capsule/0 = 32×3692<br>carpet/0 = 32×3692<br>grid/0 = 32×3692<br>hazelnut/0 = 32×3692<br>leather/0 = 32×3692<br>metal_nut/0 = 32×3692<br>pill/0 = 32×3692<br>screw/0 = 32×3692<br>tile/0 = 32×3692<br>toothbrush/0 = 32×3692<br>transistor/0 = 32×3692<br>wood/0 = 32×3692<br>zipper/0 = 32×3692<br>bottle/1 = 32×3692<br>bottle/2 = 32×3692 |
| 72 | `Ứng viên whitened_d64__b262144` | bottle/0 = 64×630<br>cable/0 = 64×630<br>capsule/0 = 64×630<br>carpet/0 = 64×630<br>grid/0 = 64×630<br>hazelnut/0 = 64×630<br>leather/0 = 64×630<br>metal_nut/0 = 64×630<br>pill/0 = 64×630<br>screw/0 = 64×630<br>tile/0 = 64×630<br>toothbrush/0 = 64×630<br>transistor/0 = 64×630<br>wood/0 = 64×630<br>zipper/0 = 64×630<br>bottle/1 = 64×630<br>bottle/2 = 64×630 |
| 73 | `Ứng viên whitened_d64__b524288` | bottle/0 = 64×1654<br>cable/0 = 64×1654<br>capsule/0 = 64×1654<br>carpet/0 = 64×1654<br>grid/0 = 64×1654<br>hazelnut/0 = 64×1654<br>leather/0 = 64×1654<br>metal_nut/0 = 64×1654<br>pill/0 = 64×1654<br>screw/0 = 64×1654<br>tile/0 = 64×1654<br>toothbrush/0 = 64×1654<br>transistor/0 = 64×1654<br>wood/0 = 64×1654<br>zipper/0 = 64×1654<br>bottle/1 = 64×1654<br>bottle/2 = 64×1654 |

## Ngưỡng clip INT4 Robust theo lớp và seed

Các dòng `before` và `after` dùng chung ngưỡng của từng quantile. Dấu `—` là quantile không được dùng ở lớp và seed này.

| Lớp | Seed | q=0,85 | q=0,95 | q=0,99 |
|---|---:|---:|---:|---:|
| bottle | 0 | 1.0623573064804077 | 1.1412084102630615 | 1.2507787942886353 |
| cable | 0 | 1.6905642747879028 | 1.7839829921722412 | 1.9004825353622437 |
| capsule | 0 | 0.8427730202674866 | 0.931633710861206 | 1.0886260271072388 |
| carpet | 0 | 1.0073052644729614 | 1.0664799213409424 | 1.1396571397781372 |
| grid | 0 | 1.0234826803207397 | 1.1283851861953735 | 1.2362537384033203 |
| hazelnut | 0 | 1.4738717079162598 | 1.5783510208129883 | 1.7004594802856445 |
| leather | 0 | 0.9122434854507446 | 0.9662541747093201 | 1.0292847156524658 |
| metal_nut | 0 | 1.5098639726638794 | 1.6024123430252075 | 1.705808162689209 |
| pill | 0 | 1.2543432712554932 | 1.3404300212860107 | 1.4422607421875 |
| screw | 0 | 1.1367096900939941 | 1.2829279899597168 | 1.461850881576538 |
| tile | 0 | 1.2559527158737183 | 1.3135502338409424 | 1.3906632661819458 |
| toothbrush | 0 | 1.1093254089355469 | 1.1946139335632324 | 1.2827352285385132 |
| transistor | 0 | 1.3149994611740112 | 1.406157374382019 | 1.5143969058990479 |
| wood | 0 | 1.1074029207229614 | 1.1686155796051025 | 1.2492921352386475 |
| zipper | 0 | 0.8088176250457764 | 0.8766089677810669 | 0.9737227559089661 |

## Manifest của từng lớp

| Lớp | Build | Calibration | Test | SHA-256 |
|---|---:|---:|---:|---|
| bottle (nguồn tham chiếu trong workspace thực nghiệm) | 167 | 42 | 83 | `19d5b0b09cf57571c2eba193629e05758681cc600fb9097531fae1fbc0bc0e90` |
| cable (nguồn tham chiếu trong workspace thực nghiệm) | 179 | 45 | 150 | `8a42440a940c1c2662cd1a08c17a77851dfbe2b722e5a58db7b254ced926686b` |
| capsule (nguồn tham chiếu trong workspace thực nghiệm) | 175 | 44 | 132 | `7fc11a312451026db503c02900e63bde50734139a58b50175bc5b15ebcc5fdbf` |
| carpet (nguồn tham chiếu trong workspace thực nghiệm) | 224 | 56 | 117 | `c65b6e64eced8e3886357d27eb451f62d96a9debf8ff40fc7642846c89340fa3` |
| grid (nguồn tham chiếu trong workspace thực nghiệm) | 211 | 53 | 78 | `9f1b631209b039eced0a7fc3c8a3f95568306887d654c6b807a33cd43eca663f` |
| hazelnut (nguồn tham chiếu trong workspace thực nghiệm) | 313 | 78 | 110 | `ef62b179cc27bc1c76e79f0984bac83613d2550b9e97e5f38a7075da62f7978c` |
| leather (nguồn tham chiếu trong workspace thực nghiệm) | 196 | 49 | 124 | `08675c4568d671456caf4d31b80693c83649ddae34779f8d1418f1d84572247a` |
| metal_nut (nguồn tham chiếu trong workspace thực nghiệm) | 176 | 44 | 115 | `d74d8b53a87515bfe4ea4906610706c49213cb5dae09d853bbaa031d9a9c4b5a` |
| pill (nguồn tham chiếu trong workspace thực nghiệm) | 214 | 53 | 167 | `55f5610fff514089297e4cea6c69e72b62d013d171550081535c603603cfe9cd` |
| screw (nguồn tham chiếu trong workspace thực nghiệm) | 256 | 64 | 160 | `6317557d402e31b07c3a4e805035f9316979db3ad829da02d85f4912d68efa50` |
| tile (nguồn tham chiếu trong workspace thực nghiệm) | 184 | 46 | 117 | `5f1158ceeb2955640ee1f97736a5723b2e8c1886921f07cd70c24f837aed9841` |
| toothbrush (nguồn tham chiếu trong workspace thực nghiệm) | 48 | 12 | 42 | `5217e57982878572db45cd21b0bbfd30c4385224c1fa2db956ab976bff100bef` |
| transistor (nguồn tham chiếu trong workspace thực nghiệm) | 170 | 43 | 100 | `37e3a6b07c75be373886cb673aa67c21054afa6f117b40f28c364facbc1a8760` |
| wood (nguồn tham chiếu trong workspace thực nghiệm) | 198 | 49 | 79 | `cd7424a71487bca9e084c790dcf1905bec922217d909425609349ad0a496f594` |
| zipper (nguồn tham chiếu trong workspace thực nghiệm) | 192 | 48 | 151 | `c71b20441480ef7e84f50e40659f3936e75bac22297a6c8182dc6b03a81c4975` |
