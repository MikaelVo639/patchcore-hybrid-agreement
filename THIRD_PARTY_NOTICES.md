# Nguồn mô hình và dữ liệu

Mã của nhóm được phát triển trong dự án SC203, với thư viện `patchcore_budget` từ repo https://github.com/MikaelVo639/patchcore-memory-allocation. Giữ bản MIT license của nhóm trong `LICENSE`.

Encoder WRN được trích từ WideResNet50-2 ImageNet1K V1 của Torchvision, chỉ giữ layer1/layer2 và chuyển FP16. Encoder DINOv2 được chuyển FP16 từ checkpoint `timm/vit_small_patch14_dinov2.lvd142m`. File manifest ghi checksum của bản đóng gói.

- Torchvision và pretrained models thuộc dự án PyTorch, với license và nguồn tại https://github.com/pytorch/vision và https://pytorch.org/vision/stable/models.html.
- DINOv2 có mã nguồn và license tại https://github.com/facebookresearch/dinov2. Checkpoint gốc tại https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m.
- PatchCore gốc của Amazon là nguồn phương pháp tại https://github.com/amazon-science/patchcore-inspection.
- AnomalyDINO là nguồn tham khảo cho patch token và cosine tại https://github.com/dammsi/AnomalyDINO.
- Các bank được fit từ ảnh train normal MVTec AD. Điều kiện dataset và thông tin sử dụng phi thương mại tại https://www.mvtec.com/research-teaching/datasets/mvtec-ad. Các bank giữ điều kiện áp dụng từ dữ liệu nguồn.

Các package phụ thuộc giữ license riêng. Repo bao gồm encoder pretrained đóng gói và memory bank đã fit, cùng số liệu benchmark của nhóm.
