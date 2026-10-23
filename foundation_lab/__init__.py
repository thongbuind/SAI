"""SAI training lab (dev3).

Model chính thức (thứ được phát hành) nằm trong package ``SAI``; lab phụ thuộc ``SAI``, không có chiều ngược lại.
Lab được tổ chức như các phòng ban, mỗi phòng sở hữu một registry và một hợp đồng giao diện:

- ``core``        Ban điều hành: registry, config nghiêm ngặt, experiment/pipeline spec.
- ``arch``        Phòng Kiến trúc: bản lắp ghép ``SAI_modular``, attention kernel, layer, họ mô hình.
- ``data``        Phòng Dữ liệu: nguồn dữ liệu, split, trộn, sampler, collate, ``LMBatch``.
- ``tokenizer``   Phòng Tokenizer: train SentencePiece, encode văn bản thô thành ``.npz``.
- ``objectives``  Phòng Objective: các hàm loss thuần.
- ``engine``      Phòng Vận hành: device/AMP/compile, microbatch, optimizer, vòng lặp, profiling.
- ``state``       Phòng Lưu trữ: thư mục run, checkpoint, lineage, metrics.
- ``recipes``     Phòng Phương pháp huấn luyện: pretrain, SFT (sau này distill, DPO, RLHF...).
- ``eval``        Phòng Đánh giá: loss, công cụ so sánh.

Muốn thêm một biến thể: đăng ký implementation mới vào registry của phòng tương ứng
rồi chọn nó qua config, không sửa code của phòng khác.
"""

__version__ = "0.3.0"
