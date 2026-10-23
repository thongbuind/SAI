"""Quản lý phát hành SAI trên Hugging Face.

- ``adapter/``      file được upload để tải model bằng Transformers (``trust_remote_code=True``)
- ``releases/``     mỗi YAML chọn repo, nguồn trọng số và model card; version lấy từ SAI.version
- ``model_card.md`` mẫu README trên Hub
- ``export``        release spec → thư mục đúng định dạng repo HF
- ``verify``        tải thư mục export như người dùng, so với ``SAI.modeling`` (bắt buộc trước khi push)
- ``push``          đưa lên Hub thành một commit + tag phiên bản (cần xác nhận)
- ``artifact``      manifest công khai, kiểm tra hash mọi file và ràng buộc báo cáo verify

Phiên bản SAI, config, manifest và tag HF phải khớp nhau. Export đang dirty dùng để thử;
push chính thức yêu cầu export từ commit sạch. Xem trước push lưu kế hoạch và commit cha trên Hub;
--confirm chỉ chấp nhận kế hoạch chưa đổi. Tiến độ upload được ghi trước khi gắn tag để có thể tiếp tục.

Phụ thuộc: ``SAI`` (code chính thức) và ``foundation_lab`` (đọc thư mục run). ``SAI`` và ``foundation_lab`` không
phụ thuộc package này.
"""
