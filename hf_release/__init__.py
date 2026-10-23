"""Quản lý phát hành SAI trên Hugging Face.

- ``adapter/``      file được upload để tải model bằng Transformers (``trust_remote_code=True``)
- ``releases/``     mỗi file YAML mô tả một lần phát hành: repo, phiên bản, nguồn trọng số, model card
- ``model_card.md`` mẫu README trên Hub
- ``export``        release spec → thư mục đúng định dạng repo HF
- ``verify``        tải thư mục export như người dùng, so với ``SAI.modeling`` (bắt buộc trước khi push)
- ``push``          đưa lên Hub thành một commit + tag phiên bản (cần xác nhận)

Phụ thuộc: ``SAI`` (code chính thức) và ``foundation_lab`` (đọc thư mục run). ``SAI`` và ``foundation_lab`` không
phụ thuộc package này.
"""
