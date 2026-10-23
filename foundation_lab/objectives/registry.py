from foundation_lab.core.registry import Registry

# Objective là hàm thuần trả về TỔNG loss (và recipe tự quyết mẫu số), để engine chuẩn hoá thống nhất.
# Hợp đồng: (hidden (N, d_model), head_weight (vocab, d_model), targets (N,) long, token_weight (N,) float,
# *, chunk_size) -> tổng loss có trọng số, scalar float32. Recipe chọn objective cho bước train qua
# ``params.objective``; val/test luôn tính bằng ``lm_ce`` để loss so sánh được giữa các thí nghiệm.
OBJECTIVES = Registry("objective")
