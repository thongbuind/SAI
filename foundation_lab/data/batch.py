from dataclasses import dataclass, replace
import torch

@dataclass
class LMBatch:
    """Hợp đồng batch cho causal LM.

    - ``input_ids``, ``labels``: (B, T) long; giá trị ở vị trí padding không mang nghĩa (vị trí padding
      được đánh dấu bằng ``attention_mask`` và ``loss_weight``, không suy ra từ giá trị token).
    - ``labels`` đã shift: ``labels[b, t]`` là token đích khi model đọc ``input_ids[b, :t+1]``.
    - ``loss_weight``: (B, T) float; 0 ở PAD và ở token không chịu loss (prompt trong SFT).
    - ``attention_mask``: (B, T) float, 1 = token thật. Model SAI hiện bỏ qua mask này (xem
      ``SAIModel.forward_features``), nên mọi batch phải giữ ``padding_side == "right"``.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_weight: torch.Tensor
    attention_mask: torch.Tensor
    has_padding: bool
    padding_side: str = "right"

    @property
    def shape(self):
        return tuple(self.input_ids.shape)

    @property
    def num_loss_tokens(self) -> torch.Tensor:
        return self.loss_weight.sum()

    def to(self, device, non_blocking=False) -> "LMBatch":
        return replace(
            self,
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            loss_weight=self.loss_weight.to(device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "LMBatch":
        """DataLoader gọi hàm này khi ``pin_memory=True``."""
        return replace(
            self,
            input_ids=self.input_ids.pin_memory(),
            labels=self.labels.pin_memory(),
            loss_weight=self.loss_weight.pin_memory(),
            attention_mask=self.attention_mask.pin_memory(),
        )
