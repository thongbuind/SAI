import numpy as np
import torch
from foundation_lab.core.log import log_progress
from foundation_lab.data.batch import LMBatch
from foundation_lab.data.samplers import RandomizedSortishBatchSampler, TokenBalancedSortishBatchSampler

PAD_ID = 0
BUCKET_SIZE = 128

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, lengths, indices, loss_masks=None):
        self.X = X
        self.Y = Y
        self.lengths = lengths
        self.indices = indices
        self.loss_masks = loss_masks

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (
            self.X[real_idx],
            self.Y[real_idx],
            self.lengths[real_idx],
            self.loss_masks[real_idx] if self.loss_masks is not None else None
        )

class BucketPadCollator:
    """Pad phải tới bội số của ``bucket_size`` (tối đa ``max_seq_len``) và dựng ``LMBatch``.

    Trọng số loss: pretrain = 1 ở mọi vị trí token thật (theo độ dài mẫu); SFT dùng loss mask đã encode.
    ``pad_id`` chỉ là giá trị lấp chỗ trống, không dùng để nhận diện padding.
    """

    def __init__(self, max_seq_len: int, with_mask: bool, bucket_size: int = BUCKET_SIZE, pad_id: int = PAD_ID):
        self.max_seq_len = max_seq_len
        self.with_mask = with_mask
        self.bucket_size = bucket_size
        self.pad_id = pad_id

    def __call__(self, batch) -> LMBatch:
        max_seq_len, bucket_size = self.max_seq_len, self.bucket_size
        X_batch = [item[0] for item in batch]
        Y_batch = [item[1] for item in batch]
        lm_batch = [item[3] for item in batch]

        max_len = max(len(x) for x in X_batch)
        if max_len > (max_seq_len - (max_seq_len%bucket_size)):
            bucket = max_seq_len
        else:
            bucket = int(np.ceil(max_len / bucket_size) * bucket_size)

        bsz = len(X_batch)
        X_padded = torch.full((bsz, bucket), self.pad_id, dtype=torch.long)
        Y_padded = torch.full((bsz, bucket), self.pad_id, dtype=torch.long)
        has_padding = any(len(x) < bucket for x in X_batch)

        if self.with_mask:
            loss_mask_padded = torch.zeros((bsz, bucket), dtype=torch.float)

        for i, (x, y) in enumerate(zip(X_batch, Y_batch)):
            x_len = len(x)
            y_len = len(y)

            X_padded[i, :x_len] = torch.as_tensor(x, dtype=torch.long)
            Y_padded[i, :y_len] = torch.as_tensor(y, dtype=torch.long)

            if self.with_mask:
                lm = lm_batch[i]
                lm_len = len(lm)
                loss_mask_padded[i, :lm_len] = torch.as_tensor(lm, dtype=torch.float)

        # Vị trí token thật lấy theo độ dài mẫu, không theo giá trị: PAD_ID = 0 trùng id của [UNK].
        lengths = torch.as_tensor([len(x) for x in X_batch])            # (B,)
        real_tokens = (torch.arange(bucket) < lengths[:, None]).float()  # (B, T)
        attention_mask = real_tokens

        if not self.with_mask:
            sample_weight = real_tokens
        else:
            sample_weight = loss_mask_padded

        return LMBatch(
            input_ids=X_padded,
            labels=Y_padded,
            loss_weight=sample_weight,
            attention_mask=attention_mask,
            has_padding=has_padding,
        )

def build_dataloader(
    X, Y, lengths, batch_size, max_seq_len, num_workers, shuffle,
    mode, loss_masks=None,
):
    """DataLoader kiểu dev_2.

    - ``pretrain`` + shuffle: ``RandomizedSortishBatchSampler`` (batch theo số mẫu).
    - ``finetune`` + shuffle: ``TokenBalancedSortishBatchSampler`` (batch theo số token chịu loss).
    - Không shuffle (val/test): xếp theo độ dài, cắt đều ``batch_size`` mẫu.

    Khác dev_2 duy nhất: ``persistent_workers`` chỉ bật khi ``num_workers > 0``
    (dev_2 luôn bật nên báo lỗi với ``num_workers=0``).
    """
    if mode not in {"pretrain", "finetune"}:
        raise ValueError(
            f"mode phải là 'pretrain' hoặc 'finetune', nhận được: {mode!r}"
        )
    if mode == "finetune" and loss_masks is None:
        raise ValueError("Finetune dataloader bắt buộc phải có loss_masks")

    log_progress(f"Đang tạo dataset từ {len(X)} samples...")
    indices = np.arange(len(X))
    dataset = TokenizedDataset(X, Y, lengths, indices, loss_masks)

    loader_kwargs = {
        "dataset": dataset,
        "collate_fn": BucketPadCollator(max_seq_len, with_mask=loss_masks is not None),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    if shuffle and mode == "pretrain":
        batch_sampler = RandomizedSortishBatchSampler(
            lengths=lengths,
            batch_size=batch_size,
            pool_batches=50,
            drop_last=False,
        )
        dataloader = torch.utils.data.DataLoader(
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
        log_progress(
            f"Sortish sampler: pool={batch_sampler.pool_size} samples, "
            f"{len(batch_sampler)} batch/epoch"
        )
    elif shuffle and mode == "finetune":
        loss_token_counts = np.fromiter(
            (np.asarray(mask).sum() for mask in loss_masks),
            dtype=np.int64,
            count=len(loss_masks),
        )
        batch_sampler = TokenBalancedSortishBatchSampler(
            lengths=lengths,
            loss_token_counts=loss_token_counts,
            batch_size=batch_size,
            pool_batches=50,
            max_samples=batch_size * 4,
        )
        dataloader = torch.utils.data.DataLoader(
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
        log_progress(
            f"SFT token-balanced sortish: target={batch_sampler.target_loss_tokens:,} "
            f"loss-token/batch, pool={batch_sampler.pool_size}, "
            f"max_samples={batch_sampler.max_samples}, "
            f"{len(batch_sampler)} batch/epoch"
        )
    else:
        length_order = np.argsort(np.asarray(lengths), kind="stable")
        batch_sampler = [
            length_order[i:i + batch_size].tolist()
            for i in range(0, len(length_order), batch_size)
        ]
        dataloader = torch.utils.data.DataLoader(
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )

    log_progress(f"Dataset mode={mode}, batch_size={batch_size}")
    return dataloader
