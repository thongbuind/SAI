import numpy as np
import torch

class RandomizedSortishBatchSampler(torch.utils.data.Sampler):
    """Batch theo số mẫu cố định; mẫu được xếp gần theo độ dài trong từng pool để giảm padding."""

    def __init__(self, lengths, batch_size, pool_batches=50, drop_last=False, seed=None):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.pool_size = max(batch_size, batch_size * pool_batches)
        self.drop_last = drop_last
        self.seed = seed if seed is not None else int(
            torch.empty((), dtype=torch.int64).random_().item()
        )
        self.epoch = 0
        self._skip_batches = 0

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.seed = state["seed"]
        self.epoch = state["epoch"]

    def skip_next_batches(self, count: int):
        """Lượt duyệt kế tiếp không trả ra ``count`` batch đầu (resume giữa epoch).

        Thứ tự batch vẫn y như lượt duyệt đầy đủ, nhưng DataLoader không phải đọc và collate các batch bị bỏ.
        """
        self._skip_batches = count

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return int(np.ceil(len(self.lengths) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled_indices = rng.permutation(len(self.lengths))
        self.epoch += 1
        skip = self._skip_batches
        self._skip_batches = 0

        for pool_start in range(0, len(shuffled_indices), self.pool_size):
            pool = shuffled_indices[pool_start:pool_start + self.pool_size]
            length_order = np.argsort(self.lengths[pool], kind="stable")
            sorted_pool = pool[length_order]

            pool_batches = []
            for batch_start in range(0, len(sorted_pool), self.batch_size):
                batch = sorted_pool[batch_start:batch_start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batch = batch.tolist()
                rng.shuffle(batch)
                pool_batches.append(batch)

            # Vẫn dựng và xáo mọi pool (tiêu thụ RNG như lượt đầy đủ); chỉ không trả ra batch bị bỏ.
            rng.shuffle(pool_batches)
            for batch in pool_batches:
                if skip > 0:
                    skip -= 1
                    continue
                yield batch

class TokenBalancedSortishBatchSampler(torch.utils.data.Sampler):
    """Batch gom đến khi đủ ``target_loss_tokens`` token chịu loss (trung bình × batch_size) hoặc đủ ``max_samples`` mẫu."""

    def __init__(
        self, lengths, loss_token_counts, batch_size, pool_batches=50,
        max_samples=None, seed=None,
    ):
        self.lengths = np.asarray(lengths)
        self.loss_token_counts = np.asarray(loss_token_counts, dtype=np.int64)
        self.pool_size = max(batch_size, batch_size * pool_batches)
        self.max_samples = max_samples or batch_size * 4
        self.target_loss_tokens = max(
            1, int(round(self.loss_token_counts.mean() * batch_size))
        )
        self.seed = seed if seed is not None else int(
            torch.empty((), dtype=torch.int64).random_().item()
        )
        self.epoch = 0
        self._cached_epoch = None
        self._cached_batches = None
        self._skip_batches = 0

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.seed = state["seed"]
        self.epoch = state["epoch"]
        self._cached_epoch = None
        self._cached_batches = None

    def skip_next_batches(self, count: int):
        """Lượt duyệt kế tiếp không trả ra ``count`` batch đầu (resume giữa epoch).

        Thứ tự batch vẫn y như lượt duyệt đầy đủ, nhưng DataLoader không phải đọc và collate các batch bị bỏ.
        """
        self._skip_batches = count

    def _build_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled_indices = rng.permutation(len(self.lengths))
        batches = []
        current_batch = []
        current_loss_tokens = 0

        for pool_start in range(0, len(shuffled_indices), self.pool_size):
            pool = shuffled_indices[pool_start:pool_start + self.pool_size]
            length_order = np.argsort(self.lengths[pool], kind="stable")
            for idx in pool[length_order]:
                if current_batch and (
                    current_loss_tokens >= self.target_loss_tokens
                    or len(current_batch) >= self.max_samples
                ):
                    rng.shuffle(current_batch)
                    batches.append(current_batch)
                    current_batch = []
                    current_loss_tokens = 0

                current_batch.append(int(idx))
                current_loss_tokens += int(self.loss_token_counts[idx])

        if current_batch:
            rng.shuffle(current_batch)
            batches.append(current_batch)

        rng.shuffle(batches)
        return batches

    def _get_batches(self):
        if self._cached_epoch != self.epoch:
            self._cached_batches = self._build_batches()
            self._cached_epoch = self.epoch
        return self._cached_batches

    def __len__(self):
        return len(self._get_batches())

    def __iter__(self):
        batches = self._get_batches()
        self.epoch += 1
        self._cached_epoch = None
        self._cached_batches = None
        skip = self._skip_batches
        self._skip_batches = 0
        yield from batches[skip:]
