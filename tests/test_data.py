import numpy as np
import pytest
import torch
from foundation_lab.data import build_data_source
from foundation_lab.data.loaders import BucketPadCollator
from foundation_lab.data.samplers import RandomizedSortishBatchSampler, TokenBalancedSortishBatchSampler

LENGTHS = np.random.default_rng(0).integers(5, 300, size=203)

@pytest.mark.parametrize("make_sampler", [
    lambda seed: RandomizedSortishBatchSampler(LENGTHS, batch_size=8, pool_batches=3, seed=seed),
    lambda seed: TokenBalancedSortishBatchSampler(
        LENGTHS, loss_token_counts=LENGTHS // 2, batch_size=8, pool_batches=3, seed=seed,
    ),
])
def test_sampler_covers_each_index_once_and_is_deterministic(make_sampler):
    first = list(make_sampler(7))
    flat = sorted(i for batch in first for i in batch)
    assert flat == list(range(len(LENGTHS)))
    assert first == list(make_sampler(7))
    assert first != list(make_sampler(8))

@pytest.mark.parametrize("cls", [RandomizedSortishBatchSampler, TokenBalancedSortishBatchSampler])
def test_sampler_state_dict_reproduces_epoch(cls):
    kwargs = {"batch_size": 8, "pool_batches": 3, "seed": 3}
    if cls is TokenBalancedSortishBatchSampler:
        kwargs["loss_token_counts"] = LENGTHS // 2
    sampler = cls(LENGTHS, **kwargs)
    list(sampler)                       # epoch 0
    state = sampler.state_dict()        # trước epoch 1
    epoch1 = list(sampler)
    restored = cls(LENGTHS, **{**kwargs, "seed": 99})
    restored.load_state_dict(state)
    assert list(restored) == epoch1

@pytest.mark.parametrize("cls", [RandomizedSortishBatchSampler, TokenBalancedSortishBatchSampler])
def test_skip_next_batches_yields_tail_of_same_epoch_once(cls):
    kwargs = {"batch_size": 8, "pool_batches": 3, "seed": 3}
    if cls is TokenBalancedSortishBatchSampler:
        kwargs["loss_token_counts"] = LENGTHS // 2
    reference = cls(LENGTHS, **kwargs)
    epoch0, epoch1 = list(reference), list(reference)
    sampler = cls(LENGTHS, **kwargs)
    sampler.skip_next_batches(5)
    assert list(sampler) == epoch0[5:]
    assert list(sampler) == epoch1              # chỉ lượt duyệt kế tiếp bị bỏ batch

def test_token_balanced_respects_max_samples():
    sampler = TokenBalancedSortishBatchSampler(
        LENGTHS, loss_token_counts=np.ones_like(LENGTHS), batch_size=4, max_samples=6, seed=0,
    )
    assert max(len(batch) for batch in sampler) <= 6

def test_collate_pads_right_to_bucket_and_builds_weights():
    items = [([1, 5, 6], [5, 6, 2], 3, [0, 1, 1]), ([1, 7], [7, 2], 2, [1, 1])]
    batch = BucketPadCollator(max_seq_len=256, with_mask=True)(items)
    assert batch.shape == (2, 128) and batch.has_padding and batch.padding_side == "right"
    assert batch.input_ids[0, :3].tolist() == [1, 5, 6] and batch.input_ids[0, 3:].eq(0).all()
    assert batch.loss_weight[0, :3].tolist() == [0, 1, 1] and batch.loss_weight[0, 3:].eq(0).all()
    assert batch.attention_mask[1].sum() == 2

    lm_batch = BucketPadCollator(max_seq_len=256, with_mask=False)([(x, y, n, None) for x, y, n, _ in items])
    assert torch.equal(lm_batch.loss_weight, (lm_batch.labels != 0).float())

def test_collate_finds_padding_by_length_not_token_value():
    # Token id 0 ([UNK]) trùng PAD_ID: ở giữa mẫu nó vẫn là token thật, vẫn chịu loss.
    items = [([1, 0, 6], [0, 6, 2], 3, None), ([1, 7], [7, 2], 2, None)]
    batch = BucketPadCollator(max_seq_len=256, with_mask=False)(items)
    assert batch.attention_mask[0, :3].tolist() == [1, 1, 1] and batch.attention_mask[0, 3:].eq(0).all()
    assert batch.loss_weight[0, :3].tolist() == [1, 1, 1] and batch.loss_weight[0, 3:].eq(0).all()
    assert batch.num_loss_tokens.item() == 5

def test_long_sequence_uses_max_seq_len_bucket():
    items = [(list(range(1, 251)), list(range(1, 251)), 250, None)]
    batch = BucketPadCollator(max_seq_len=250, with_mask=False)(items)
    assert batch.shape == (1, 250)

def test_mixing_ratio_and_disjoint_splits(tiny_data):
    source = build_data_source({
        "kind": "lm", "main": str(tiny_data["continued"]), "sub": str(tiny_data["pretrain"]),
        "mix_ratio": 3, "train_ratio": 0.8, "val_ratio": 0.1,
    })
    X, _, M, _ = source.load()
    assert len(X) == 10 + 30 and M is None
    splits = source.load_splits()
    assert [len(splits[name]) for name in ("train", "val", "test")] == [32, 4, 4]

    def keys(split):
        return {tuple(x.tolist()) for x in split.X}
    assert not keys(splits["train"]) & keys(splits["val"])
    assert not keys(splits["train"]) & keys(splits["test"])

def test_sft_source_returns_masks(tiny_data):
    source = build_data_source({"kind": "sft", "main": str(tiny_data["sft1"])})
    X, Y, M, L = source.load()
    assert len(X) == len(M) == 40
    assert all(len(m) == len(y) for m, y in zip(M, Y))
    assert all(int(length) == len(x) for x, length in zip(X, L))
