import torch
import torch.nn.functional as F
from foundation_lab.objectives import chunked_lm_loss

def _inputs(n=50, d=16, vocab=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n, d, generator=g, requires_grad=True)
    weight = torch.randn(vocab, d, generator=g, requires_grad=True)
    targets = torch.randint(0, vocab, (n,), generator=g)
    token_weight = (torch.rand(n, generator=g) > 0.3).float()
    return hidden, weight, targets, token_weight

def test_matches_naive_full_logits():
    hidden, weight, targets, token_weight = _inputs()
    naive = (F.cross_entropy(hidden @ weight.T, targets, reduction="none") * token_weight).sum()
    chunked = chunked_lm_loss(hidden, weight, targets, token_weight, chunk_size=7)
    torch.testing.assert_close(chunked, naive, atol=1e-5, rtol=1e-6)

def test_independent_of_chunk_size_including_gradients():
    results = []
    for chunk_size in (1, 7, 4096):
        hidden, weight, targets, token_weight = _inputs()
        loss = chunked_lm_loss(hidden, weight, targets, token_weight, chunk_size=chunk_size)
        loss.backward()
        results.append((loss.detach(), hidden.grad, weight.grad))
    for loss, h_grad, w_grad in results[1:]:
        torch.testing.assert_close(loss, results[0][0], atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(h_grad, results[0][1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(w_grad, results[0][2], atol=1e-6, rtol=1e-5)

def test_masked_tokens_get_no_gradient():
    hidden, weight, targets, token_weight = _inputs()
    loss = chunked_lm_loss(hidden, weight, targets, token_weight, chunk_size=8)
    loss.backward()
    masked = token_weight == 0
    assert masked.any()
    assert torch.count_nonzero(hidden.grad[masked]) == 0
    assert torch.count_nonzero(hidden.grad[~masked]) > 0

def test_no_grad_path_matches_grad_path():
    hidden, weight, targets, token_weight = _inputs()
    with_grad = chunked_lm_loss(hidden, weight, targets, token_weight, chunk_size=8)
    with torch.no_grad():
        without_grad = chunked_lm_loss(hidden, weight, targets, token_weight, chunk_size=8)
    torch.testing.assert_close(with_grad.detach(), without_grad, atol=0, rtol=0)
