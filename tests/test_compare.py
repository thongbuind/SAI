from foundation_lab.cli import main
from foundation_lab.eval.compare import compare_attention_kernels

def test_compare_sdpa_with_eager_reference():
    report = compare_attention_kernels("sdpa", "eager", seq_len=16, head_dim=8, repeats=1)
    for key in ("out", "grad_q", "grad_k", "grad_v"):
        assert report[f"max_abs_diff_{key}"] < 1e-5, (key, report)
    assert report["reference_ms"] > 0 and report["candidate_ms"] > 0

def test_list_command_shows_every_department(capsys):
    main(["list"])
    out = capsys.readouterr().out
    for department in ("[arch]", "[data]", "[objectives]", "[engine]", "[recipes]"):
        assert department in out
    assert "sdpa" in out and "causal_lm" in out and "sft" in out
