import json
import os
import pytest
import torch
import yaml
from safetensors.torch import load_file
from helpers import TINY_MODEL
from SAI.version import COMPONENT_VERSIONS, MODEL_VERSION
from foundation_lab.catalog import build_model
from foundation_lab.core.config import ConfigError
from hf_release.export import HF_DIR, export_release
from hf_release.push import LOCAL_ONLY_FILES, plan_file_changes
from hf_release.spec import load_release_spec

@pytest.fixture(scope="module", autouse=True)
def isolated_hf_modules_cache(tmp_path_factory):
    """Code tải bằng trust_remote_code được Transformers chép vào HF_MODULES_CACHE; giữ nó trong thư mục tạm."""
    previous = os.environ.get("HF_MODULES_CACHE")
    os.environ["HF_MODULES_CACHE"] = str(tmp_path_factory.mktemp("hf_modules"))
    yield
    if previous is None:
        os.environ.pop("HF_MODULES_CACHE")
    else:
        os.environ["HF_MODULES_CACHE"] = previous

EXAMPLE_MESSAGES = [
    {"role": "system", "content": "Bạn là trợ lý trả lời ngắn gọn."},
    {"role": "user", "content": 'Câu hỏi "có nháy" cho SAI_tiny'},
]

@pytest.fixture
def tiny_release(tmp_path):
    torch.manual_seed(0)
    model = build_model(TINY_MODEL)
    weights = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), weights)
    spec = {
        "repo_id": "someone/SAI_tiny",
        "version": "v0.1",
        "source": {"weights": str(weights), "model": TINY_MODEL},
        "card": {"model_name": "SAI_tiny", "example_messages": EXAMPLE_MESSAGES},
    }
    spec_path = tmp_path / "SAI_tiny.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path, model

def test_export_contains_only_what_users_need(tmp_path, tiny_release):
    spec_path, model = tiny_release
    out = export_release(spec_path, out_dir=tmp_path / "export")
    core_files = {"modeling.py", "generate.py", "rope.py", "attention.py", "ffn.py", "block.py", "version.py"}
    uploaded = core_files | {
        "model.safetensors", "config.json", "modeling_SAI.py",
        "configuration_SAI.py", "tokenization_SAI.py", "tokenizer_config.json", "tokenizer.model",
        "README.md",
    }
    assert {p.name for p in out.iterdir()} == uploaded | {"release.json"}
    assert "release.json" in LOCAL_ONLY_FILES

    config = json.loads((out / "config.json").read_text())
    assert config["model_type"] == "SAI" and config["architectures"] == ["SAIForCausalLM"]
    assert config["auto_map"]["AutoModelForCausalLM"] == "modeling_SAI.SAIForCausalLM"
    assert config["num_kv_heads"] == TINY_MODEL["num_kv_heads"] == config["num_key_value_heads"]

    tensors = load_file(str(out / "model.safetensors"))
    assert "lm_head.weight" not in tensors
    assert torch.equal(tensors["embed.weight"], model.embed.weight)

    card = (out / "README.md").read_text(encoding="utf-8")
    assert "someone/SAI_tiny" in card and "`v0.1`" in card and "model_name: SAI_tiny" in card
    assert f"`{MODEL_VERSION}`" in card
    assert '"content": "Câu hỏi \\"có nháy\\" cho SAI_tiny",' in card
    for name in core_files:
        assert (out / name).read_text() == (HF_DIR.parent / "SAI" / name).read_text()

    manifest = json.loads((out / "release.json").read_text())
    assert manifest["model_version"] == MODEL_VERSION
    assert manifest["component_versions"] == COMPONENT_VERSIONS

def test_export_refuses_to_overwrite(tmp_path, tiny_release):
    spec_path, _ = tiny_release
    export_release(spec_path, out_dir=tmp_path / "export")
    with pytest.raises(FileExistsError):
        export_release(spec_path, out_dir=tmp_path / "export")

def test_export_rejects_non_official_model(tmp_path, tiny_release):
    spec_path, _ = tiny_release
    spec = yaml.safe_load(spec_path.read_text())
    spec["source"]["model"] = {**TINY_MODEL, "family": "SAI_modular"}
    spec_path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ConfigError, match="family: SAI"):
        export_release(spec_path, out_dir=tmp_path / "export")

def test_verify_loads_export_like_a_hub_user(tmp_path, tiny_release):
    from hf_release.verify import verify_export

    spec_path, _ = tiny_release
    out = export_release(spec_path, out_dir=tmp_path / "export")
    report = verify_export(out)
    assert report["passed"], report
    assert (out / "verify.json").exists()

def test_push_plan_removes_stale_hub_files():
    upload, delete = plan_file_changes(
        ["README.md", "modeling.py", "model.safetensors"],
        [".gitattributes", "README.md", "TransformerModel.py", "source_config/35M.json"],
    )
    assert upload == ["README.md", "model.safetensors", "modeling.py"]
    assert delete == ["TransformerModel.py", "source_config/35M.json"]

def test_release_spec_rejects_mixed_sources(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({
        "repo_id": "a/b", "version": "v1", "card": {"model_name": "b", "example_messages": EXAMPLE_MESSAGES},
        "source": {"run": "runs/x", "stage": "sft2", "weights": "w.pt"},
    }))
    with pytest.raises(ConfigError, match="không trộn"):
        load_release_spec(path)

@pytest.mark.parametrize("messages, error", [
    ([], "không được rỗng"),
    ([{"role": "user"}], "đúng hai khoá"),
    ([{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}], "role phải là"),
    ([{"role": "user", "content": "x"}, {"role": "model", "content": "y"}], "tin nhắn cuối"),
])
def test_release_spec_rejects_bad_example_messages(tmp_path, messages, error):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({
        "repo_id": "a/b", "version": "v1", "card": {"model_name": "b", "example_messages": messages},
        "source": {"weights": "w.pt", "model": "SAI_35M"},
    }))
    with pytest.raises(ConfigError, match=error):
        load_release_spec(path)

def test_shipped_release_specs_are_valid():
    for path in sorted((HF_DIR / "releases").glob("*.yaml")):
        load_release_spec(path)
