import ast
import dataclasses
import inspect
import json
import os
import hashlib
import re
from types import SimpleNamespace
import pytest
import torch
import yaml
from safetensors.torch import load_file
from helpers import TINY_MODEL
from SAI.generate import GenerationConfig, generate
from SAI.version import COMPONENT_VERSIONS, MODEL_VERSION
from foundation_lab.catalog import build_model
from foundation_lab.core.config import ConfigError
from hf_release.export import HF_DIR, export_release
from hf_release.push import LOCAL_ONLY_FILES, plan_file_changes
from hf_release.spec import load_release_spec
from hf_release.artifact import MANIFEST_FILE, check_verified, read_json, sha256_file, validate_artifact, write_json

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
def tiny_release(tmp_path, monkeypatch):
    import sentencepiece as spm
    import hf_release.export as exporter

    # Tokenizer nhỏ do test tự tạo, không đọc tokenizer hoặc checkpoint thật của dự án.
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("xin chào, đây là mô hình ngôn ngữ sai của việt nam. user model system bạn tôi trả lời câu hỏi hello world abcdefghijklmnopqrstuvwxyz\n" * 30), encoding="utf-8")
    prefix = str(tmp_path / "tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=prefix, vocab_size=64,
        unk_piece="[UNK]", bos_piece="[BOS]", eos_piece="[EOS]",
        user_defined_symbols=["<|im_start|>", "<|im_end|>"], hard_vocab_limit=False, minloglevel=2,
    )
    monkeypatch.setattr(exporter, "TOKENIZER_MODEL", tmp_path / "tokenizer.model")
    monkeypatch.setattr(exporter, "git_info", lambda: {"commit": "a" * 40, "branch": "test", "dirty": False})
    torch.manual_seed(0)
    tiny_config = {**TINY_MODEL, "vocab_size": spm.SentencePieceProcessor(model_file=prefix + ".model").get_piece_size()}
    model = build_model(tiny_config)
    weights = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), weights)
    spec = {
        "repo_id": "someone/SAI_tiny",
        "source": {"weights": str(weights), "model": tiny_config},
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
        "README.md", MANIFEST_FILE,
    }
    assert {p.name for p in out.iterdir()} == uploaded | {"release.json"}
    assert "release.json" in LOCAL_ONLY_FILES

    config = json.loads((out / "config.json").read_text())
    assert config["model_type"] == "SAI" and config["architectures"] == ["SAIForCausalLM"]
    assert config["auto_map"]["AutoModelForCausalLM"] == "modeling_SAI.SAIForCausalLM"
    assert config["num_kv_heads"] == TINY_MODEL["num_kv_heads"] == config["num_key_value_heads"]

    tensors = load_file(str(out / "model.safetensors"))
    assert "model.lm_head.weight" not in tensors
    assert torch.equal(tensors["model.embed.weight"], model.embed.weight)

    card = (out / "README.md").read_text(encoding="utf-8")
    assert "someone/SAI_tiny" in card and f"`{MODEL_VERSION}`" in card and "model_name: SAI_tiny" in card
    assert f"`{MODEL_VERSION}`" in card
    assert '"content": "Câu hỏi \\"có nháy\\" cho SAI_tiny",' in card
    for name in core_files:
        assert (out / name).read_text() == (HF_DIR.parent / "SAI" / name).read_text()

    manifest = validate_artifact(out)
    assert manifest["version"] == config["sai_version"] == MODEL_VERSION
    assert manifest["component_versions"] == COMPONENT_VERSIONS
    assert json.loads((out / "tokenizer_config.json").read_text())["model_max_length"] == TINY_MODEL["max_seq_len"]

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

@pytest.mark.parametrize("strategy", ["greedy", "beam", "sample"])
def test_chat_messages_match_legacy_calls_and_core_algorithm(tmp_path, tiny_release, strategy):
    """Giao diện mới phải giữ nguyên token, văn bản và thống kê của cách gọi chuỗi cũ."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec_path, core = tiny_release
    folder = export_release(spec_path, out_dir=tmp_path / "export")
    adapted = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
    core.eval()
    config = GenerationConfig(strategy=strategy, max_new_tokens=3, beam_size=2, seed=7)
    system, question = (message["content"] for message in EXAMPLE_MESSAGES)
    expected = generate(core, question, tokenizer.sp_model, system=system, config=config, return_result=True)
    core_parameters = inspect.signature(core.generate_response).parameters
    hf_parameters = inspect.signature(adapted.generate_response).parameters
    assert [(name, p.kind, p.default) for name, p in core_parameters.items()] == [
        (name, p.kind, p.default) for name, p in hf_parameters.items()
    ]

    for model, chat_tokenizer in (
        (core, tokenizer.sp_model), (core, tokenizer), (adapted, tokenizer.sp_model), (adapted, tokenizer),
    ):
        actual = model.generate_response(
            tokenizer=chat_tokenizer, messages=EXAMPLE_MESSAGES, config=config, return_result=True,
        )
        legacy = model.generate_response(question, chat_tokenizer, system=system, config=config, return_result=True)
        assert dataclasses.asdict(actual) == dataclasses.asdict(legacy) == dataclasses.asdict(expected)
        assert model.generate_response(
            tokenizer=chat_tokenizer, messages=EXAMPLE_MESSAGES, config=config,
        ) == expected.text

def test_chat_history_instruction_and_constraint_match_messages(tmp_path, tiny_release):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec_path, core = tiny_release
    folder = export_release(spec_path, out_dir=tmp_path / "export")
    adapted = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
    core.eval()
    system, question = (message["content"] for message in EXAMPLE_MESSAGES)
    history = [{"user": "xin chào", "model": "chào bạn"}]
    messages = [
        EXAMPLE_MESSAGES[0], {"role": "user", "content": history[0]["user"]},
        {"role": "model", "content": history[0]["model"]}, EXAMPLE_MESSAGES[1],
    ]
    config = GenerationConfig(max_new_tokens=2)
    eos = tokenizer.sp_model.eos_id()
    for model in (core, adapted):
        options = {"tokenizer": tokenizer, "config": config, "constraint": lambda tokens: [eos], "return_result": True}
        legacy = model.generate_response(question, instruction=system, history=history, **options)
        actual = model.generate_response(messages=messages, **options)
        assert dataclasses.asdict(actual) == dataclasses.asdict(legacy)
        assert actual.token_ids == [eos] and actual.finish_reason == "stop"

def test_chat_interfaces_reject_mixed_inputs_and_unknown_options(tmp_path, tiny_release):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec_path, core = tiny_release
    folder = export_release(spec_path, out_dir=tmp_path / "export")
    adapted = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
    config = GenerationConfig(strategy="greedy", max_new_tokens=2)
    for model, chat_tokenizer in ((core, tokenizer.sp_model), (adapted, tokenizer)):
        with pytest.raises(ValueError, match="messages or user_input"):
            model.generate_response(
                user_input="xin chào", tokenizer=chat_tokenizer, messages=EXAMPLE_MESSAGES, config=config,
            )
        with pytest.raises(ValueError):
            model.generate_response(messages=EXAMPLE_MESSAGES, config=config)
        with pytest.raises(TypeError, match="unsupported_option"):
            model.generate_response(tokenizer=chat_tokenizer, messages=EXAMPLE_MESSAGES, unsupported_option=True)

def test_documented_chat_calls_work_with_core_and_hf(tmp_path, tiny_release):
    """Chạy biểu thức gọi model thật trong tài liệu để bắt lỗi ví dụ lệch API."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec_path, core = tiny_release
    folder = export_release(spec_path, out_dir=tmp_path / "export")
    adapted = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
    core.eval()
    documents = (
        HF_DIR.parent / "README.md", folder / "README.md",
        HF_DIR.parent / "web/site/src/pages/index.astro",
        HF_DIR.parent / "web/site/src/components/try/TryStatic.astro",
    )
    config = GenerationConfig(strategy="greedy", max_new_tokens=2)
    for path in documents:
        match = re.search(r"model\.generate(?:_response)?\([\s\S]*?\)", path.read_text(encoding="utf-8"))
        assert match is not None, f"Không tìm thấy ví dụ gọi model trong {path}"
        expression = ast.parse(match.group(), mode="eval")
        # Chỉ giới hạn số token cho model tí hon; tên hàm và tham số lấy nguyên từ ví dụ.
        expression.body.keywords.append(ast.keyword(arg="config", value=ast.Name(id="config", ctx=ast.Load())))
        code = compile(ast.fix_missing_locations(expression), str(path), "eval")
        for model, chat_tokenizer in ((core, tokenizer.sp_model), (adapted, tokenizer)):
            answer = eval(code, {
                "model": model, "tokenizer": chat_tokenizer, "messages": EXAMPLE_MESSAGES, "config": config,
            })
            assert isinstance(answer, str)

@pytest.mark.parametrize("family", ["SAI", "SAI_modular"])
def test_web_and_lab_chat_match_core_defaults(tmp_path, tiny_release, family):
    """Web và decoder lab phải giữ đầu ra mặc định khi chuyển sang messages."""
    from foundation_lab.inference.decoders import sai_decoder
    from web.backend.model_loader import LoadedModel
    import sentencepiece as spm

    _, source = tiny_release
    config = dataclasses.asdict(source.config)
    config["max_seq_len"] += GenerationConfig().max_new_tokens
    model = build_model({"family": family, **config}).eval()
    model.load_state_dict(source.state_dict())
    tokenizer = spm.SentencePieceProcessor(model_file=str(tmp_path / "tokenizer.model"))
    question = "xin chào"
    expected = generate(model, question, tokenizer, return_result=True)
    web = LoadedModel(model, tokenizer).generate(question)
    lab = sai_decoder(model, tokenizer, question, GenerationConfig())
    assert dataclasses.asdict(web) == dataclasses.asdict(lab) == dataclasses.asdict(expected)

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
        "repo_id": "a/b", "card": {"model_name": "b", "example_messages": EXAMPLE_MESSAGES},
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
        "repo_id": "a/b", "card": {"model_name": "b", "example_messages": messages},
        "source": {"weights": "w.pt", "model": "SAI_35M"},
    }))
    with pytest.raises(ConfigError, match=error):
        load_release_spec(path)

def test_shipped_release_specs_are_valid():
    for path in sorted((HF_DIR / "releases").glob("*.yaml")):
        load_release_spec(path)


@pytest.fixture
def verified_release(tmp_path, tiny_release):
    from hf_release.verify import verify_export

    spec_path, _ = tiny_release
    folder = export_release(spec_path, out_dir=tmp_path / "export")
    report = verify_export(folder)
    assert report["passed"], report
    return folder


@pytest.mark.parametrize("name", ["modeling.py", "config.json", "tokenizer.model", "README.md"])
def test_changes_after_verify_block_publish(verified_release, name):
    path = verified_release / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="File thay đổi"):
        check_verified(verified_release)


@pytest.mark.parametrize("change", ["add", "delete", "manifest"])
def test_artifact_inventory_and_manifest_are_bound_to_verify(verified_release, change):
    folder = verified_release
    if change == "add":
        (folder / "unexpected.py").write_text("# File ngoài bản đã kiểm chứng\n")
    elif change == "delete":
        (folder / "README.md").unlink()
    else:
        manifest = read_json(folder / MANIFEST_FILE)
        manifest["git"]["commit"] = "b" * 40
        write_json(folder / MANIFEST_FILE, manifest)
    with pytest.raises(ValueError, match="Danh sách file|Manifest thay đổi"):
        check_verified(folder)


def test_config_and_snapshot_version_must_match_even_if_hashes_are_updated(verified_release):
    folder = verified_release
    config = read_json(folder / "config.json")
    config["sai_version"] = "dev3.9.0"
    write_json(folder / "config.json", config)
    manifest = read_json(folder / MANIFEST_FILE)
    manifest["files"]["config.json"] = sha256_file(folder / "config.json")
    write_json(folder / MANIFEST_FILE, manifest)
    with pytest.raises(ValueError, match="Phiên bản code SAI"):
        validate_artifact(folder)


def test_manifest_cannot_omit_a_component_from_snapshot(verified_release):
    folder = verified_release
    config = read_json(folder / "config.json")
    manifest = read_json(folder / MANIFEST_FILE)
    del config["component_versions"]["attention"]
    del manifest["component_versions"]["attention"]
    write_json(folder / "config.json", config)
    manifest["files"]["config.json"] = sha256_file(folder / "config.json")
    write_json(folder / MANIFEST_FILE, manifest)
    with pytest.raises(ValueError, match="Danh sách thành phần"):
        validate_artifact(folder)


class FakeHub:
    """Hub tí hon trong RAM; mô phỏng commit cha, metadata nội dung và lỗi khi gắn tag."""

    def __init__(self, annotated_tags=True):
        self.head = "base"
        self.snapshots = {self.head: {".gitattributes": b"", "old.py": b"# Ban cu\n"}}
        self.tags = {}
        self.commits = 0
        self.fail_tag_once = False
        self.annotated_tags = annotated_tags

    def whoami(self):
        return {"name": "tester"}

    def repo_exists(self, repo_id):
        return True

    def model_info(self, repo_id, revision=None, files_metadata=False):
        revision = self.tags.get(revision, revision) if revision else self.head
        siblings = []
        for name, data in self.snapshots[revision].items():
            lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if name.endswith(".safetensors") else None
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            siblings.append(SimpleNamespace(rfilename=name, size=len(data), lfs=lfs, blob_id=blob))
        return SimpleNamespace(sha=revision, siblings=siblings)

    def list_repo_files(self, repo_id, revision=None):
        return list(self.snapshots[revision or self.head])

    def list_repo_refs(self, repo_id):
        return SimpleNamespace(tags=[SimpleNamespace(
            name=name, target_commit=f"tag-object-{commit}" if self.annotated_tags else commit,
        ) for name, commit in self.tags.items()])

    def create_commit(self, *, repo_id, operations, revision, parent_commit, commit_message, commit_description):
        from huggingface_hub import CommitOperationAdd

        assert revision == "main" and parent_commit == self.head
        files = dict(self.snapshots[self.head])
        for operation in operations:
            if isinstance(operation, CommitOperationAdd):
                from pathlib import Path
                files[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
            else:
                del files[operation.path_in_repo]
        self.commits += 1
        self.head = f"release-{self.commits}"
        self.snapshots[self.head] = files
        return SimpleNamespace(oid=self.head)

    def create_tag(self, repo_id, *, tag, revision, tag_message):
        if self.fail_tag_once:
            self.fail_tag_once = False
            raise ConnectionError("Lỗi mạng giả lập lúc tạo tag")
        assert tag not in self.tags
        self.tags[tag] = revision


@pytest.mark.parametrize("annotated_tags", [False, True])
def test_publish_version_matches_sai_and_tag_and_is_idempotent(verified_release, monkeypatch, annotated_tags):
    from hf_release.push import push_release
    import SAI

    api = FakeHub(annotated_tags=annotated_tags)
    folder = verified_release
    monkeypatch.setattr("builtins.input", lambda _: "someone/SAI_tiny")
    plan = push_release(folder, api=api)
    assert api.commits == 0 and plan["delete"] == ["old.py"]
    record = push_release(folder, confirm=True, api=api)
    config = read_json(folder / "config.json")
    assert record["version"] == config["sai_version"] == SAI.__version__ == MODEL_VERSION
    assert api.tags[MODEL_VERSION] == record["hub_commit"]
    assert record["status"] == "complete"
    assert "release.json" not in api.snapshots[api.head]
    assert "verify.json" not in api.snapshots[api.head]
    push_release(folder, confirm=True, api=api)
    assert api.commits == 1


def test_existing_tag_with_different_content_cannot_be_overwritten(verified_release):
    from hf_release.push import push_release

    api = FakeHub()
    api.tags[MODEL_VERSION] = api.head
    with pytest.raises(RuntimeError, match="Danh sách file trên Hub khác"):
        push_release(verified_release, confirm=True, api=api)
    assert api.commits == 0 and api.tags[MODEL_VERSION] == "base"


def test_tag_failure_resumes_without_reupload(verified_release, monkeypatch):
    from hf_release.push import push_release

    api = FakeHub()
    folder = verified_release
    monkeypatch.setattr("builtins.input", lambda _: "someone/SAI_tiny")
    push_release(folder, api=api)
    api.fail_tag_once = True
    with pytest.raises(ConnectionError, match="giả lập"):
        push_release(folder, confirm=True, api=api)
    assert read_json(folder / "push.json")["status"] == "uploaded"
    push_release(folder, confirm=True, api=api)
    assert api.commits == 1 and api.tags[MODEL_VERSION] == api.head


def test_remote_change_requires_reviewing_new_plan(verified_release, monkeypatch):
    from hf_release.push import push_release

    api = FakeHub()
    monkeypatch.setattr("builtins.input", lambda _: "someone/SAI_tiny")
    push_release(verified_release, api=api)
    api.snapshots["concurrent"] = dict(api.snapshots[api.head])
    api.head = "concurrent"
    with pytest.raises(RuntimeError, match="kế hoạch đã thay đổi"):
        push_release(verified_release, confirm=True, api=api)
    assert api.commits == 0


def test_dirty_source_cannot_be_published(verified_release):
    from hf_release.push import push_release

    folder = verified_release
    manifest = read_json(folder / MANIFEST_FILE)
    manifest["git"]["dirty"] = True
    write_json(folder / MANIFEST_FILE, manifest)
    report = read_json(folder / "verify.json")
    report["manifest_sha256"] = sha256_file(folder / MANIFEST_FILE)
    write_json(folder / "verify.json", report)
    with pytest.raises(RuntimeError, match="Git commit sạch"):
        push_release(folder, confirm=True, api=FakeHub())


@pytest.mark.parametrize("beams", [1, 3])
def test_hf_batch_generation_matches_single_sequences(tmp_path, tiny_release, beams):
    from transformers import AutoModelForCausalLM

    folder = export_release(tiny_release[0], out_dir=tmp_path / "export")
    model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True).eval()
    ids = torch.tensor([[5, 6, 7, 8], [0, 9, 10, 11]])
    mask = ids.ne(0).long()
    options = {"max_new_tokens": 3, "min_new_tokens": 3, "do_sample": False, "num_beams": beams}
    with torch.no_grad():
        batched = model.generate(input_ids=ids, attention_mask=mask, **options)
        uncached = model.generate(input_ids=ids, attention_mask=mask, use_cache=False, **options)
        assert torch.equal(batched, uncached)
        for index in range(2):
            tokens = ids[index][mask[index].bool()].unsqueeze(0)
            single = model.generate(input_ids=tokens, **options)
            assert torch.equal(single[0, tokens.shape[1]:], batched[index, ids.shape[1]:])


def test_hf_save_load_and_runtime_buffers_preserve_dtype(tmp_path, tiny_release):
    from transformers import AutoModelForCausalLM

    folder = export_release(tiny_release[0], out_dir=tmp_path / "export")
    model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True, dtype=torch.float64).eval()
    ids = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        expected = model(ids).logits
    assert model.model.rope.cos_cached.dtype == model.get_input_embeddings().weight.dtype == torch.float64
    saved = tmp_path / "saved"
    model.save_pretrained(saved)
    loaded = AutoModelForCausalLM.from_pretrained(saved, trust_remote_code=True, dtype=torch.float64).eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, loaded(ids).logits, atol=0, rtol=0)
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight


def test_hf_forward_output_loss_and_rejected_options(tmp_path, tiny_release):
    from transformers import AutoModelForCausalLM, DynamicCache
    from transformers.modeling_outputs import CausalLMOutputWithPast

    folder = export_release(tiny_release[0], out_dir=tmp_path / "export")
    model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True).eval()
    ids = torch.tensor([[5, 6, 7]])
    result = model(ids, labels=ids, use_cache=False)
    assert isinstance(result, CausalLMOutputWithPast)
    expected = torch.nn.functional.cross_entropy(result.logits[:, :-1].reshape(-1, result.logits.shape[-1]), ids[:, 1:].reshape(-1))
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    assert model.get_input_embeddings().weight.grad is not None
    assert isinstance(model(ids, return_dict=False), tuple)
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        cached = model(ids, past_key_values=cache, use_cache=True)
        assert cached.past_key_values is cache and cache.get_seq_length() == ids.shape[1]
        extended = model(torch.tensor([[8]]), past_key_values=cache, use_cache=True)
        assert extended.past_key_values is cache and cache.get_seq_length() == ids.shape[1] + 1
        expected_extended = model(torch.cat((ids, torch.tensor([[8]])), dim=1), use_cache=False)
        torch.testing.assert_close(extended.logits, expected_extended.logits[:, -1:], atol=1e-5, rtol=1e-5)
    with pytest.raises(TypeError, match="chưa hỗ trợ"):
        model(ids, unsupported_option=True)
    with pytest.raises(NotImplementedError):
        model(ids, output_attentions=True)


def test_release_spec_cannot_override_sai_version(tmp_path, tiny_release):
    path = tiny_release[0]
    spec = yaml.safe_load(path.read_text())
    spec["version"] = "dev3.9.0"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ConfigError, match="version"):
        load_release_spec(path)
