"""Export: dựng thư mục đúng định dạng repo Hugging Face từ một release spec.

Thư mục export chỉ gồm những gì người dùng cần để tải và chạy model:
- ``model.safetensors``, ``config.json``                       trọng số + config (định dạng Transformers)
- ``modeling.py``, ``generate.py``, ``rope.py``...             bản copy mọi file ``.py`` trong ``SAI/`` (code chính thức)
- ``modeling_SAI.py``, ``configuration_SAI.py``,
  ``tokenization_SAI.py``, ``tokenizer_config.json``           bản copy của ``hf_release/adapter/``
- ``tokenizer.model``                                          tokenizer
- ``README.md``                                                model card sinh từ ``hf_release/model_card.md``
và ``release.json`` (manifest: spec, git commit lúc export, sha256 từng file) chỉ giữ trên máy, không upload.

Mọi file code trên Hub đều là bản copy được sinh ra ở đây; không sửa tay trong thư mục export.
"""

import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
import torch
import yaml
from SAI.modeling import SAIConfig, SAIModel
from SAI.version import COMPONENT_VERSIONS, MODEL_VERSION
from foundation_lab.architecture import official_model_config
from foundation_lab.paths import PROJECT_ROOT, RUNS_DIR, TOKENIZER_MODEL, resolve_project_path
from foundation_lab.state.lineage import git_info, sha256_file
from foundation_lab.state.run import RunDir
from hf_release.checkpoint import save_safetensors
from hf_release.spec import ReleaseSpec, load_release_spec

HF_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = HF_DIR / "adapter"
CARD_TEMPLATE = HF_DIR / "model_card.md"
EXPORT_ROOT = RUNS_DIR / "releases"
# Mọi file code của SAI/ (tokenizer.model được copy riêng); __init__.py dùng import tuyệt đối
# ``SAI.*`` nên không chạy được trên Hub và không cần ở đó.
CORE_FILES = {
    path.name: path
    for path in sorted((PROJECT_ROOT / "SAI").glob("*.py"))
    if path.name != "__init__.py"
}
ADAPTER_SUFFIXES = {".py", ".json"}      # tránh copy file rác của hệ điều hành (.DS_Store...) lên Hub
TOKENIZER_FILE = "tokenizer.model"
MANIFEST_FILE = "release.json"
AUTO_MAP = {
    "AutoConfig": "configuration_SAI.SAIConfig",
    "AutoModelForCausalLM": "modeling_SAI.SAIForCausalLM",
}

@dataclass
class ResolvedSource:
    model_config: SAIConfig
    weights: Path
    description: str                 # mô tả nguồn trọng số cho model card
    result: dict | None              # kết quả stage (val/test loss) nếu nguồn là run

def resolve_source(spec: ReleaseSpec) -> ResolvedSource:
    source = spec.source
    if source.run is not None:
        run_dir = RunDir(resolve_project_path(source.run))
        stage_dir = run_dir.path / "stages" / source.stage
        weights = stage_dir / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(f"Không tìm thấy {weights}")
        resolved = run_dir.read_config()
        result = run_dir.read_result(source.stage) if (stage_dir / "done.json").exists() else None
        return ResolvedSource(
            model_config=official_model_config(resolved["model"], "resolved_config.model"),
            weights=weights,
            description=f"run `{source.run}`, stage `{source.stage}`",
            result=result,
        )
    weights = resolve_project_path(source.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Không tìm thấy {weights}")
    return ResolvedSource(
        model_config=official_model_config(source.model, "source.model"),
        weights=weights,
        description=f"file `{source.weights}`",
        result=None,
    )

def _load_weights(model: SAIModel, path: Path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

def _write_config(model_config: SAIConfig, out_dir: Path):
    """Ghi config.json bằng chính class SAIConfig của adapter để khoá khớp với lúc tải."""
    module_spec = importlib.util.spec_from_file_location("hf_release_configuration", ADAPTER_DIR / "configuration_SAI.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    config = module.SAIConfig(**asdict(model_config))
    config.architectures = ["SAIForCausalLM"]
    config.auto_map = dict(AUTO_MAP)
    config.save_pretrained(out_dir)

def _release_table(spec: ReleaseSpec, source: ResolvedSource, num_params: int, git: dict) -> str:
    rows = [
        ("Phiên bản", f"`{spec.version}`"),
        ("Phiên bản mô hình", f"`{MODEL_VERSION}`"),
        ("Số tham số", f"{num_params:,}"),
        ("Nguồn trọng số", source.description),
        ("Git commit lúc export", f"`{git['commit']}`" + (" (có thay đổi chưa commit)" if git["dirty"] else "")),
    ]
    if source.result is not None:
        rows.append(("Val loss (best)", f"{source.result['best_val_loss']:.4f}"))
        rows.append((f"Test loss ({source.result['test_weights']} weights)", f"{source.result['test_loss']:.4f}"))
    lines = ["| | |", "|---|---|"] + [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)

def _format_messages(messages: list[dict[str, str]]) -> str:
    """In ``messages`` thành literal Python cho đoạn code mẫu trong README.

    Chuỗi đi qua ``json.dumps`` để dấu nháy, xuống dòng trong nội dung vẫn là Python hợp lệ;
    ``ensure_ascii=False`` giữ nguyên dấu tiếng Việt.
    """
    lines = ["["]
    for message in messages:
        lines.append("    {")
        for key in ("role", "content"):
            lines.append(f"        {json.dumps(key)}: {json.dumps(message[key], ensure_ascii=False)},")
        lines.append("    },")
    lines.append("]")
    return "\n".join(lines)

def _render_card(spec: ReleaseSpec, source: ResolvedSource, num_params: int, git: dict) -> str:
    card = spec.card
    front_matter = {
        "language": card.language,
        "license": card.license,
        "library_name": "transformers",
        "model_name": card.model_name,
        "pipeline_tag": "text-generation",
        "tags": card.tags,
    }
    if card.base_model:
        front_matter["base_model"] = card.base_model
    template = Template(CARD_TEMPLATE.read_text(encoding="utf-8"))
    return template.substitute(
        front_matter=yaml.safe_dump(front_matter, sort_keys=False, allow_unicode=True).strip(),
        release_table=_release_table(spec, source, num_params, git),
        repo_id=spec.repo_id,
        example_messages=_format_messages(card.example_messages),
    )

def export_release(spec_path, out_dir=None) -> Path:
    spec_path = Path(spec_path)
    spec = load_release_spec(spec_path)
    source = resolve_source(spec)
    out_dir = Path(out_dir) if out_dir is not None else EXPORT_ROOT / spec_path.stem / spec.version
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} đã tồn tại; không ghi đè bản export cũ")

    model = SAIModel(source.model_config)
    _load_weights(model, source.weights)
    num_params = sum(p.numel() for p in model.parameters())
    git = git_info()

    out_dir.mkdir(parents=True)
    save_safetensors(model, out_dir / "model.safetensors")
    _write_config(source.model_config, out_dir)
    for path in sorted(ADAPTER_DIR.iterdir()):
        if path.is_file() and path.suffix in ADAPTER_SUFFIXES:
            shutil.copyfile(path, out_dir / path.name)
    for name, path in CORE_FILES.items():
        shutil.copyfile(path, out_dir / name)
    shutil.copyfile(TOKENIZER_MODEL, out_dir / TOKENIZER_FILE)
    (out_dir / "README.md").write_text(_render_card(spec, source, num_params, git), encoding="utf-8")

    files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    manifest = {
        "repo_id": spec.repo_id,
        "version": spec.version,
        "spec": asdict(spec),
        "model_version": MODEL_VERSION,
        "component_versions": COMPONENT_VERSIONS,
        "num_params": num_params,
        "git": git,
        "files": {name: sha256_file(out_dir / name) for name in files},
    }
    with open(out_dir / MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out_dir
