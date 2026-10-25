"""Dòng lệnh của lab.

    python -m foundation_lab run foundation_lab/core/configs/SAI_375M.yaml
    python -m foundation_lab run foundation_lab/core/configs/SAI_375M.yaml --stages sft1,sft2 --init sft1=model/pretrained_375M.pt
    python -m foundation_lab run --resume runs/SAI_375M/20261021-120000
    python -m foundation_lab validate foundation_lab/core/configs/SAI_375M.yaml --set pipeline.sft1.params.epochs=1
    python -m foundation_lab list
    python -m foundation_lab encode pretrain --input data/raw/pretrain_data_*.jsonl --output-dir data/processed --manifest pretrain_manifest.json --model SAI_375M
    python -m foundation_lab encode sft --input data/raw/SFT_1.json --output data/processed/SFT1_data_ids.npz --model SAI_375M
    python -m foundation_lab train-tokenizer --input data/raw/pretrain_data.jsonl --output-dir runs/tokenizers/new
    python -m foundation_lab compare-kernel --set candidate=eager     # so với kernel mặc định, trên shape của cả ba cỡ model
    python -m foundation_lab compare-kernel --set candidate=eager --set "models=[SAI_modular_35M]"
    python -m foundation_lab compare-objective --set candidate=lm_ce --set candidate_chunk_size=1024   # mốc: stage pretrain
    python -m foundation_lab compare-objective --set candidate=lm_ce --set stage=sft1 --set "experiments=[SAI_35M]"
    python -m foundation_lab infer foundation_lab/inference/configs/compare_375M.yaml --set device=cpu
    python -m foundation_lab infer - < spec.json       # spec đọc từ stdin (web lab gửi spec theo cách này)
    python -m foundation_lab infer runs/infer/20261025-053314-e398/resolved_config.yaml   # chạy lại một run
    python -m foundation_lab describe                  # JSON: phòng, registry, schema spec của từng lệnh, model, experiment, trọng số

Tác vụ (``compare-kernel``, ``compare-objective``, ``infer``: xem ``foundation_lab.core.tasks``) nhận spec là file,
``-`` hoặc bỏ trống (chỉ dùng ``--set``), và ghi kết quả vào ``runs/<lệnh>/<run_id>/``.

Mã thoát: 0 xong; 2 lỗi config (chưa chạy gì); 1 lỗi khi chạy.
"""

import argparse
import json
import re
import sys
import yaml
from foundation_lab.core.config import ConfigError
from foundation_lab.tokenizer.train import OFFICIAL_VOCAB_SIZE

# Tăng khi đổi ý nghĩa hoặc xoá khoá trong output của "describe".
DESCRIBE_FORMAT_VERSION = 1

def _parse_init(values) -> dict:
    result = {}
    for text in values or []:
        if "=" not in text:
            raise ConfigError(f"--init '{text}' phải có dạng stage=duong_dan.pt")
        stage, path = text.split("=", 1)
        result[stage.strip()] = path.strip()
    return result

def cmd_run(args):
    from foundation_lab.core.config import config_hash
    from foundation_lab.core.experiment import resolve_experiment, validate_experiment
    from foundation_lab.pipeline import run_experiment, stage_data_files
    from foundation_lab.state.lineage import collect_lineage, git_info
    from foundation_lab.state.run import RunDir

    stages = args.stages.split(",") if args.stages else None
    init_overrides = _parse_init(args.init)
    if args.resume:
        if args.experiment or args.set:
            raise ConfigError("--resume dùng resolved_config.yaml của run; không truyền experiment hay --set")
        run_dir = RunDir(args.resume)
        resolved = run_dir.read_config()
        validate_experiment(resolved)
    else:
        if not args.experiment:
            raise ConfigError("Cần đường dẫn experiment hoặc --resume")
        resolved = resolve_experiment(args.experiment, args.set)
        spec = validate_experiment(resolved)
        run_dir = RunDir.create(spec.name, root=args.runs_root, run_id=args.run_id)
        digest = run_dir.write_config(resolved)
        run_dir.write_json("lineage.json", collect_lineage(digest, stage_data_files(spec)))
        print(f"Run dir: {run_dir.path}  (config_hash {digest[:12]})")
    # lineage.json chỉ ghi lúc tạo run; ghi git của từng lần gọi để biết code nào đã chạy khi --resume.
    run_dir.append_invocation({
        "argv": sys.argv, "stages": stages, "init": init_overrides, "resume": bool(args.resume),
        "config_hash": config_hash(resolved), "git": git_info(),
    })
    run_experiment(resolved, run_dir, stages=stages, init_overrides=init_overrides, resume=bool(args.resume))

def cmd_validate(args):
    from foundation_lab.core.experiment import resolve_experiment, validate_experiment

    resolved = resolve_experiment(args.experiment, args.set)
    validate_experiment(resolved)
    yaml.safe_dump(resolved, sys.stdout, sort_keys=False, allow_unicode=True)
    print("# OK: config hợp lệ", file=sys.stderr)

def cmd_list(_args):
    from foundation_lab.catalog import DEPARTMENTS

    for department, registries in DEPARTMENTS.items():
        print(f"[{department}]")
        for registry in registries:
            print(f"  {registry.kind:<18} {', '.join(registry.names())}")

def cmd_encode(args):
    from foundation_lab.architecture import resolve_model_config
    from foundation_lab.core.config import load_with_extends
    from foundation_lab.paths import MODEL_CONFIG_DIR, TOKENIZER_MODEL, resolve_project_path
    from foundation_lab.tokenizer import encode_file, encode_shards

    _, model_config = resolve_model_config(load_with_extends(MODEL_CONFIG_DIR / f"{args.model}.yaml"))
    tokenizer = resolve_project_path(args.tokenizer) if args.tokenizer else TOKENIZER_MODEL
    inputs = [resolve_project_path(path) for path in args.input]
    if args.output:
        if len(inputs) != 1:
            raise ConfigError("--output chỉ dùng với đúng một --input; nhiều file thì dùng --output-dir và --manifest")
        encode_file(args.kind, inputs[0], resolve_project_path(args.output), tokenizer, model_config.max_seq_len)
        return
    if not args.manifest:
        raise ConfigError("--output-dir cần đi kèm --manifest (tên file manifest ghi trong thư mục đó)")
    encode_shards(
        args.kind, inputs, resolve_project_path(args.output_dir), args.manifest, tokenizer, model_config.max_seq_len,
    )

def cmd_train_tokenizer(args):
    from foundation_lab.paths import resolve_project_path
    from foundation_lab.tokenizer import train_tokenizer

    model_path = train_tokenizer(resolve_project_path(args.input), resolve_project_path(args.output_dir), args.vocab_size)
    print(f"Tokenizer mới: {model_path}")

def _read_spec(source: str | None) -> dict:
    """Spec từ file YAML/JSON (có ``extends``), từ stdin khi ``source`` là ``-``, rỗng khi không truyền (chỉ dùng ``--set``)."""
    from foundation_lab.core.config import load_with_extends

    if source is None:
        return {}
    if source != "-":
        return load_with_extends(source)
    data = yaml.safe_load(sys.stdin)
    if not isinstance(data, dict):
        raise ConfigError("spec đọc từ stdin: gốc phải là mapping")
    return data

def cmd_task(args):
    """Chạy một tác vụ theo hợp đồng chung (``foundation_lab.core.tasks``); tên tác vụ là tên lệnh con."""
    from foundation_lab.catalog import TASKS
    from foundation_lab.state.events import EventLog
    from foundation_lab.state.lineage import collect_lineage, git_info
    from foundation_lab.state.run import RunDir

    task = TASKS.get(args.command)
    resolved = task.resolve(_read_spec(args.spec), args.set)
    spec = task.validate(resolved)
    run_dir = RunDir.create(args.command, root=args.runs_root, run_id=args.run_id)
    digest = run_dir.write_config(resolved)
    run_dir.write_json("lineage.json", collect_lineage(digest, task.inputs(spec)))
    run_dir.append_invocation({"argv": sys.argv, "config_hash": digest, "git": git_info()})
    print(f"Run dir: {run_dir.path}  (config_hash {digest[:12]})")

    events = EventLog(run_dir.events_path)
    try:
        result = task.run(spec, run_dir, events)
    except Exception as error:
        # Ghi lý do vào run rồi để lỗi đi tiếp: thư mục run tự cho biết lượt chạy đã thất bại.
        events.emit("failed", error=f"{type(error).__name__}: {error}")
        raise
    for metric in result.metrics:
        print(f"{metric.group:<20} {metric.name:<24} {metric.value:.4g} {metric.unit}")

def _size_order(name: str) -> list:
    """Khoá sắp tên model theo số trong tên (SAI_35M trước SAI_100M) thay cho thứ tự chữ cái."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]

def cmd_describe(_args):
    from foundation_lab import __version__
    from foundation_lab.catalog import DEPARTMENTS, TASKS
    from foundation_lab.core.config import describe_dataclass, load_with_extends
    from foundation_lab.core.experiment import StageSpec, resolve_config_ref
    from foundation_lab.paths import EXPERIMENT_CONFIG_DIR, INFERENCE_BENCHMARK, MODEL_CONFIG_DIRS, PROJECT_ROOT
    from foundation_lab.state.checkpoint import find_weights, weights_by_model

    # Web hiện model theo thứ tự này, nên spec mặc định của web có models cùng thứ tự với mặc định của CLI.
    model_names = sorted((path.stem for directory in MODEL_CONFIG_DIRS for path in directory.glob("*.yaml")), key=_size_order)
    model_configs = {name: resolve_config_ref(name, MODEL_CONFIG_DIRS, "model") for name in model_names}
    experiment_paths = sorted(EXPERIMENT_CONFIG_DIR.glob("*.yaml"), key=lambda path: _size_order(path.stem))
    experiment_configs = {path.stem: load_with_extends(path) for path in experiment_paths}
    description = {
        "format_version": DESCRIBE_FORMAT_VERSION,
        "lab_version": __version__,
        "departments": {
            department: [{"kind": registry.kind, "names": registry.names()} for registry in registries]
            for department, registries in DEPARTMENTS.items()
        },
        "commands": {
            name: {"department": task.department, "spec": describe_dataclass(task.spec_cls)}
            for name, task in TASKS.items()
        },
        "models": model_names,
        # Tên model → họ mô hình, để web lọc model theo họ (vd. chỉ bản SAI_modular chọn attention kernel qua config).
        "model_families": {name: config["family"] for name, config in model_configs.items()},
        # Experiment chạy được → các stage đang bật, theo thứ tự pipeline. Bỏ file gốc không chọn model (base.yaml).
        "experiments": {
            name: [stage for stage, config in data["pipeline"].items() if config.get("enabled", StageSpec.enabled)]
            for name, data in experiment_configs.items() if "model" in data
        },
        "weights": find_weights(PROJECT_ROOT),
        # Model → trọng số nạp được vào model đó; chỉ có model đang có trọng số trên máy.
        "checkpoints": weights_by_model(PROJECT_ROOT, model_configs),
        "benchmark_path": INFERENCE_BENCHMARK.relative_to(PROJECT_ROOT).as_posix(),
    }
    json.dump(description, sys.stdout, ensure_ascii=False, indent=2)
    print()

def build_parser() -> argparse.ArgumentParser:
    from foundation_lab.catalog import TASKS

    parser = argparse.ArgumentParser(prog="python -m foundation_lab", description="SAI training lab")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="chạy pipeline của một experiment")
    run.add_argument("experiment", nargs="?", help="file experiment YAML")
    run.add_argument("--resume", help="thư mục run để chạy tiếp (bỏ qua stage đã xong)")
    run.add_argument("--stages", help="chỉ chạy các stage này, cách nhau bởi dấu phẩy")
    run.add_argument("--init", action="append", help="stage=file.pt: ghi đè trọng số khởi tạo của stage")
    run.add_argument("--set", action="append", help="override khoa.con=gia_tri (YAML)")
    run.add_argument("--runs-root", help="thư mục chứa các run (mặc định runs/)")
    run.add_argument("--run-id", help="tên thư mục run (mặc định theo thời gian)")
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser("validate", help="kiểm tra và in config đã resolve")
    validate.add_argument("experiment")
    validate.add_argument("--set", action="append")
    validate.set_defaults(func=cmd_validate)

    listing = sub.add_parser("list", help="liệt kê implementation đã đăng ký theo phòng ban")
    listing.set_defaults(func=cmd_list)

    encode = sub.add_parser("encode", help="encode văn bản thô thành .npz bằng tokenizer chính thức")
    encode.add_argument("kind", help="kiểu encode đã đăng ký (xem: python -m foundation_lab list)")
    encode.add_argument("--input", nargs="+", required=True, help="file thô (JSONL cho pretrain, JSON cho sft)")
    destination = encode.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", help="file .npz đầu ra (một --input)")
    destination.add_argument("--output-dir", help="thư mục ghi mỗi --input thành một shard <tên>_ids.npz")
    encode.add_argument("--manifest", help="tên manifest ghi trong --output-dir")
    encode.add_argument("--model", required=True, help="config trong config/models/ để lấy max_seq_len")
    encode.add_argument("--tokenizer", help="file tokenizer .model (mặc định SAI/tokenizer.model)")
    encode.set_defaults(func=cmd_encode)

    tokenizer = sub.add_parser("train-tokenizer", help="train tokenizer SentencePiece mới vào thư mục mới")
    tokenizer.add_argument("--input", required=True, help="file JSONL {\"text\": ...}")
    tokenizer.add_argument("--output-dir", required=True, help="thư mục chưa tồn tại")
    tokenizer.add_argument("--vocab-size", type=int, default=OFFICIAL_VOCAB_SIZE)
    tokenizer.set_defaults(func=cmd_train_tokenizer)

    for name, task in TASKS.items():
        command = sub.add_parser(name, help=task.help)
        command.add_argument("spec", nargs="?", help="file spec YAML/JSON, - để đọc từ stdin, bỏ trống để chỉ dùng --set")
        command.add_argument("--set", action="append", help="override khoa.con=gia_tri (YAML)")
        command.add_argument("--runs-root", help="thư mục chứa các run (mặc định runs/)")
        command.add_argument("--run-id", help="tên thư mục run (mặc định theo thời gian)")
        command.set_defaults(func=cmd_task)

    describe = sub.add_parser("describe", help="in JSON mô tả lab cho công cụ bên ngoài (web lab)")
    describe.set_defaults(func=cmd_describe)
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ConfigError as error:
        print(f"Lỗi config: {error}", file=sys.stderr)
        sys.exit(2)
