"""Dòng lệnh của lab.

    python -m foundation_lab run config/experiments/dev2_35M.yaml
    python -m foundation_lab run config/experiments/dev2_35M.yaml --stages sft1,sft2 --init sft1=model/pretrained_35M.pt
    python -m foundation_lab run --resume runs/dev2_35M/20261021-120000
    python -m foundation_lab validate config/experiments/dev2_35M.yaml --set pipeline.sft1.params.epochs=1
    python -m foundation_lab list
    python -m foundation_lab compare-kernel sdpa eager
    python -m foundation_lab encode pretrain --input data/raw/pretrain_data_*.jsonl --output-dir data/processed --manifest pretrain_manifest.json --model SAI_35M
    python -m foundation_lab encode sft --input data/raw/SFT_1.json --output data/processed/SFT1_data_ids.npz --model SAI_35M
    python -m foundation_lab train-tokenizer --input data/raw/pretrain_data.jsonl --output-dir runs/tokenizers/new
"""

import argparse
import sys
import yaml
import torch
from foundation_lab.core.config import ConfigError
from foundation_lab.tokenizer.train import OFFICIAL_VOCAB_SIZE

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

def cmd_compare_kernel(args):
    from foundation_lab.eval.compare import compare_attention_kernels

    report = compare_attention_kernels(
        args.reference, args.candidate,
        batch_size=args.batch_size, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        seq_len=args.seq_len, head_dim=args.head_dim, dtype=getattr(torch, args.dtype),
        device=args.device, causal=not args.no_causal, repeats=args.repeats,
    )
    for key, value in report.items():
        print(f"{key:<22} {value}")

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

def build_parser() -> argparse.ArgumentParser:
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

    compare = sub.add_parser("compare-kernel", help="so hai attention kernel: output, gradient, tốc độ")
    compare.add_argument("reference")
    compare.add_argument("candidate")
    compare.add_argument("--batch-size", type=int, default=2)
    compare.add_argument("--num-heads", type=int, default=8)
    compare.add_argument("--num-kv-heads", type=int, default=2)
    compare.add_argument("--seq-len", type=int, default=128)
    compare.add_argument("--head-dim", type=int, default=64)
    compare.add_argument("--dtype", default="float32")
    compare.add_argument("--device", default="cpu")
    compare.add_argument("--no-causal", action="store_true")
    compare.add_argument("--repeats", type=int, default=5)
    compare.set_defaults(func=cmd_compare_kernel)

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
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ConfigError as error:
        print(f"Lỗi config: {error}", file=sys.stderr)
        sys.exit(2)
