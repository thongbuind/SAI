"""Chạy pipeline các stage của một experiment.

Mỗi stage tương ứng một lần gọi script của dev_2 (``pretrain.py --phase pretrain``, ``finetune.py --phase sft1``...):
1. Đặt seed nếu ``runtime.seed`` có giá trị (stage thứ i dùng ``seed + i``).
2. Dựng model mới từ registry (tiêu thụ RNG khởi tạo như khi dev_2 khởi động script), compile nếu bật.
3. Nạp trọng số khởi tạo theo ``init`` (bỏ qua khi resume từ checkpoint của chính stage).
4. Chạy recipe; kết quả ghi vào ``stages/<stage>/done.json``.
"""

import torch
from foundation_lab import catalog
from foundation_lab.core.config import ConfigError, from_dict
from foundation_lab.core.experiment import parse_init_ref, validate_experiment
from foundation_lab.core.log import banner, log_progress
from foundation_lab.core.seed import seed_everything
from foundation_lab.engine.runtime import Runtime
from foundation_lab.paths import resolve_project_path
from foundation_lab.recipes.base import StageContext
from foundation_lab.state.checkpoint import load_model_weights
from foundation_lab.state.metrics import MetricsLogger

def resolve_init_path(stage_name, stage, spec, run_dir, init_overrides):
    """Đường dẫn trọng số khởi tạo cho stage, hoặc None (khởi tạo mới)."""
    if stage_name in init_overrides:
        return resolve_project_path(init_overrides[stage_name])
    ref = parse_init_ref(stage.init, spec.pipeline)
    if ref is None:
        return None
    kind, value = ref
    if kind == "stage":
        path = run_dir.best_weights(value)
        if not path.exists():
            raise FileNotFoundError(
                f"Stage '{stage_name}' cần {stage.init} nhưng chưa có {path}. "
                f"Chạy stage '{value}' trước hoặc truyền --init {stage_name}=<file .pt>"
            )
        return path
    return resolve_project_path(value)

def run_experiment(resolved: dict, run_dir, *, stages=None, init_overrides=None, resume=False) -> dict:
    """Chạy các stage được bật (hoặc chỉ ``stages`` nếu truyền vào) theo thứ tự trong pipeline."""
    spec = validate_experiment(resolved)
    init_overrides = dict(init_overrides or {})
    unknown = sorted((set(stages or []) | set(init_overrides)) - set(spec.pipeline))
    if unknown:
        raise ConfigError(f"Stage không có trong pipeline: {unknown}; có: {list(spec.pipeline)}")

    runtime = Runtime(spec.runtime)
    metrics = MetricsLogger(run_dir.metrics_path)
    results = {}
    for index, (name, stage) in enumerate(spec.pipeline.items()):
        if stages is not None:
            if name not in stages:
                continue
        elif not stage.enabled:
            continue
        if resume and run_dir.is_done(name):
            log_progress(f"[{name}] đã xong trong run này, bỏ qua")
            results[name] = run_dir.read_result(name)
            continue

        banner(f"STAGE {name.upper()} (recipe={stage.recipe})")
        if spec.runtime.seed is not None:
            seed_everything(spec.runtime.seed + index)

        model = catalog.build_model(spec.model).to(runtime.device)
        features = runtime.compile_features(model)

        resume_checkpoint = run_dir.last_checkpoint(name) if resume else None
        if resume_checkpoint is not None and resume_checkpoint.exists():
            log_progress(f"[{name}] resume từ {resume_checkpoint}")
        else:
            resume_checkpoint = None
            init_path = resolve_init_path(name, stage, spec, run_dir, init_overrides)
            if init_path is not None:
                log_progress(f"[{name}] khởi tạo trọng số từ {init_path}")
                load_model_weights(model, init_path, runtime.device)

        recipe_cls = catalog.RECIPES.get(stage.recipe)
        recipe_cfg = from_dict(recipe_cls.config_cls, stage.params, f"pipeline.{name}.params")
        ctx = StageContext(
            name=name,
            index=index,
            model=model,
            features=features,
            runtime=runtime,
            data=catalog.build_data_source(stage.data, f"pipeline.{name}.data"),
            stage_dir=run_dir.stage_dir(name),
            metrics=metrics,
            num_workers=spec.runtime.num_workers,
            resume_checkpoint=resume_checkpoint,
        )
        result = recipe_cls(recipe_cfg, ctx).run()
        result["best_weights"] = str(ctx.best_weights_path)
        run_dir.mark_done(name, result)
        results[name] = result
        log_progress(f"[{name}] xong: {result}")

        del ctx, features, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results

def stage_data_files(spec) -> dict:
    """File dữ liệu mà mỗi stage đọc — dùng cho lineage."""
    return {
        name: catalog.build_data_source(stage.data).read_files()
        for name, stage in spec.pipeline.items()
        if stage.enabled
    }
