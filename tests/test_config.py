import pytest
from helpers import TINY_MODEL, tiny_experiment
from SAI.modeling import SAIConfig
from foundation_lab.core.config import ConfigError, apply_overrides, from_dict, load_with_extends
from foundation_lab.core.experiment import resolve_experiment, resolve_experiment_dict, validate_experiment
from foundation_lab.paths import EXPERIMENT_CONFIG_DIR

MODEL_FIELDS = {k: v for k, v in TINY_MODEL.items() if k != "family"}

def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="khoá không hợp lệ"):
        from_dict(SAIConfig, {**MODEL_FIELDS, "d_modle": 32})

def test_missing_key_is_rejected():
    fields = dict(MODEL_FIELDS)
    del fields["ff_dim"]
    with pytest.raises(ConfigError, match="ff_dim: thiếu khoá"):
        from_dict(SAIConfig, fields)

@pytest.mark.parametrize("key,value", [("num_layers", "2"), ("num_layers", True), ("dropout", "abc")])
def test_wrong_type_is_rejected(key, value):
    with pytest.raises(ConfigError):
        from_dict(SAIConfig, {**MODEL_FIELDS, key: value})

def test_float_accepts_int_and_yaml_exponent_string():
    cfg = from_dict(SAIConfig, {**MODEL_FIELDS, "dropout": "5e-2"})
    assert cfg.dropout == 0.05
    assert from_dict(SAIConfig, {**MODEL_FIELDS, "dropout": 0}).dropout == 0.0

def test_extends_deep_merges(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\nb: [1, 2]\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na: {y: 3}\nb: [9]\n")
    assert load_with_extends(tmp_path / "child.yaml") == {"a": {"x": 1, "y": 3}, "b": [9]}

def test_overrides_parse_yaml_values():
    data = apply_overrides({"a": {"b": 1}}, ["a.b=2.5", "a.c=[1, 2]", "d=null"])
    assert data == {"a": {"b": 2.5, "c": [1, 2]}, "d": None}

@pytest.mark.parametrize("size,freeze,workers", [
    ("35M", [0, 1], 1), ("100M", [0, 1, 2], 4), ("375M", [0, 1, 2, 3], 1), ("500M", [0, 1], 1),
])
def test_shipped_dev2_experiments_resolve_to_dev2_values(size, freeze, workers):
    resolved = resolve_experiment(EXPERIMENT_CONFIG_DIR / f"dev2_{size}.yaml")
    spec = validate_experiment(resolved)
    assert spec.name == f"dev2_{size}"
    assert spec.runtime.num_workers == workers and spec.runtime.seed is None
    assert [n for n, s in spec.pipeline.items() if s.enabled] == ["pretrain", "sft1", "sft2"]
    pretrain, sft1, sft2 = (spec.pipeline[n].params for n in ("pretrain", "sft1", "sft2"))
    assert pretrain["learning_rate"] == 2e-4 and pretrain["epochs"] == 1
    assert pretrain["max_microbatch_tokens"] == 16384 and pretrain["batch_size"] == 64
    assert sft1["freeze_blocks"] == freeze == sft2["freeze_blocks"]
    assert (sft1["epochs"], sft2["epochs"]) == (4, 5)
    assert spec.pipeline["sft1"].init == "pretrain.best" and spec.pipeline["sft2"].init == "sft1.best"
    assert sft2["resample_per_epoch"] is True

def test_modular_model_config_extends_official():
    from foundation_lab.paths import MODEL_CONFIG_DIR
    official = load_with_extends(MODEL_CONFIG_DIR / "SAI_35M.yaml")
    modular = load_with_extends(MODEL_CONFIG_DIR / "SAI_modular_35M.yaml")
    assert modular["family"] == "SAI_modular"
    assert {k: v for k, v in modular.items() if k in official and k != "family"} == \
        {k: v for k, v in official.items() if k != "family"}

def test_unknown_recipe_param_is_rejected(tiny_data):
    data = tiny_experiment(tiny_data)
    data["pipeline"]["sft1"]["params"]["learnig_rate"] = 1e-4
    with pytest.raises(ConfigError, match="pipeline.sft1.params"):
        validate_experiment(resolve_experiment_dict(data))

@pytest.mark.parametrize("stage,key,value,message", [
    ("pretrain", "objective", "lm_cee", "objective 'lm_cee' không tồn tại"),
    ("sft1", "objective", "lm_cee", "objective 'lm_cee' không tồn tại"),
    ("sft2", "lr_schedule", "cosine", "lr_schedule 'cosine' không tồn tại"),
    ("sft1", "final_eval", "Best", "phải là một trong"),
])
def test_invalid_recipe_param_value_is_rejected_at_validate(tiny_data, stage, key, value, message):
    data = tiny_experiment(tiny_data)
    data["pipeline"][stage]["params"][key] = value
    with pytest.raises(ConfigError, match=message):
        validate_experiment(resolve_experiment_dict(data))

def test_recipe_rejects_wrong_data_kind(tiny_data):
    data = tiny_experiment(tiny_data)
    data["pipeline"]["sft1"]["data"]["kind"] = "lm"
    with pytest.raises(ConfigError, match="nhận dữ liệu loại"):
        validate_experiment(resolve_experiment_dict(data))

def test_init_must_reference_an_earlier_stage(tiny_data):
    data = tiny_experiment(tiny_data)
    data["pipeline"]["pretrain"]["init"] = "sft2.best"
    with pytest.raises(ConfigError, match="stage đứng trước"):
        validate_experiment(resolve_experiment_dict(data))
