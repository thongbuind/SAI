import pytest
from SAI.version import COMPONENT_VERSIONS, MODEL_VERSION, latest_version, parse_version

def test_every_component_is_in_the_same_dev_stage_as_the_model():
    model_stage = parse_version(MODEL_VERSION)[0]
    for name, version in COMPONENT_VERSIONS.items():
        assert parse_version(version)[0] == model_stage, f"{name}: {version}"

def test_model_version_is_latest_component_version():
    assert MODEL_VERSION in COMPONENT_VERSIONS.values()
    assert all(parse_version(MODEL_VERSION) >= parse_version(v) for v in COMPONENT_VERSIONS.values())

def test_retraining_bump_outranks_any_compatible_bump():
    assert latest_version(["dev3.0.12", "dev3.1.0", "dev3.0.9"]) == "dev3.1.0"
    assert latest_version(["dev3.1.2", "dev3.1.10"]) == "dev3.1.10"

@pytest.mark.parametrize("version", ["3.0.0", "dev3.0", "dev3.0.0-rc", "v3.0.0", "dev3.a.0"])
def test_malformed_version_is_rejected(version):
    with pytest.raises(ValueError, match="devX.Y.Z"):
        parse_version(version)
