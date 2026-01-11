import builtins
import sys
from types import ModuleType, SimpleNamespace

from omegaconf import OmegaConf

import dialogue_summarization.loggers.log_hydra as log_hydra

_sentinel = object()

def _install_fake_comet(monkeypatch, mocker, exp=_sentinel):
    if exp is _sentinel:
        exp = mocker.Mock()
        
    comet_config = ModuleType("comet_ml.config")
    comet_config.get_global_experiment = mocker.Mock(return_value=exp)
    comet_ml = ModuleType("comet_ml")
    comet_ml.config = comet_config
    monkeypatch.setitem(sys.modules, "comet_ml", comet_ml)
    monkeypatch.setitem(sys.modules, "comet_ml.config", comet_config)
    return exp, comet_config


def test_on_train_begin_logs_hydra_outputs_and_cfg(tmp_path, mocker, monkeypatch):
    run_dir = tmp_path / "run-123"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text("a: 1\n", encoding="utf-8")

    cfg = OmegaConf.create({"trainer": {"lr": 0.1}})
    mocker.patch.object(
        log_hydra.HydraConfig,
        "get",
        return_value=SimpleNamespace(runtime=SimpleNamespace(output_dir=str(run_dir))),
    )

    exp, _ = _install_fake_comet(monkeypatch, mocker)
    callback = log_hydra.HydraOutputsToCometCallback(cfg=cfg)

    control = object()
    state = SimpleNamespace(is_world_process_zero=True)
    result = callback.on_train_begin(None, state, control)

    assert result is control
    assert callback._logged is True

    exp.log_asset.assert_called_once()
    _, kwargs = exp.log_asset.call_args
    assert kwargs["file_name"] == "hydra_outputs__run-123.zip"
    exp.log_parameter.assert_called_once_with("hydra/output_dir", str(run_dir))

    exp.log_text.assert_called_once()
    text_args, text_kwargs = exp.log_text.call_args
    assert text_args[0] == OmegaConf.to_yaml(cfg)
    assert text_kwargs["metadata"] == {"kind": "hydra_effective_cfg"}


def test_on_train_begin_skips_when_hydra_dir_missing(tmp_path, mocker, monkeypatch):
    run_dir = tmp_path / "run-456"
    run_dir.mkdir()
    mocker.patch.object(
        log_hydra.HydraConfig,
        "get",
        return_value=SimpleNamespace(runtime=SimpleNamespace(output_dir=str(run_dir))),
    )

    exp, _ = _install_fake_comet(monkeypatch, mocker)
    callback = log_hydra.HydraOutputsToCometCallback()

    control = object()
    state = SimpleNamespace(is_world_process_zero=True)
    result = callback.on_train_begin(None, state, control)

    assert result is control
    assert callback._logged is False
    exp.log_asset.assert_not_called()
    exp.log_parameter.assert_not_called()
    exp.log_text.assert_not_called()


def test_on_train_begin_skips_on_non_zero_process(mocker):
    callback = log_hydra.HydraOutputsToCometCallback()
    control = object()
    state = SimpleNamespace(is_world_process_zero=False)

    mocker.patch.object(
        log_hydra.HydraConfig,
        "get",
        side_effect=AssertionError("HydraConfig.get should not be called"),
    )

    result = callback.on_train_begin(None, state, control)

    assert result is control
    assert callback._logged is False


def test_on_train_begin_skips_when_comet_missing(monkeypatch, mocker):
    callback = log_hydra.HydraOutputsToCometCallback()
    control = object()
    state = SimpleNamespace(is_world_process_zero=True)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "comet_ml" or name.startswith("comet_ml."):
            raise ImportError("comet_ml not available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = callback.on_train_begin(None, state, control)

    assert result is control
    assert callback._logged is False


def test_on_train_begin_skips_when_no_global_experiment(monkeypatch, mocker):
    callback = log_hydra.HydraOutputsToCometCallback()
    control = object()
    state = SimpleNamespace(is_world_process_zero=True)

    mocker.patch.object(
        log_hydra.HydraConfig,
        "get",
        side_effect=AssertionError("HydraConfig.get should not be called"),
    )

    _install_fake_comet(monkeypatch, mocker, exp=None)

    result = callback.on_train_begin(None, state, control)

    assert result is control
    assert callback._logged is False
