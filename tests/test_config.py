import tomllib

import pytest

from roboarena.config import Config, dump_toml, load_config
from roboarena.core.errors import ConfigError

from .conftest import QUICK


def test_defaults():
    cfg = load_config()
    assert cfg == Config()
    assert cfg.task.name == "pusht"


def test_file_then_overrides(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('seed = 3\n[task]\nname = "toy"\n[eval]\nn_inits = 4\n')
    cfg = load_config(path, ["eval.n_inits=7", "eval.suite=held-out", "seed=9"])
    assert (cfg.seed, cfg.task.name, cfg.eval.n_inits, cfg.eval.suite) == (9, "toy", 7, "held-out")
    assert cfg.eval.batch == Config().eval.batch  # 没提到的项保持默认


@pytest.mark.parametrize(
    "content, message",
    [
        ("sed = 1\n", "未知的配置项：sed"),
        ("[eval]\nmax_step = 5\n", "未知的配置项：eval.max_step"),
        ('[eval]\nn_inits = "many"\n', "eval.n_inits 应为 int"),
        ("[eval]\nn_inits = true\n", "eval.n_inits 应为 int"),
        ("eval = 3\n", "eval 应为一个表"),
        ("seed = \n", "不是合法的 TOML"),
    ],
)
def test_bad_files_are_rejected(tmp_path, content, message):
    path = tmp_path / "c.toml"
    path.write_text(content)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_bad_overrides_are_rejected():
    with pytest.raises(ConfigError, match="未知的配置项：eval.nope"):
        load_config(overrides=["eval.nope=1"])
    with pytest.raises(ConfigError, match="键=值"):
        load_config(overrides=["eval.n_inits"])
    with pytest.raises(ConfigError, match="找不到配置文件"):
        load_config("no-such-file.toml")


def test_toml_round_trip():
    cfg = load_config(overrides=['workspace.root="a \\"quoted\\" dir"', "eval.max_steps=12"])
    again = Config()
    from roboarena.config import _apply

    _apply(again, tomllib.loads(cfg.to_toml()))
    assert again == cfg


def test_dump_toml_nested_tables():
    text = dump_toml({"a": 1, "t": {"b": 2.5, "u": {"c": [1, "x"], "d": False}}})
    assert tomllib.loads(text) == {"a": 1, "t": {"b": 2.5, "u": {"c": [1, "x"], "d": False}}}


def test_digest_changes_with_content():
    assert load_config().digest() == load_config().digest()
    assert load_config().digest() != load_config(overrides=["seed=1"]).digest()


def test_quick_env_selects_quick_config(monkeypatch):
    monkeypatch.setenv("ROBOARENA_QUICK", "1")
    monkeypatch.setenv("ROBOARENA_CONFIG_DIR", str(QUICK.parent))
    assert load_config().task.name == "toy"
    monkeypatch.setenv("ROBOARENA_QUICK", "0")
    assert load_config().task.name == "pusht"
