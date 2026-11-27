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


def test_list_values(tmp_path):
    cfg = load_config(
        overrides=["policy.snapshots=[10, 20]", "zoo.exec_noise=[0.1, 1]", 'data.rollouts=["zoo"]']
    )
    assert (
        cfg.policy.snapshots == [10, 20] and cfg.zoo.exec_noise == [0.1, 1.0] and cfg.data.rollouts == ["zoo"]
    )
    assert all(type(x) is float for x in cfg.zoo.exec_noise)
    with pytest.raises(ConfigError, match=r"policy.snapshots\[1\] 应为 int"):
        load_config(overrides=['policy.snapshots=[1, "x"]'])
    with pytest.raises(ConfigError, match="应为列表"):
        load_config(overrides=["policy.snapshots=3"])


def test_toml_round_trip():
    cfg = load_config(
        overrides=['workspace.root="a \\"quoted\\" dir"', "eval.max_steps=12", "policy.snapshots=[5, 9]"]
    )
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


def test_llm_models_are_a_table_of_tables(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        '[llm]\ndefault = "big"\nfallback = "small"\nretries = 0\n'
        '[llm.models.big]\nbase_url = "http://127.0.0.1:1/v1"\nmodel = "m-big"\nprice_in = 2\n'
        '[llm.models.small]\nbase_url = "http://127.0.0.1:1/v1"\nmodel = "m-small"\n'
    )
    cfg = load_config(path, ['llm.models.small.model="m-tiny"', 'llm.models.extra.model="m-x"'])
    assert (cfg.llm.default, cfg.llm.fallback, cfg.llm.retries) == ("big", "small", 0)
    assert cfg.llm.models["big"].model == "m-big" and cfg.llm.models["big"].price_in == 2.0
    assert cfg.llm.models["small"].model == "m-tiny"  # 覆盖项只改提到的那一项
    assert cfg.llm.models["small"].base_url == "http://127.0.0.1:1/v1"
    assert cfg.llm.models["extra"].backend == "openai"  # 新名称用默认值补齐

    again = Config()
    from roboarena.config import _apply

    _apply(again, tomllib.loads(cfg.to_toml()))
    assert again == cfg


def test_default_llm_models_point_at_the_local_server():
    llm = Config().llm
    assert llm.default in llm.models and llm.fallback == ""
    assert all(
        m.base_url.startswith("http://127.0.0.1:") and m.api_key_env == "" for m in llm.models.values()
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        (['llm.default="nope"'], "llm.default 指的模型 nope 没有在 llm.models 里定义"),
        (['llm.fallback="nope"'], "llm.fallback 指的模型 nope 没有在 llm.models 里定义"),
        (["llm.models.big.modle=1"], "未知的配置项：llm.models.big.modle"),
        (["llm.models.big=3"], "llm.models.big 应为一个表"),
        (["llm.models=3"], "llm.models 应为一个表"),
        (['llm.models.big.backend="grpc"'], "llm.models.big.backend 应为 openai"),
        (["llm.retries=-1"], "llm.retries 不能是负数"),
        (["llm.timeout_s=0"], "llm.timeout_s 应大于 0"),
        (['llm.models.a:b.model="x"'], "llm.models 里的名称 a:b 只能由字母、数字、下划线、连字符组成"),
    ],
)
def test_bad_llm_settings_are_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=overrides)
