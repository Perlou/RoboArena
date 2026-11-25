import io
import json
import logging

import numpy as np

from roboarena.core.hashing import canonical_json, combine_digests, digest_arrays, sha256_file
from roboarena.core.log import get_logger, setup_logging, trace_id_var
from roboarena.core.seeding import derive_seed, make_rng


def test_derive_seed_is_stable_and_separates_components():
    assert derive_seed(0, "a") == derive_seed(0, "a")
    assert derive_seed(0, "a") != derive_seed(0, "b")
    assert derive_seed(0, "a") != derive_seed(1, "a")
    assert 0 <= derive_seed(123, "x") < 2**63
    # 跨进程、跨版本都不能变：评测集和数据采集的可复现性建立在它上面
    assert derive_seed(0, "suite:main") == 1919807112974827286


def test_make_rng_streams_are_independent():
    a, b = make_rng(0, "policy"), make_rng(0, "data")
    assert a.random() != b.random()
    assert make_rng(0, "policy").random() == make_rng(0, "policy").random()


def test_digest_arrays_tracks_content_dtype_shape_and_name():
    x = np.arange(6, dtype=np.float32)
    base = digest_arrays({"x": x})
    assert digest_arrays({"x": x.copy()}) == base
    changed = x.copy()
    changed[3] += 1
    assert digest_arrays({"x": changed}) != base
    assert digest_arrays({"x": x.astype(np.float64)}) != base
    assert digest_arrays({"x": x.reshape(2, 3)}) != base
    assert digest_arrays({"y": x}) != base
    # 非连续内存的视图与它的连续副本内容相同，摘要也应相同
    assert digest_arrays({"x": x[::2]}) == digest_arrays({"x": np.ascontiguousarray(x[::2])})


def test_combine_digests_ignores_order():
    assert combine_digests(["b", "a"]) == combine_digests(["a", "b"])
    assert combine_digests(["a", "b"]) != combine_digests(["a", "c"])


def test_canonical_json_ignores_key_order():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})


def test_sha256_file(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_json_log_has_fields_extras_and_trace_id():
    stream = io.StringIO()
    setup_logging(stream=stream)
    token = trace_id_var.set("req-1")
    try:
        get_logger("tests").info("评测完成", extra={"policy": "bc", "n": 3})
    finally:
        trace_id_var.reset(token)
        setup_logging()
    entry = json.loads(stream.getvalue())
    assert entry["msg"] == "评测完成"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "roboarena.tests"
    assert entry["trace_id"] == "req-1"
    assert (entry["policy"], entry["n"]) == ("bc", 3)
    assert "ts" in entry


def test_setup_logging_does_not_stack_handlers():
    setup_logging()
    setup_logging()
    assert len(logging.getLogger("roboarena").handlers) == 1
