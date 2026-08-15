import json
import warnings

import pytest

ENVELOPE = json.dumps({"type": "call", "confidence": 0.9,
                       "function_calls": [{"name": "City",
                                           "arguments": {"city": "Paris"}}]}).encode("utf-8")


class _Stub:
    def __init__(self, calls):
        self.calls = calls

    def needle_init(self, system, tools, index):
        self.calls.append("init")
        return 0

    def needle_load(self, blob, size):
        self.calls.append("load")
        return 0

    def needle_complete(self, text, max_new_tokens, buffer, size):
        self.calls.append("complete")
        buffer.value = ENVELOPE
        return 0

    def needle_reset(self):
        self.calls.append("reset")


@pytest.fixture
def engine(monkeypatch):
    import needle

    calls = []
    stub = _Stub(calls)
    monkeypatch.setattr(needle, "_lib", lambda: stub)
    monkeypatch.setattr(needle, "_active", None)
    monkeypatch.setattr(needle, "_active_weights", None)
    monkeypatch.setattr(needle, "_active_blob", None)
    return calls


@pytest.fixture
def tuned(tmp_path):
    path = tmp_path / "tuned.cact"
    path.write_bytes(b"tuned weights")
    return str(path)


def _tuned_agent(path):
    import needle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return needle.Needle(tools="[]", weights=path)


def test_new_base_agent_refuses_once_tuned_weights_are_loaded(engine, tuned):
    import needle

    _tuned_agent(tuned)
    with pytest.raises(RuntimeError, match="cannot unload"):
        needle.Needle(tools="[]")


def test_existing_base_agent_refuses_instead_of_answering_on_tuned_weights(engine, tuned):
    import needle

    base = needle.Needle(tools="[]")
    base.complete("hello")
    _tuned_agent(tuned)
    with pytest.raises(RuntimeError, match="cannot unload"):
        base.complete("hello")


def test_tuned_agent_rebinds_without_reloading_its_own_weights(engine, tuned):
    agent = _tuned_agent(tuned)
    other = _tuned_agent(tuned)
    other.complete("hello")
    agent.complete("hello")
    assert engine.count("load") == 1


def test_extract_inherits_the_loaded_weights(engine, tuned):
    import needle

    _tuned_agent(tuned)
    schema = {"name": "City", "parameters": {"type": "object",
                                             "properties": {"city": {"type": "string"}}}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert needle.extract("the weather in Paris", schema) == {"city": "Paris"}
    assert needle._active_weights == tuned
    assert engine.count("load") == 1
