"""HFBackend finds a wrapped text trunk, and the block loop feeds it the kwargs that trunk expects.

No checkpoint: the resolution tests are plain objects, and the loop test is a tiny fake decoder
skipped when torch or transformers is absent. A real parity run is `scripts/exit_parity.py`
(or this file's engine test, when ANYJEV_ENGINE_MODEL points at a local checkpoint).
"""
import os
import subprocess
import sys

import pytest

from anyjev.backends.hf import HFBackend


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get_text_config(self):
        return self


def _backend(model):
    be = HFBackend.__new__(HFBackend)
    be.model = model
    return be


def test_trunk_on_model_is_used_with_its_own_config():
    inner = _Cfg(layers=[object()], embed_tokens=object())
    cfg = _Cfg(num_hidden_layers=28, hidden_size=1024, final_logit_softcapping=None)
    be = _backend(_Cfg(model=inner, config=cfg))
    assert be._text_trunk() is inner
    assert be._text_config() is cfg
    assert be._logit_cap() is None


def test_wrapped_trunk_is_language_model_and_text_config():
    lang = _Cfg(layers=[object()], embed_tokens=object())
    # the outer decoder wrapper has the towers, not the blocks
    wrapper = _Cfg(language_model=lang)
    text = _Cfg(num_hidden_layers=42, hidden_size=2560, final_logit_softcapping=30.0)
    outer = _Cfg(text_config=text, final_logit_softcapping=None, get_text_config=lambda: text)
    be = _backend(_Cfg(model=wrapper, config=outer))
    assert be._text_trunk() is lang
    assert be._text_config() is text
    assert be._logit_cap() == 30.0


def test_config_without_get_text_config_is_used_as_is():
    cfg = type("Bare", (), {"num_hidden_layers": 4, "final_logit_softcapping": 5.0})()
    be = _backend(_Cfg(model=_Cfg(layers=[1], embed_tokens=object()), config=cfg))
    assert be._text_config() is cfg
    assert be._logit_cap() == 5.0


def test_missing_trunk_raises():
    be = _backend(_Cfg(model=_Cfg(), config=_Cfg()))
    with pytest.raises(NotImplementedError):
        be._text_trunk()


def _torch_loop():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return torch


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

    def __call__(self, prompts, return_tensors="pt", padding=True, add_special_tokens=False):
        import torch
        n = len(prompts)
        ids = torch.tensor([[1, 2, 3]] * n)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _Layer:
    def __init__(self):
        self.calls = []

    def __call__(self, hidden, per_layer_input=None, **kwargs):
        self.calls.append((per_layer_input, kwargs))
        return hidden


class _RopePerType:
    def forward(self, x, position_ids, layer_type):
        return ("rope", layer_type)

    def __call__(self, x, position_ids, layer_type):
        return self.forward(x, position_ids, layer_type)


class _RopeShared:
    def forward(self, x, position_ids):
        return ("rope", "one")

    def __call__(self, x, position_ids):
        return self.forward(x, position_ids)


class _Trunk:
    def __init__(self, per_type):
        self.layers = [_Layer(), _Layer()]
        self.rotary_emb = _RopePerType() if per_type else _RopeShared()
        self.hidden_size_per_layer_input = 4 if per_type else 0
        self.unique_layer_types = ["sliding_attention", "full_attention"]

    def embed_tokens(self, ids):
        import torch
        b, t = ids.shape
        return torch.zeros(b, t, 8)

    def get_per_layer_inputs(self, input_ids, inputs_embeds):
        return input_ids

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
        import torch
        b, t, _ = inputs_embeds.shape
        n, d = len(self.layers), self.hidden_size_per_layer_input
        return torch.arange(b * t * n * d, dtype=torch.float32).reshape(b, t, n, d)

    def norm(self, hidden):
        return hidden


def _install(monkeypatch, causal_keys, spelling):
    """`spelling` is the embed parameter the installed masking helper declares."""
    import transformers
    import transformers.masking_utils as masking

    class FakeCache:
        def __init__(self, *args, **kwargs):
            self.args = args

    if spelling == "inputs_embeds":
        def causal(config, inputs_embeds, attention_mask, past_key_values, position_ids=None):
            causal_keys.append({"inputs_embeds"})
            return "FULL"

        def slide(config, inputs_embeds, attention_mask, past_key_values, position_ids=None):
            causal_keys.append({"inputs_embeds", "slide"})
            return "SLIDE"
    else:
        def causal(config, input_embeds, attention_mask, past_key_values, position_ids=None,
                   cache_position=None):
            causal_keys.append({"input_embeds", "cache_position"})
            return "FULL"

        def slide(config, input_embeds, attention_mask, past_key_values, position_ids=None,
                  cache_position=None):
            causal_keys.append({"input_embeds", "cache_position", "slide"})
            return "SLIDE"

    monkeypatch.setattr(transformers, "DynamicCache", FakeCache, raising=False)
    monkeypatch.setattr(masking, "create_causal_mask", causal)
    monkeypatch.setattr(masking, "create_sliding_window_causal_mask", slide)


def _be(trunk, layer_types):
    import torch
    text = _Cfg(num_hidden_layers=2, hidden_size=8, final_logit_softcapping=None, layer_types=layer_types)
    wrapper = _Cfg(language_model=trunk) if layer_types else trunk
    # Plain shape: the trunk itself is .model and has layers. Wrapped shape: .model has language_model.
    if layer_types:
        model = _Cfg(model=wrapper, config=text, device=torch.device("cpu"))
    else:
        model = _Cfg(model=trunk, config=text, device=torch.device("cpu"))
    be = _backend(model)
    be.tokenizer = _Tok()
    be.n_layers = 2
    be.hidden_size = 8
    be.batch_size = 4
    return be


def test_per_type_loop_stops_early_and_passes_ple_and_sliding_mask(monkeypatch):
    torch = _torch_loop()
    keys = []
    _install(monkeypatch, keys, "inputs_embeds")
    trunk = _Trunk(per_type=True)
    be = _be(trunk, ["sliding_attention", "full_attention"])
    be.hidden_states_to(["hello"], [1], max_layer=1)
    assert trunk.layers[1].calls == []
    ple_i, kw = trunk.layers[0].calls[0]
    ids = torch.tensor([[1, 2, 3]])
    embeds = trunk.embed_tokens(ids)
    ple = trunk.project_per_layer_inputs(embeds, trunk.get_per_layer_inputs(ids, embeds))
    assert torch.equal(ple_i, ple[:, :, 0, :])
    assert kw["attention_mask"] == "SLIDE"
    assert kw["position_embeddings"] == ("rope", "sliding_attention")
    assert kw["past_key_values"] is not None
    assert "past_key_value" not in kw and "cache_position" not in kw
    from collections import UserDict
    assert isinstance(kw["shared_kv_states"], UserDict)
    assert any("inputs_embeds" in k for k in keys)


def test_shared_rope_loop_keeps_past_key_value_and_cache_position(monkeypatch):
    _torch_loop()
    keys = []
    _install(monkeypatch, keys, "input_embeds")
    trunk = _Trunk(per_type=False)
    be = _be(trunk, None)
    be.hidden_states_to(["hello"], [1], max_layer=1)
    assert trunk.layers[1].calls == []
    ple_i, kw = trunk.layers[0].calls[0]
    assert ple_i is None
    assert kw["attention_mask"] == "FULL"
    assert kw["position_embeddings"] == ("rope", "one")
    assert "past_key_value" in kw and "cache_position" in kw
    assert "shared_kv_states" not in kw
    assert any("input_embeds" in k for k in keys)
    assert not any("inputs_embeds" in k for k in keys)


def test_shared_rope_loop_uses_the_5x_mask_spelling(monkeypatch):
    _torch_loop()
    keys = []
    _install(monkeypatch, keys, "inputs_embeds")
    trunk = _Trunk(per_type=False)
    be = _be(trunk, None)
    be.hidden_states_to(["hello"], [1], max_layer=1)
    assert trunk.layers[0].calls
    assert any("inputs_embeds" in k for k in keys)
    assert not any("cache_position" in k or "input_embeds" in k for k in keys)


def test_embed_scale_on_the_trunk_still_refuses(monkeypatch):
    _torch_loop()
    _install(monkeypatch, [], "inputs_embeds")
    trunk = _Trunk(per_type=True)
    trunk.embed_scale = 1.0
    be = _be(trunk, ["sliding_attention", "full_attention"])
    with pytest.raises(NotImplementedError, match="scaled embeddings"):
        be.hidden_states_to(["hello"], [1], max_layer=1)
    assert trunk.layers[0].calls == []


@pytest.mark.engine
def test_exit_parity_when_checkpoint_present():
    model = os.environ.get("ANYJEV_ENGINE_MODEL", "")
    if not model or not os.path.isdir(model):
        pytest.skip("set ANYJEV_ENGINE_MODEL to a local checkpoint with a wrapped, per-type trunk")
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "exit_parity.py")
    subprocess.check_call([sys.executable, os.path.abspath(script), "--model", model, "--dtype", "float32"])
