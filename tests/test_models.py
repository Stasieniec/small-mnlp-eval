"""The loader registry and the custom-loader contract.

The custom loader is the promise the whole plugin story rests on: a pruned
model with its own modelling code becomes a fully evaluated system by writing
one function. These tests hold that contract in place.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from mnlp_eval.config import ModelSpec
from mnlp_eval.languages import parse_direction
from mnlp_eval.models.base import Translator
from mnlp_eval.models.custom import load_custom
from mnlp_eval.models.hf_causal import CausalTranslator
from mnlp_eval.models.hf_seq2seq import Seq2SeqTranslator
from mnlp_eval.models.registry import available_loaders, build_translator
from mnlp_eval.prompts import get_prompt
from stubs import FakeTokenizer

DE_EN = parse_direction("de-en")


def _spec(**overrides: object) -> ModelSpec:
    payload: dict[str, object] = {
        "name": "custom-model",
        "loader": "custom",
        "prompt": "alma",
        "entrypoint": "loader_module:load",
    }
    payload.update(overrides)
    return ModelSpec.from_dict(payload)


@pytest.fixture
def recipe_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory on sys.path where test recipes can be written."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    for name in [key for key in sys.modules if key.startswith("loader_module")]:
        del sys.modules[name]


def _write_recipe(directory: Path, body: str) -> None:
    (directory / "loader_module.py").write_text(
        textwrap.dedent(
            """
            class FakeConfig:
                is_encoder_decoder = False


            class FakeModel:
                config = FakeConfig()


            class FakeTokenizer:
                pad_token_id = 0
                eos_token_id = 1
                eos_token = "</s>"
                pad_token = "<pad>"
                padding_side = "right"


            """
        )
        + textwrap.dedent(body),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_lists_every_loader() -> None:
    assert set(available_loaders()) == {"hf_causal", "hf_seq2seq", "nllb", "custom"}


def test_unknown_loader_points_at_the_escape_hatch() -> None:
    spec = ModelSpec.from_dict(
        {"name": "x", "loader": "magic", "model_name_or_path": "y", "prompt": "alma"}
    )
    with pytest.raises(KeyError, match=r"loader 'custom'"):
        build_translator(spec)


# --------------------------------------------------------------------------
# Custom loader contract
# --------------------------------------------------------------------------


def test_custom_loader_accepts_a_model_tokenizer_tuple(recipe_dir: Path) -> None:
    _write_recipe(
        recipe_dir,
        """
        def load(**kwargs):
            return FakeModel(), FakeTokenizer()
        """,
    )
    translator = load_custom(_spec())
    assert isinstance(translator, CausalTranslator)
    assert translator.spec.name == "custom-model"


def test_custom_loader_accepts_a_mapping_with_extras(recipe_dir: Path) -> None:
    _write_recipe(
        recipe_dir,
        """
        def load(**kwargs):
            return {
                "model": FakeModel(),
                "tokenizer": FakeTokenizer(),
                "kind": "seq2seq",
                "extra": {"sparsity": kwargs.get("sparsity")},
            }
        """,
    )
    translator = load_custom(_spec(kwargs={"sparsity": 0.5}))
    assert isinstance(translator, Seq2SeqTranslator)
    info = translator.info()
    assert info.extra is not None
    assert info.extra["sparsity"] == 0.5
    assert info.extra["entrypoint"] == "loader_module:load"


def test_custom_loader_accepts_a_ready_made_translator(recipe_dir: Path) -> None:
    _write_recipe(
        recipe_dir,
        """
        from mnlp_eval.models.hf_causal import CausalTranslator
        from mnlp_eval.config import ModelSpec
        from mnlp_eval.prompts import get_prompt


        def load(**kwargs):
            spec = ModelSpec.from_dict(
                {"name": "inner", "loader": "custom", "entrypoint": "a:b", "prompt": "alma"}
            )
            return CausalTranslator(
                spec=spec, prompt=get_prompt("alma"), model=FakeModel(),
                tokenizer=FakeTokenizer(),
            )
        """,
    )
    translator = load_custom(_spec())
    assert isinstance(translator, Translator)
    assert translator.spec.name == "inner"


def test_custom_loader_passes_kwargs_through(recipe_dir: Path) -> None:
    _write_recipe(
        recipe_dir,
        """
        SEEN = {}


        def load(**kwargs):
            SEEN.update(kwargs)
            return FakeModel(), FakeTokenizer()
        """,
    )
    load_custom(_spec(kwargs={"checkpoint": "/scratch/x", "sparsity": 0.5}))
    import loader_module

    assert loader_module.SEEN == {"checkpoint": "/scratch/x", "sparsity": 0.5}


def test_custom_loader_infers_seq2seq_from_the_model_config(recipe_dir: Path) -> None:
    _write_recipe(
        recipe_dir,
        """
        class EncoderDecoderConfig:
            is_encoder_decoder = True


        def load(**kwargs):
            model = FakeModel()
            model.config = EncoderDecoderConfig()
            return model, FakeTokenizer()
        """,
    )
    assert isinstance(load_custom(_spec()), Seq2SeqTranslator)


def test_custom_loader_rejects_an_unusable_return_value(recipe_dir: Path) -> None:
    _write_recipe(
        recipe_dir,
        """
        def load(**kwargs):
            return "not a model"
        """,
    )
    with pytest.raises(TypeError, match="Return a Translator"):
        load_custom(_spec())


def test_custom_loader_reports_a_mapping_missing_keys(recipe_dir: Path) -> None:
    _write_recipe(
        recipe_dir,
        """
        def load(**kwargs):
            return {"model": FakeModel()}
        """,
    )
    with pytest.raises(TypeError, match="'model' and 'tokenizer' are required"):
        load_custom(_spec())


def test_missing_module_explains_the_import_path(recipe_dir: Path) -> None:
    with pytest.raises(ImportError, match=r"sys\.path"):
        load_custom(_spec(entrypoint="no_such_module:load"))


def test_missing_function_lists_available_names(recipe_dir: Path) -> None:
    _write_recipe(
        recipe_dir,
        """
        def other_name(**kwargs):
            return FakeModel(), FakeTokenizer()
        """,
    )
    with pytest.raises(AttributeError, match="other_name"):
        load_custom(_spec(entrypoint="loader_module:load"))


# --------------------------------------------------------------------------
# Translator behaviour that does not need weights
# --------------------------------------------------------------------------


def test_causal_translator_uses_left_padding() -> None:
    # Right padding puts pad tokens between the prompt and the continuation,
    # which silently corrupts decoder-only generation.
    spec = ModelSpec.from_dict(
        {"name": "x", "loader": "hf_causal", "model_name_or_path": "y", "prompt": "alma"}
    )
    translator = CausalTranslator(
        spec=spec, prompt=get_prompt("alma"), model=object(), tokenizer=FakeTokenizer()
    )
    assert translator.tokenizer.padding_side == "left"
    assert translator.strips_prompt_from_output


def test_seq2seq_translator_keeps_default_padding() -> None:
    spec = ModelSpec.from_dict(
        {
            "name": "x",
            "loader": "hf_seq2seq",
            "model_name_or_path": "y",
            "prompt": "passthrough",
        }
    )
    translator = Seq2SeqTranslator(
        spec=spec, prompt=get_prompt("passthrough"), model=object(), tokenizer=FakeTokenizer()
    )
    assert translator.tokenizer.padding_side == "right"
    assert not translator.strips_prompt_from_output


def test_missing_pad_token_falls_back_to_eos() -> None:
    # Llama-family checkpoints, ALMA included, ship without a pad token.
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = None
    spec = ModelSpec.from_dict(
        {"name": "x", "loader": "hf_causal", "model_name_or_path": "y", "prompt": "alma"}
    )
    CausalTranslator(spec=spec, prompt=get_prompt("alma"), model=object(), tokenizer=tokenizer)
    assert tokenizer.pad_token == tokenizer.eos_token


def test_causal_translator_builds_the_alma_prompt() -> None:
    spec = ModelSpec.from_dict(
        {"name": "x", "loader": "hf_causal", "model_name_or_path": "y", "prompt": "alma"}
    )
    translator = CausalTranslator(
        spec=spec, prompt=get_prompt("alma"), model=object(), tokenizer=FakeTokenizer()
    )
    assert translator.build_input(DE_EN, "Das ist ein Test.").endswith("\nEnglish:")
    assert translator.add_special_tokens


def test_chat_prompt_disables_extra_special_tokens() -> None:
    # A chat template already emits BOS; adding another shifts every position.
    spec = ModelSpec.from_dict(
        {"name": "x", "loader": "hf_causal", "model_name_or_path": "y", "prompt": "alma_chat"}
    )
    translator = CausalTranslator(
        spec=spec, prompt=get_prompt("alma_chat"), model=object(), tokenizer=FakeTokenizer()
    )
    assert not translator.add_special_tokens
