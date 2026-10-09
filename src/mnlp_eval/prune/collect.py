"""Activation statistics over a calibration set.

Reads what ``mnlp-eval calibration`` writes and runs it through a model,
accumulating what a criterion needs about each prunable projection's input.

Three things are load bearing. The rendered prompt is tokenized, not the bare
source, because that is the distribution the model runs on. Padding is excluded
via the attention mask, or every channel's mean drifts toward the pad embedding
and the variance the criterion reads shrinks. Statistics accumulate in float32
whatever the model's dtype, since the criterion ranks quantities that end up
close together.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mnlp_eval.prune import PruneError

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "CALIBRATION_MANIFEST",
    "PRUNABLE_INPUTS",
    "InputStats",
    "collect_input_stats",
    "load_calibration_prompts",
    "refuse_contaminated",
    "tokenized_batches",
]

#: The projections whose *input* channels index a prunable group, which is why
#: every activation-based criterion hooks exactly these two.
PRUNABLE_INPUTS = ("o_proj", "down_proj")

#: What ``mnlp-eval calibration`` writes beside the per-direction files.
CALIBRATION_MANIFEST = "calibration.json"


@dataclass
class InputStats:
    """Running mean and variance of one projection's input channels.

    Chan's parallel update rather than a sum and a sum of squares, which
    cancels badly when the mean dwarfs the spread, as it does here.
    """

    count: int
    mean: Tensor
    #: Sum of squared deviations from the mean, the numerator of the variance.
    m2: Tensor

    @property
    def variance(self) -> Tensor:
        """Sample variance per input channel, the fluctuation FLAP scores on."""
        if self.count < 2:
            return self.m2 * 0.0
        return self.m2 / (self.count - 1)

    def update(self, batch: Tensor) -> None:
        """Fold in a ``(tokens, channels)`` block of activations."""
        tokens = int(batch.shape[0])
        if tokens == 0:
            return
        batch_mean = batch.mean(0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(0)
        total = self.count + tokens
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (tokens / total)
        self.m2 = self.m2 + batch_m2 + delta**2 * self.count * tokens / total
        self.count = total


def refuse_contaminated(root: Path) -> None:
    """Refuse a set whose own manifest records test-set collisions.

    ``mnlp-eval calibration`` exits non-zero on a collision, but only after
    writing the set under the name every config reads, so the exit code alone
    does not stop a later prune or repair job from using it. A set with no
    manifest predates the check and is let through.
    """
    import json

    manifest = root / CALIBRATION_MANIFEST
    if not manifest.is_file():
        return
    contamination = json.loads(manifest.read_text(encoding="utf-8")).get("contamination") or {}
    collisions = contamination.get("total_collisions")
    if collisions:
        msg = (
            f"{root}: its {CALIBRATION_MANIFEST} records {collisions} segment(s) that also "
            f"appear in {contamination.get('dataset')}. A set that overlaps the test set "
            "must not be used."
        )
        raise PruneError(msg)


def load_calibration_prompts(
    directory: str | Path,
    *,
    directions: Sequence[str] | None = None,
    with_target: bool = False,
    with_generated: bool = False,
) -> list[str] | list[list[int]]:
    """Read the rendered prompts from a calibration set, in a stable order.

    ``directions`` restricts it, which is how a pair-specific subnetwork is
    calibrated from a set holding all ten.

    ``with_target`` appends each record's reference translation to its prompt,
    joined as ALMA's training joins them, with no space. The model then reads
    the target language as it would have written it, so the criterion sees the
    activations of producing the output as well as of reading the source.
    ``with_generated`` instead returns the cached input and continuation token
    IDs (including EOS), preserving generation truncation and token boundaries.
    Without either option, ``en-xx`` calibration never puts a word of the target language
    through the model.
    """
    from mnlp_eval.artifacts import read_jsonl_dicts

    root = Path(directory).expanduser()
    if with_target and with_generated:
        raise PruneError("choose reference or generated calibration, not both")
    import json

    manifest_path = root / CALIBRATION_MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if with_generated or "generated_schema_version" in manifest:
        from mnlp_eval.data.generated_calibration import validate_cache

        validate_cache(root)
    refuse_contaminated(root)
    files = sorted(root.glob("*.jsonl"))
    if directions is not None:
        wanted = {str(direction) for direction in directions}
        files = [path for path in files if path.stem in wanted]
        missing = wanted - {path.stem for path in files}
        if missing:
            msg = f"{root} has no calibration data for {sorted(missing)}"
            raise PruneError(msg)
    if not files:
        msg = (
            f"{root} holds no .jsonl calibration files. Build one first with "
            "mnlp-eval calibration --spec configs/calibration/<name>.yaml"
        )
        raise PruneError(msg)

    prompts: list[str] = []
    generated_sequences: list[list[int]] = []
    for path in files:
        for record in read_jsonl_dicts(path):
            prompt = record.get("prompt")
            if not prompt:
                msg = f"{path}: a record has no 'prompt' field"
                raise PruneError(msg)
            if with_generated:
                generated_sequences.append(
                    record["input_token_ids"] + record["generated_token_ids"]
                )
            elif with_target:
                target = record.get("target")
                if not target:
                    msg = f"{path}: a record has no 'target' field to calibrate on"
                    raise PruneError(msg)
                prompts.append(str(prompt) + str(target))
            else:
                prompts.append(str(prompt))
    return generated_sequences if with_generated else prompts


def tokenized_batches(
    prompts: Sequence[str] | Sequence[list[int]],
    tokenizer: Any,
    *,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cpu",
    append_eos: bool = False,
) -> Iterator[dict[str, Tensor]]:
    """Tokenize prompts into padded batches with their attention masks.

    ``append_eos`` ends every text with the end-of-sequence token, as ALMA's
    training text ends and as a cached generation ends. With it, a prompt with
    its reference and a prompt with the dense model's own translation differ
    only in the translation, not in whether the model also saw it stop.
    """
    if append_eos and prompts and isinstance(prompts[0], str):
        eos = tokenizer.eos_token_id
        if eos is None:
            msg = "append_eos needs a tokenizer with an eos token"
            raise PruneError(msg)
        prompts = [
            [*tokenizer(text, truncation=True, max_length=max_length - 1)["input_ids"], eos]
            for text in prompts
        ]
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        if chunk and isinstance(chunk[0], list):
            if any(len(ids) > max_length for ids in chunk):
                raise PruneError(
                    "generated calibration exceeds max_length; increase it to preserve "
                    "the complete prompt and continuation (up to 768 tokens for ALMA)"
                )
            encoded = tokenizer.pad(
                [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in chunk],
                padding=True,
                return_tensors="pt",
            )
        else:
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        yield {key: value.to(device) for key, value in encoded.items()}


def collect_input_stats(
    model: Any,
    batches: Iterator[dict[str, Tensor]],
    *,
    targets: Sequence[str] = PRUNABLE_INPUTS,
) -> dict[Any, InputStats]:
    """Mean and variance of every target projection's input, over the batches.

    Keyed by the module, so a criterion looks its statistics up from the
    projection it already holds. One pass over the dense model with a hook per
    target, not a layer-at-a-time walk: a global budget compares layers before
    anything is removed, so every layer's statistics must exist before the
    first unit is pruned.
    """
    import torch

    stats: dict[Any, InputStats] = {}
    mask: Tensor | None = None
    handles = []

    def capture(module: Any, args: tuple[Any, ...]) -> None:
        activations = args[0]
        flat = activations.reshape(-1, activations.shape[-1]).float()
        if mask is not None:
            flat = flat[mask.reshape(-1).bool()]
        entry = stats.get(module)
        if entry is None:
            width = int(flat.shape[-1])
            entry = InputStats(
                count=0,
                mean=torch.zeros(width, dtype=torch.float32, device=flat.device),
                m2=torch.zeros(width, dtype=torch.float32, device=flat.device),
            )
            stats[module] = entry
        entry.update(flat)

    suffixes = set(targets)
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in suffixes:
            handles.append(module.register_forward_pre_hook(capture))
    if not handles:
        msg = f"no modules named {sorted(suffixes)} to collect statistics from"
        raise PruneError(msg)

    try:
        with torch.inference_mode():
            for batch in batches:
                mask = batch.get("attention_mask")
                model(**batch)
    finally:
        for handle in handles:
            handle.remove()

    if not stats:
        msg = "the calibration set produced no batches, so no statistics were collected"
        raise PruneError(msg)
    return stats
