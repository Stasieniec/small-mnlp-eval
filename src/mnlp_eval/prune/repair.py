"""LoRA repair of a pruned checkpoint.

Pruning removes units, and even with SlimGPT's compensation the survivors were
trained to work alongside the ones that are gone. Repair trains a low-rank
update on the pruned model so it relearns to translate without them, then
merges the update into the weights. The result has exactly the pruned model's
shapes and loads through ``recipes/pruned.py`` unchanged, so the repaired and
unrepaired systems differ in their weights alone.

The recipe is ALMA's own LoRA stage (``runs/parallel_ft_lora.sh`` and
``utils/utils.py`` in ``fe1ixxu/ALMA``), not a new one:

- rank 16, alpha 32, dropout 0.05, on every linear layer except the output
  head;
- AdamW at 2e-3, weight decay 0.01, inverse square root schedule after 1
  percent warmup, one epoch, 128 sequences per optimizer step;
- the training text is the prompt with the reference appended directly, no
  space, then eos, and the loss is taken on the reference and eos only.

Departures from ALMA, for the write-up:

- 128 sequences per step come from 8 per micro-batch and 16 accumulation
  steps on one GPU, where ALMA used 8 processes of 4 and 4 steps. Gradient
  checkpointing is on, which changes memory and speed but not the result.
- The loss is a per-token mean over the whole step; ALMA's Trainer averaged
  per-micro-batch means.
- The final adapter is kept. ALMA evaluated every 5 percent of training and
  kept the best checkpoint.
- ALMA left the prompt's final token as a label and put eos after 511 tokens;
  here the prompt is fully masked and eos is inside the 511.

A held-out slice is scored before and after training, so whether training did
anything is known before a GPU-hour of generation is spent finding out. The
slice is split by English sentence: ALMA's data is partly multi-way parallel
and every pair appears in both directions, so a split by record would leave
most held-out sentences in training in another direction. Even so the slice
comes from the training distribution; it is a sanity check, not a measure of
generalisation.

Every torch, transformers and peft import is function-local, like the rest of
the package.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mnlp_eval.config import ConfigError, _reject_unknown, load_yaml_config
from mnlp_eval.languages import parse_directions
from mnlp_eval.prune import PruneError

__all__ = [
    "PRUNED_ENTRYPOINT",
    "Example",
    "RepairSpec",
    "encode_example",
    "length_grouped_batches",
    "run_repair",
    "split_held_out",
    "train_lora",
]

#: The loader a pruned model config routes through. Repair reads and writes
#: checkpoints in its format, so a repaired model evaluates with no new code.
PRUNED_ENTRYPOINT = "recipes.pruned:load"

#: Written beside the weights, as ``prune`` does. The loader needs it.
DESCRIPTOR_NAME = "subnetwork.json"

#: Label value the loss ignores, as in transformers.
IGNORE = -100


@dataclass(frozen=True)
class RepairSpec:
    """One repair run. Defaults are ALMA's LoRA recipe."""

    #: Name of the repaired system, and of its checkpoint directory.
    name: str
    #: The model config ``prune`` emitted for the checkpoint to repair, for
    #: example ``configs/models/alma-7b-slimgpt20-multi.yaml``.
    source: str
    #: A set written by ``mnlp-eval calibration``, whose records carry both
    #: ``prompt`` and ``target``.
    data: str
    #: Restrict training to these directions. Empty means every direction the
    #: data holds.
    directions: list[str] | None = None
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    #: Passed to peft. ``all-linear`` is every linear layer but the output head.
    target_modules: str | list[str] = "all-linear"
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation: int = 16
    warmup_ratio: float = 0.01
    max_grad_norm: float = 1.0
    #: ALMA's max_source_length + max_new_tokens - 1.
    max_length: int = 511
    #: Segments kept out of training and scored before and after it.
    held_out: int = 500
    #: Cap on training segments after shuffling. For smoke tests only: a
    #: capped run is not the recipe, and its name should say so.
    max_examples: int | None = None
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    #: Optimizer steps between progress lines.
    log_every: int = 10
    #: Exit non-zero when held-out loss did not fall. Off only for smoke tests,
    #: whose few steps make the comparison noise.
    fail_without_improvement: bool = True
    seed: int = 1234

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepairSpec:
        payload = dict(data)
        _reject_unknown(cls, payload, "repair spec")
        for required in ("name", "source", "data"):
            if not payload.get(required):
                msg = f"repair spec: {required!r} is required"
                raise ConfigError(msg)
        if payload.get("directions"):
            payload["directions"] = sorted(
                str(item) for item in parse_directions(payload["directions"])
            )
        spec = cls(**payload)
        spec.validate()
        return spec

    def validate(self) -> None:
        positive = {
            "rank": self.rank,
            "alpha": self.alpha,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "max_length": self.max_length,
            "log_every": self.log_every,
        }
        for field, value in positive.items():
            if value < 1:
                msg = f"repair spec: {field} must be at least 1, got {value}"
                raise ConfigError(msg)
        if not 0.0 <= self.dropout < 1.0:
            msg = f"repair spec: dropout must be in [0, 1), got {self.dropout}"
            raise ConfigError(msg)
        if self.learning_rate <= 0.0:
            msg = f"repair spec: learning_rate must be positive, got {self.learning_rate}"
            raise ConfigError(msg)
        if not 0.0 <= self.warmup_ratio < 1.0:
            msg = f"repair spec: warmup_ratio must be in [0, 1), got {self.warmup_ratio}"
            raise ConfigError(msg)
        if self.held_out < 0:
            msg = f"repair spec: held_out must be at least 0, got {self.held_out}"
            raise ConfigError(msg)
        if self.max_examples is not None and self.max_examples < 1:
            msg = f"repair spec: max_examples must be at least 1, got {self.max_examples}"
            raise ConfigError(msg)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation

    def describe(self, segments: int, directions: Sequence[str]) -> str:
        """What goes into ``compression.repair``, quoted verbatim in the report."""
        scope = "all ten directions" if len(directions) == 10 else "+".join(directions)
        epochs = "epoch" if self.epochs == 1 else "epochs"
        return (
            f"LoRA r={self.rank} alpha={self.alpha} on {self.target_modules}, merged; "
            f"{self.epochs} {epochs} over {segments} segments of {Path(self.data).name} "
            f"({scope}), lr {self.learning_rate:g}, batch {self.effective_batch_size}"
        )


@dataclass(frozen=True)
class Example:
    """One tokenized training segment."""

    direction: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def n_target_tokens(self) -> int:
        """Positions the loss is taken on. Position 0 is never predicted."""
        return sum(1 for label in self.labels[1:] if label != IGNORE)


def encode_example(
    tokenizer: Any, prompt: str, target: str, *, max_length: int, direction: str = ""
) -> Example | None:
    """Tokenize ``prompt + target + eos`` with the loss masked to the target.

    The prompt is masked by the longest prefix the two tokenizations share,
    which is the whole prompt unless the tokenizer merged its last token with
    the target's first. Returns None when truncation leaves no target token.
    """
    eos = tokenizer.eos_token_id
    if eos is None:
        msg = "the tokenizer has no eos token, so a repaired model could not learn to stop"
        raise PruneError(msg)
    prompt_ids = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    full_ids = list(tokenizer(prompt + target, add_special_tokens=True)["input_ids"])
    if full_ids and full_ids[-1] == eos:
        full_ids = full_ids[:-1]
    full_ids = [*full_ids[: max_length - 1], eos]

    shared = 0
    for left, right in zip(prompt_ids, full_ids, strict=False):
        if left != right:
            break
        shared += 1
    # At least one position before eos is context, and eos is always learned.
    shared = max(1, min(shared, len(full_ids) - 1))
    if len(full_ids) - 1 - shared < 1:
        # Truncation left eos and nothing of the reference.
        return None
    labels = [IGNORE] * shared + full_ids[shared:]
    return Example(direction=direction, input_ids=tuple(full_ids), labels=tuple(labels))


def length_grouped_batches(
    examples: Sequence[Example], batch_size: int, *, seed: int, window: int = 64
) -> list[list[int]]:
    """Batches of indices, of similar length within a batch, in random order.

    Sorting within windows of ``window`` batches cuts padding without making
    the order a length curriculum. transformers' ``group_by_length`` does the
    same.
    """
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    span = batch_size * window
    batches: list[list[int]] = []
    for start in range(0, len(order), span):
        block = sorted(
            order[start : start + span], key=lambda index: len(examples[index].input_ids)
        )
        batches.extend(
            block[offset : offset + batch_size] for offset in range(0, len(block), batch_size)
        )
    rng.shuffle(batches)
    return batches


def _collate(examples: Sequence[Example], pad_id: int, device: Any) -> dict[str, Any]:
    """Right-pad a batch. Padding is masked out of attention and out of the loss."""
    import torch

    width = max(len(example.input_ids) for example in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    attention = torch.zeros((len(examples), width), dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        ids[row, :length] = torch.tensor(example.input_ids)
        labels[row, :length] = torch.tensor(example.labels)
        attention[row, :length] = 1
    return {
        "input_ids": ids.to(device),
        "attention_mask": attention.to(device),
        "labels": labels.to(device),
    }


def _loss_sum(model: Any, batch: dict[str, Any]) -> tuple[Any, int]:
    """Summed next-token loss over label positions, and how many there were."""
    import torch

    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = batch["labels"][:, 1:]
    count = int((shifted_labels != IGNORE).sum())
    loss = torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE,
        reduction="sum",
    )
    return loss, count


def _mean_loss(
    model: Any, examples: Sequence[Example], batch_size: int, pad_id: int, device: Any
) -> float | None:
    """Token-weighted mean loss over ``examples``, or None when there are none."""
    import torch

    if not examples:
        return None
    was_training = model.training
    model.eval()
    total = 0.0
    tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = _collate(examples[start : start + batch_size], pad_id, device)
            loss, count = _loss_sum(model, batch)
            total += float(loss)
            tokens += count
    model.train(was_training)
    return total / max(tokens, 1)


def train_lora(
    model: Any,
    train: Sequence[Example],
    held_out: Sequence[Example],
    spec: RepairSpec,
    *,
    pad_id: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Attach LoRA, train it, merge it, and return the merged model and a log.

    ``model`` must already be on ``device``. The returned model is a plain
    transformers model with the input's shapes.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_inverse_sqrt_schedule

    if not train:
        msg = "no training segments survived tokenization; check max_length and the data"
        raise PruneError(msg)

    torch.manual_seed(spec.seed)
    if spec.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    lora = LoraConfig(
        r=spec.rank,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=spec.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model: Any = get_peft_model(model, lora)
    trainable = [parameter for parameter in peft_model.parameters() if parameter.requires_grad]
    n_trainable = sum(parameter.numel() for parameter in trainable)
    low_precision = sorted({str(p.dtype) for p in trainable if p.dtype != torch.float32})
    if low_precision:
        # peft before 0.12 casts adapters to the base dtype, and a bfloat16
        # adapter with bfloat16 optimizer state trains measurably worse.
        msg = (
            f"LoRA parameters are {', '.join(low_precision)}, expected float32. "
            "Upgrade peft to 0.12 or later."
        )
        raise PruneError(msg)

    batches = length_grouped_batches(train, spec.batch_size, seed=spec.seed)
    steps_per_epoch = math.ceil(len(batches) / spec.gradient_accumulation)
    total_steps = steps_per_epoch * spec.epochs
    warmup_steps = math.ceil(spec.warmup_ratio * total_steps)

    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate, weight_decay=spec.weight_decay)
    scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=warmup_steps)

    held_before = _mean_loss(peft_model, held_out, spec.batch_size, pad_id, device)
    _log(
        f"repair: {len(train)} training segments in {len(batches)} micro-batches, "
        f"{total_steps} optimizer steps of {spec.effective_batch_size}, "
        f"{n_trainable:,} trainable parameters"
    )
    if held_before is not None:
        _log(f"repair: held-out loss before training {held_before:.4f}")

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    tokens_seen = 0
    step = 0
    peft_model.train()
    for epoch in range(spec.epochs):
        order = (
            batches
            if epoch == 0
            else length_grouped_batches(train, spec.batch_size, seed=spec.seed + epoch)
        )
        for window_start in range(0, len(order), spec.gradient_accumulation):
            window = order[window_start : window_start + spec.gradient_accumulation]
            # Normalise by the target tokens in the whole optimizer step, so
            # accumulating micro-batches of different lengths gives the same
            # gradient as one large batch would.
            window_tokens = sum(
                train[index].n_target_tokens for indices in window for index in indices
            )
            window_loss = 0.0
            for indices in window:
                batch = _collate([train[index] for index in indices], pad_id, device)
                loss, count = _loss_sum(peft_model, batch)
                (loss / max(window_tokens, 1)).backward()
                window_loss += float(loss.detach())
                tokens_seen += count
            step_loss = window_loss / max(window_tokens, 1)
            if not math.isfinite(step_loss):
                msg = (
                    f"repair: loss became {step_loss} at step {step + 1}. Nothing was saved. "
                    "Lower learning_rate or check the data."
                )
                raise PruneError(msg)
            norm = torch.nn.utils.clip_grad_norm_(trainable, spec.max_grad_norm)
            if not torch.isfinite(norm):
                msg = (
                    f"repair: gradient norm became {float(norm)} at step {step + 1}. Nothing "
                    "was saved. Lower learning_rate or check the data."
                )
                raise PruneError(msg)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            entry = {
                "step": step,
                "epoch": epoch,
                "loss": round(step_loss, 5),
                "grad_norm": round(float(norm), 4),
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(entry)
            if step % spec.log_every == 0 or step == total_steps:
                elapsed = time.perf_counter() - started
                _log(
                    f"repair: step {step}/{total_steps} loss {step_loss:.4f} "
                    f"lr {entry['lr']:.2e} grad {entry['grad_norm']:.3f} "
                    f"{tokens_seen / max(elapsed, 1e-9):,.0f} target tok/s "
                    f"{elapsed / 60:.1f} min"
                )

    held_after = _mean_loss(peft_model, held_out, spec.batch_size, pad_id, device)
    if held_after is not None:
        _log(f"repair: held-out loss after training {held_after:.4f}")

    merged: Any = peft_model.merge_and_unload()
    if spec.gradient_checkpointing:
        merged.gradient_checkpointing_disable()
    merged.config.use_cache = True
    merged.eval()

    log: dict[str, Any] = {
        "trainable_parameters": n_trainable,
        "optimizer_steps": step,
        "warmup_steps": warmup_steps,
        "micro_batches_per_epoch": len(batches),
        "target_tokens_seen": tokens_seen,
        "train_seconds": round(time.perf_counter() - started, 1),
        "held_out_loss_before": held_before,
        "held_out_loss_after": held_after,
        "final_train_loss": history[-1]["loss"] if history else None,
        "history": history,
    }
    return merged, log


def load_repair_data(
    directory: str | Path, *, directions: Sequence[str] | None = None
) -> list[tuple[str, str, str, str]]:
    """Read ``(direction, prompt, target, english)`` from a ``mnlp-eval calibration`` set.

    ``english`` is the English side of the segment, the source for ``en-xx``
    and the target for ``xx-en``, which is what ties a segment to its
    translations in the other directions.
    """
    from mnlp_eval.artifacts import read_jsonl_dicts
    from mnlp_eval.prune.collect import refuse_contaminated

    root = Path(directory).expanduser()
    refuse_contaminated(root)
    files = sorted(root.glob("*.jsonl"))
    if directions:
        wanted = set(directions)
        files = [path for path in files if path.stem in wanted]
        missing = wanted - {path.stem for path in files}
        if missing:
            msg = f"{root} has no repair data for {sorted(missing)}"
            raise PruneError(msg)
    if not files:
        msg = (
            f"{root} holds no .jsonl files. Build the repair set first, on a login node: "
            "mnlp-eval calibration --spec configs/calibration/repair-multi.yaml"
        )
        raise PruneError(msg)
    records: list[tuple[str, str, str, str]] = []
    for path in files:
        english_side = "source" if path.stem.startswith("en-") else "target"
        for record in read_jsonl_dicts(path):
            if not record.get("prompt") or not record.get("target"):
                msg = f"{path}: a record lacks 'prompt' or 'target', so it cannot train anything"
                raise PruneError(msg)
            english = str(record.get(english_side) or "").strip()
            records.append((path.stem, str(record["prompt"]), str(record["target"]), english))
    return records


def split_held_out(
    records: Sequence[tuple[str, str, str, str]], held_out: int, *, seed: int
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    """Hold out whole English sentences, with every translation of each.

    ALMA's data is partly multi-way parallel and every pair appears in both
    directions, so holding out single records leaves most of them in training
    under another direction. Returns ``(train, held)``; ``held`` has at least
    ``held_out`` records unless there are fewer in total.
    """
    groups: dict[str, list[tuple[str, str, str, str]]] = {}
    for record in records:
        groups.setdefault(record[3], []).append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    held: list[tuple[str, str, str, str]] = []
    train: list[tuple[str, str, str, str]] = []
    for key in keys:
        (held if len(held) < held_out else train).extend(groups[key])
    rng = random.Random(seed + 1)
    rng.shuffle(train)
    return train, held


def resolve_source(path: str | Path) -> dict[str, Any]:
    """Read the pruned model's config and check it is one repair can act on."""
    resolved = load_yaml_config(path)
    if resolved.get("loader") != "custom" or resolved.get("entrypoint") != PRUNED_ENTRYPOINT:
        msg = (
            f"{path}: repair acts on a pruned checkpoint loaded through {PRUNED_ENTRYPOINT}, "
            f"but this config uses loader {resolved.get('loader')!r}, entrypoint "
            f"{resolved.get('entrypoint')!r}"
        )
        raise PruneError(msg)
    checkpoint = (resolved.get("kwargs") or {}).get("checkpoint")
    if not checkpoint:
        msg = f"{path}: no kwargs.checkpoint to repair"
        raise PruneError(msg)
    if not (Path(str(checkpoint)).expanduser() / DESCRIPTOR_NAME).is_file():
        msg = (
            f"{path}: {checkpoint} has no {DESCRIPTOR_NAME}. Has the prune job that writes "
            "it finished?"
        )
        raise PruneError(msg)
    if (resolved.get("compression") or {}).get("repair"):
        msg = f"{path} is already repaired; repair the pruned checkpoint it came from instead"
        raise PruneError(msg)
    return resolved


def run_repair(spec: RepairSpec, out_dir: Path, *, config_dir: Path) -> dict[str, Any]:
    """Repair a pruned checkpoint and write the merged checkpoint and its config."""
    import torch
    from transformers import AutoTokenizer

    from mnlp_eval.artifacts import atomic_write_json
    from mnlp_eval.models.loading import default_device, resolve_dtype
    from mnlp_eval.seeding import seed_everything

    source = resolve_source(spec.source)
    checkpoint = Path(str(source["kwargs"]["checkpoint"])).expanduser()
    seed_everything(spec.seed)

    records = load_repair_data(spec.data, directions=spec.directions)
    train_records, held_records = split_held_out(records, spec.held_out, seed=spec.seed)
    if spec.max_examples is not None:
        train_records = train_records[: spec.max_examples]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    def encode(items: Sequence[tuple[str, str, str, str]]) -> list[Example]:
        encoded = (
            encode_example(tokenizer, prompt, target, max_length=spec.max_length, direction=name)
            for name, prompt, target, _english in items
        )
        return [example for example in encoded if example is not None]

    train = encode(train_records)
    held_out = encode(held_records)
    skipped = len(train_records) + len(held_records) - len(train) - len(held_out)
    directions = sorted({example.direction for example in train})
    _log(
        f"repair: {spec.name} from {checkpoint}, {len(train)} training and {len(held_out)} "
        f"held-out segments across {', '.join(directions)}; {skipped} dropped by truncation"
    )

    load_model = _pruned_loader()
    # Load as saved, then cast parameters only. Casting the whole module would
    # also put the rotary inv_freq buffer in bfloat16, which evaluation, loading
    # with dtype auto, never does.
    model, _subnetwork = load_model(checkpoint, "auto")
    if spec.dtype != "auto":
        target_dtype = resolve_dtype(spec.dtype)
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != target_dtype:
                parameter.data = parameter.data.to(target_dtype)
    device = default_device()
    model = model.to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    merged, log = train_lora(model, train, held_out, spec, pad_id=pad_id, device=device)

    target = out_dir / spec.name
    if target.exists():
        # Stale shards from an earlier attempt would be read by the loader's
        # glob alongside the new ones.
        shutil.rmtree(target)
    target.mkdir(parents=True)
    merged.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    shutil.copyfile(checkpoint / DESCRIPTOR_NAME, target / DESCRIPTOR_NAME)
    # The source's generation defaults came from the dense model; the one
    # save_pretrained writes here came from a bare config.
    if (checkpoint / "generation_config.json").is_file():
        shutil.copyfile(checkpoint / "generation_config.json", target / "generation_config.json")

    description = spec.describe(len(train), directions)
    manifest: dict[str, Any] = {
        "name": spec.name,
        "source_config": str(spec.source),
        "source_checkpoint": str(checkpoint),
        "checkpoint": str(target),
        "repair": description,
        "spec": asdict(spec),
        "segments": {"train": len(train), "held_out": len(held_out), "dropped": skipped},
        "directions": directions,
        "peak_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None
        ),
        **log,
    }
    atomic_write_json(target / "repair.json", manifest)
    config_path = _write_model_config(spec, target, description, config_dir)
    manifest["model_config"] = str(config_path)
    return manifest


def _pruned_loader() -> Any:
    """``recipes.pruned.load_model``, importable from the repository checkout."""
    import importlib

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module = importlib.import_module(PRUNED_ENTRYPOINT.partition(":")[0])
    return module.load_model


def _write_model_config(
    spec: RepairSpec, checkpoint: Path, description: str, config_dir: Path
) -> Path:
    """Emit a model config that extends the pruned one and swaps the checkpoint.

    Everything else, the loader, the baseline, the subnetwork and what it was
    pruned for, is inherited, so the report pairs the two systems.
    """
    import yaml

    from mnlp_eval.artifacts import atomic_write_text

    config_dir.mkdir(parents=True, exist_ok=True)
    parent = Path(spec.source).expanduser().resolve()
    payload = {
        "extends": os.path.relpath(parent, config_dir.resolve()),
        "name": spec.name,
        "kwargs": {"checkpoint": str(checkpoint)},
        "compression": {"repair": description},
        "notes": f"{Path(spec.source).stem} after LoRA repair. Written by mnlp-eval repair.",
    }
    path = config_dir / f"{spec.name}.yaml"
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return path


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
