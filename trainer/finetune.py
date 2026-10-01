"""Fine-tune Laya's multilingual checkpoint on the labeled insult examples.

The recipe follows Laya's own fine-tuning notebook (soft cross-entropy plus the RLCD policy-gradient
term on proper-scoring-rule rewards), on one GPU:

1. Score the test split with the base model, and record its sentiment answers on the training texts.
2. Train on the insult labels. Every training text also carries the sentiment question with the base
   model's answer as target, so the shared encoder learns insults without drifting on sentiment.
3. Fit calibration temperatures on the calib split, which training never sees.
4. Score the test split again and keep the new checkpoint only if it is better.

Sequences are built with the Agent's own encoding (`_encode_state`), the same path `predict` uses,
so training sees exactly what inference sees. Those are private Laya APIs, which is why
pyproject.toml pins the laya version.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.checkpoints import ActiveCheckpoint
from app.classifier import QUESTIONS, STATE_KEY
from app.examples import ExampleStore
from trainer.metrics import activation_decision, agreement, binary_metrics, calibration_error
from trainer.targets import distribution_target, label_target

BASE_REPO = "convaiinnovations/laya"
BASE_SUBFOLDER = "multilingual"
MIN_TRAIN_PER_CLASS = 20
MIN_CALIB_EXAMPLES = 20
QUESTION_IDS = ["insult", "sentiment"]


class NotEnoughData(Exception):
    pass


@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 16
    lr_encoder: float = 2.5e-5
    lr_head: float = 1.0e-4
    weight_decay: float = 0.01
    # RLCD exploration: GROUP_SIZE noisy copies of the logits per step, noise shrinking over training.
    group_size: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    seed: int = 0


def _check_enough(train: list[dict]) -> None:
    counts = {label: sum(e["label"] == label for e in train) for label in ("insult", "clean")}
    few = [f"{label} ({n})" for label, n in counts.items() if n < MIN_TRAIN_PER_CLASS]
    if few:
        raise NotEnoughData(
            f"Hacen falta al menos {MIN_TRAIN_PER_CLASS} ejemplos de entrenamiento de cada tipo; "
            f"faltan de: {', '.join(few)}. Etiquetá más ejemplos en /training.html."
        )


def _predict(agent, texts: list[str]) -> list[dict]:
    results = agent.predict_batch([{STATE_KEY: t} for t in texts], QUESTIONS)
    return [
        {
            "p_insult": r["answers"]["insult"]["probabilities"]["insult"],
            "sentiment": r["answers"]["sentiment"]["choice"],
            "sentiment_probs": r["answers"]["sentiment"]["probabilities"],
        }
        for r in results
    ]


def _build_items(agent, examples: list[dict], teacher: list[dict]) -> list[dict]:
    internal = {qid: agent._to_internal(QUESTIONS[qid]) for qid in QUESTION_IDS}
    items = []
    for example, t in zip(examples, teacher):
        insult_item, sentiment_item = agent._encode_state({STATE_KEY: example["text"]}, QUESTION_IDS, internal)
        insult_item["target"] = label_target(QUESTIONS["insult"]["criteria"], example["label"])
        sentiment_item["target"] = distribution_target(QUESTIONS["sentiment"]["criteria"], t["sentiment_probs"])
        for item in (insult_item, sentiment_item):
            item["label"] = max(range(len(item["target"])), key=item["target"].__getitem__)
            items.append(item)
    return items


def _train(agent, items: list[dict], config: TrainConfig, log) -> list[float]:
    import torch
    from laya.common import collate_items, proper_reward

    model = agent.model
    device = next(model.parameters()).device
    model.train()

    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": config.lr_encoder}, {"params": head_params, "lr": config.lr_head}],
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(items) / config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps_per_epoch * config.epochs), eta_min=1e-6
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    epoch_losses = []
    for epoch in range(config.epochs):
        order = items[:]
        rng.shuffle(order)
        progress = epoch / max(1, config.epochs - 1)
        sigma = config.sigma_start + (config.sigma_end - config.sigma_start) * progress
        total, t0 = 0.0, time.time()

        for start in range(0, len(order), config.batch_size):
            batch = collate_items([order[start:start + config.batch_size]], agent.tok.pad_token_id)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits, act = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            qtype = batch["qtype"].to(device)

            # RLCD: sample noisy copies of the logits, reward each with a proper scoring rule, and
            # push the model toward the copies that scored above the group average.
            eps = torch.randn((config.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                reward = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            log_prob = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(advantage * log_prob).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            # `act` feeds Laya's act/escalate head, which this task does not train; the zero term
            # keeps its parameters in the graph.
            loss = loss_rl + loss_ce + 0.0 * act.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss_ce.item()

        epoch_losses.append(total / steps_per_epoch)
        log(f"  época {epoch + 1}/{config.epochs}: pérdida {epoch_losses[-1]:.4f} ({time.time() - t0:.0f} s)")

    model.eval()
    return epoch_losses


def _calibrate(agent, calib: list[dict], teacher: list[dict]) -> dict | None:
    import torch
    from laya.calibrate import records_from_labeled

    pairs = [
        (
            {STATE_KEY: e["text"]},
            {qid: QUESTIONS[qid] for qid in QUESTION_IDS},
            {
                "insult": label_target(QUESTIONS["insult"]["criteria"], e["label"]),
                "sentiment": distribution_target(QUESTIONS["sentiment"]["criteria"], t["sentiment_probs"]),
            },
        )
        for e, t in zip(calib, teacher)
    ]
    with torch.no_grad():
        records = records_from_labeled(agent, pairs)
    return agent.fit_temperatures(records)


def _base_dir() -> Path:
    from huggingface_hub import snapshot_download

    pattern = [f"{BASE_SUBFOLDER}/*"]
    try:
        root = snapshot_download(BASE_REPO, allow_patterns=pattern, local_files_only=True)
    except Exception:
        root = snapshot_download(BASE_REPO, allow_patterns=pattern)
    return Path(root) / BASE_SUBFOLDER


def _save_checkpoint(agent, out: Path, meta: dict) -> None:
    from safetensors.torch import save_file

    base = _base_dir()
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("model.safetensors"))
    # Cloning breaks weight sharing (tied embeddings), which safetensors refuses to save; Laya
    # loads the file with strict=True, so every key has to be present.
    weights = {k: v.detach().to("cpu").clone().contiguous() for k, v in agent.model.state_dict().items()}
    save_file(weights, str(out / "model.safetensors"))

    cfg_path = out / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["temperature"] = list(agent.temperature)
    cfg["temperature_by_options"] = dict(agent.temperature_by_options)
    cfg["training"] = {**cfg.get("training", {}), "fine_tuned_from_checkpoint": True, "message_flagger": meta["config"]}
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "training_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _by_category(test: list[dict], base: list[dict], tuned: list[dict], threshold: float) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(test):
        groups[e.get("category") or "sin categoría"].append(i)

    def accuracy(preds, idx):
        return sum((preds[i]["p_insult"] >= threshold) == (test[i]["label"] == "insult") for i in idx) / len(idx)

    return {cat: {"n": len(idx), "base": accuracy(base, idx), "tuned": accuracy(tuned, idx)} for cat, idx in sorted(groups.items())}


def run(examples: ExampleStore, models_dir: Path, threshold: float, config: TrainConfig | None = None,
        activate: bool = True, device: str | None = None, log=print) -> dict:
    import laya
    import torch

    config = config or TrainConfig()
    train, calib, test = (examples.labeled(s) for s in ("train", "calib", "test"))
    _check_enough(train)
    test_counts = {label: sum(e["label"] == label for e in test) for label in ("insult", "clean")}
    log(f"Ejemplos: {len(train)} de entrenamiento, {len(calib)} de calibración, {len(test)} de prueba.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Cargando el modelo multilingüe base en {device}…")
    agent = laya.load(BASE_REPO, subfolder=BASE_SUBFOLDER, device=device)

    log("Evaluando el modelo base con los ejemplos de prueba…")
    base_test = _predict(agent, [e["text"] for e in test]) if test else []
    teacher_train = _predict(agent, [e["text"] for e in train])
    teacher_calib = _predict(agent, [e["text"] for e in calib]) if calib else []

    log("Entrenando…")
    items = _build_items(agent, train, teacher_train)
    losses = _train(agent, items, config, log)

    if len(calib) >= MIN_CALIB_EXAMPLES:
        log("Calibrando…")
        _calibrate(agent, calib, teacher_calib)
    else:
        log(f"Calibración salteada: hay {len(calib)} ejemplos de calibración (mínimo {MIN_CALIB_EXAMPLES}).")

    log("Evaluando el modelo nuevo…")
    tuned_test = _predict(agent, [e["text"] for e in test]) if test else []
    tuned_calib = _predict(agent, [e["text"] for e in calib]) if calib else []

    labels = [e["label"] for e in test]
    base_m = binary_metrics(labels, [p["p_insult"] for p in base_test], threshold)
    tuned_m = binary_metrics(labels, [p["p_insult"] for p in tuned_test], threshold)
    base_m["calibration_error"] = calibration_error(labels, [p["p_insult"] for p in base_test])
    tuned_m["calibration_error"] = calibration_error(labels, [p["p_insult"] for p in tuned_test])

    # Sentiment drift is measured on every text the weights never trained on.
    held_out = calib + test
    base_held, tuned_held = teacher_calib + base_test, tuned_calib + tuned_test
    sentiment_agreement = agreement([p["sentiment"] for p in base_held], [p["sentiment"] for p in tuned_held])
    sentiment_changes = [
        {"text": e["text"], "base": b["sentiment"], "tuned": t["sentiment"]}
        for e, b, t in zip(held_out, base_held, tuned_held)
        if b["sentiment"] != t["sentiment"]
    ]
    ok, reason = activation_decision(base_m, tuned_m, sentiment_agreement, test_counts)

    metrics = {
        "threshold": threshold,
        "base": base_m,
        "tuned": tuned_m,
        "sentiment_agreement": sentiment_agreement,
        "sentiment_changes": sentiment_changes,
        "test_counts": test_counts,
        "by_category": _by_category(test, base_test, tuned_test, threshold) if test else {},
    }
    created = datetime.now(timezone.utc)
    out = Path(models_dir) / f"ft-{created:%Y%m%d-%H%M%S}"
    meta = {
        "created_at": created.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "laya_version": laya.__version__,
        "base": f"{BASE_REPO}/{BASE_SUBFOLDER}",
        "questions": QUESTIONS,
        "config": asdict(config),
        "counts": {"train": len(train), "calib": len(calib), "test": len(test)},
        "losses": losses,
        "metrics": metrics,
        "decision": {"activate": ok, "reason": reason},
    }
    log(f"Guardando el checkpoint en {out}…")
    _save_checkpoint(agent, out, meta)

    activated = activate and ok
    if activated:
        ActiveCheckpoint(models_dir).activate(out, metrics)
    return {"checkpoint": out, "metrics": metrics, "activated": activated, "reason": reason}
