"""Laya answering the taxonomy as typed questions, zero-shot, with the same Router setup as the chat
app: text Laya identifies as English goes to the English checkpoint, the rest to the multilingual one."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata

from app.classifier import QUESTIONS as APP_QUESTIONS
from app.classifier import STATE_KEY, build_router
from bench.contestants.base import free_gpu
from bench.taxonomy import FLAG, LANGS, TARGETS_PLAYER

# The convaiinnovations/laya commit the preliminary scoreboard ran on (read from the local HF cache on 2026-10-06).
LAYA_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"

# One question per category, positive option first. A first wording, not tuned: the report says so.
TAXONOMY_QUESTIONS = {
    "insult": {
        "type": "choice",
        "instructions": "Does `message` insult, mock or harass a specific person?",
        "criteria": {
            "insult": "insults, name-calling, mockery or harassment aimed at a person",
            "no_insult": "no insult aimed at a person; includes friendly banter, self-criticism, swearing at no one and criticism of play",
        },
    },
    "threat": {
        "type": "choice",
        "instructions": "Does `message` threaten someone?",
        "criteria": {
            "threat": "threatens violence or harm against a person or group, or to expose where they live",
            "no_threat": "no threat of violence or harm",
        },
    },
    "identity_hate": {
        "type": "choice",
        "instructions": "Does `message` attack people for who they are?",
        "criteria": {
            "hate": "attacks, demeans or slurs people for race, ethnicity, nationality, religion, gender, sexual orientation, gender identity or disability",
            "no_hate": "no attack on a protected group; includes quoting or condemning hate and neutral or friendly mentions of a group",
        },
    },
    "sexual_harassment": {
        "type": "choice",
        "instructions": "Is `message` sexual harassment?",
        "criteria": {
            "harassment": "unwanted sexual comments, requests or advances aimed at a person",
            "no_harassment": "no sexual harassment",
        },
    },
    "profanity": {
        "type": "choice",
        "instructions": "Does `message` contain swearing or vulgar language?",
        "criteria": {
            "profanity": "contains swear words or vulgar language, aimed at anyone or at no one",
            "no_profanity": "no swear words or vulgar language",
        },
    },
    "target": {
        "type": "choice",
        "instructions": "Who does `message` attack?",
        "criteria": {
            "player": "a specific person in the conversation",
            "group": "a group of people",
            "nobody": "nobody; the message attacks no one",
        },
    },
}
POSITIVE = {"insult": "insult", "threat": "threat", "identity_hate": "hate", "sexual_harassment": "harassment", "profanity": "profanity"}

TAXONOMY_NOTES = (
    "Laya sin ajustar (zero-shot) con seis preguntas redactadas para este benchmark y sin pulir: el "
    "resultado depende tanto de la redacción como del modelo. El inglés que Laya identifica va al "
    "checkpoint inglés (ModernBERT-large) y el resto al multilingüe (mmBERT-base). Hace una fila de "
    "encoder por pregunta: seis por mensaje."
)
APP_NOTES = (
    "La pregunta `insult` que usa hoy la app (insultos, slurs y amenazas en una sola pregunta) como flag. "
    "Es la línea de base: lo que ya tenemos."
)


def fingerprint(questions: dict) -> str:
    return hashlib.sha1(json.dumps(questions, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def flatten(result: dict) -> dict[str, float]:
    raw = {f"{qid}.{option}": float(p) for qid, answer in result["answers"].items() for option, p in answer["probabilities"].items()}
    raw["routed_english"] = 1.0 if result["routing"]["model"] == "english" else 0.0
    return raw


def taxonomy_scores(raw: dict[str, float]) -> dict[str, float]:
    scores = {category: raw[f"{category}.{option}"] for category, option in POSITIVE.items()}
    scores[TARGETS_PLAYER] = raw["target.player"]
    return scores


def app_scores(raw: dict[str, float]) -> dict[str, float]:
    return {FLAG: raw["insult.insult"]}


class LayaContestant:
    languages = frozenset(LANGS)
    license = "Apache-2.0"

    def __init__(self, name: str, questions: dict, scores, notes: str, device: str | None = None, router=None):
        self.name = name
        self.questions = questions
        self._scores = scores
        self.notes = notes
        self.device = device
        self._router = router
        # The questions, the laya version, the checkpoint revision and the routing setup all change the raw answers.
        self.version = (f"q{fingerprint(questions)}-laya{metadata.version('laya')}-r{LAYA_REVISION[:12]}"
                        "-default-multilingual")

    def _get_router(self):
        if self._router is None:
            self._router = build_router(device=self.device, revision=LAYA_REVISION)
            self._router.preload(["english", "multilingual"])
        return self._router

    def predict(self, texts: list[str], lang: str) -> list[dict[str, float]]:
        requests = [{"state": {STATE_KEY: text}, "questions": self.questions} for text in texts]
        return [flatten(result) for result in self._get_router().predict_batch(requests)]

    def to_scores(self, raw: dict[str, float]) -> dict[str, float]:
        return self._scores(raw)

    def unload(self) -> None:
        self._router = None
        free_gpu()


def laya_taxonomy(device: str | None = None, router=None) -> LayaContestant:
    return LayaContestant("laya", TAXONOMY_QUESTIONS, taxonomy_scores, TAXONOMY_NOTES, device=device, router=router)


def laya_app(device: str | None = None, router=None) -> LayaContestant:
    return LayaContestant("laya-app", APP_QUESTIONS, app_scores, APP_NOTES, device=device, router=router)
