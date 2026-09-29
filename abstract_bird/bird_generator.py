#!/usr/bin/env python3
"""Генератор абстрактной птицы по комментариям-воспоминаниям.

Программа читает один или несколько комментариев о желаниях или о том, что
человек хочет отпустить, эвристически анализирует их (без ML-моделей),
переводит анализ в художественную спецификацию ``BirdSpec`` и процедурно
рисует статичное изображение абстрактной птицы. Результат сохраняется в PNG,
параметры анализа и генерации — в JSON.

Исходные тексты никогда не выводятся в лог и не сохраняются в JSON без
флага ``--include-source-text``.
"""

# ============================================================================
# 1. Импорты
# ============================================================================
from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
import traceback
from collections import Counter
import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ============================================================================
# 2. Исключения
# ============================================================================


class BirdGeneratorError(Exception):
    """Базовая ошибка генератора с кодом завершения."""

    exit_code = 1


class InputDataError(BirdGeneratorError):
    """Ошибка входных данных (файл, формат, пустой текст)."""

    exit_code = 3


class ConfigurationError(BirdGeneratorError):
    """Ошибка конфигурационных файлов."""

    exit_code = 4


class RenderingError(BirdGeneratorError):
    """Ошибка построения изображения."""

    exit_code = 5


class OutputWriteError(BirdGeneratorError):
    """Ошибка записи результата на диск."""

    exit_code = 6


EXIT_OK = 0
EXIT_OTHER = 1
EXIT_CLI = 2

# ============================================================================
# 3. Dataclasses
# ============================================================================


@dataclass
class PreparedInput:
    """Подготовленные комментарии и статистика входа (без исходного текста в логах)."""

    comments: list[str]
    source_comment_count: int
    unique_comment_count: int
    duplicate_comment_count: int
    analyzed_comment_count: int
    truncated: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisDiagnostics:
    """Диагностика анализа. Не содержит исходного текста."""

    matched_word_count: int = 0
    matched_phrase_count: int = 0
    intensifier_count: int = 0
    diminisher_count: int = 0
    negation_count: int = 0
    punctuation_signal_count: int = 0
    uppercase_signal_count: int = 0
    emoji_signal_count: int = 0
    recognized_token_ratio: float = 0.0
    parameter_contributions: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class CommentAnalysis:
    """Промежуточный анализ одного комментария."""

    values: dict[str, float]
    evidence: dict[str, float]
    conflict: dict[str, float]
    signal_count: int
    word_count: int
    acquire_weight: float
    release_weight: float
    release_strength: float
    marker_count: int
    targets: list[str]
    keyword_candidates: list[str]
    theme_counts: dict[str, int]
    recognized_ratio: float
    diagnostics: AnalysisDiagnostics


@dataclass
class TextAnalysis:
    """Итоговый эмоционально-семантический профиль всех комментариев."""

    language: str
    source_comment_count: int
    unique_comment_count: int
    duplicate_comment_count: int
    analyzed_comment_count: int
    word_count: int
    truncated: bool
    intent_type: str
    intent_confidence: float
    intent_targets: list[str]
    release_strength: float
    valence: float
    arousal: float
    intensity: float
    dominance: float
    emotions: dict[str, float]
    score_confidence: dict[str, float]
    keywords: list[str]
    themes: list[str]
    visual_motifs: list[str]
    confidence: float


@dataclass
class BirdSpec:
    """Художественная спецификация птицы. Рендерер работает только с ней."""

    seed: int
    background_color: tuple[int, int, int]
    primary_color: tuple[int, int, int]
    secondary_color: tuple[int, int, int]
    accent_color: tuple[int, int, int]
    outline_color: tuple[int, int, int]
    body_scale: float
    body_elongation: float
    body_roundness: float
    body_tilt: float
    body_opacity: float
    wing_span: float
    wing_sweep: float
    wing_roundness: float
    wing_asymmetry: float
    tail_length: float
    tail_spread: float
    tail_drop: float
    head_scale: float
    head_lift: float
    beak_length: float
    beak_sharpness: float
    eye_scale: float
    feather_count: int
    feather_density: float
    fragmentation: float
    glow_intensity: float
    line_width: float
    particle_count: int
    open_space: float
    decorative_motifs: list[str]
    keywords: list[str]
    intent_type: str
    intent_strength: float = 0.0
    style: str = ""
    # --- слой художественной режиссуры ---
    primary_archetype: str = ""
    secondary_archetype: str = ""
    accent_archetype: str = ""
    archetype_weights: dict[str, float] = field(default_factory=dict)
    composition_type: str = ""
    abstraction_level: float = 0.4
    material_type: str = ""
    secondary_material_type: str = ""
    silhouette_type: str = ""
    feather_style: str = ""
    edge_style: str = ""
    texture_style: str = ""
    texture_strength: float = 0.0
    layer_count: int = 3
    negative_space: float = 0.3
    inner_void: float = 0.0
    silhouette_openness: float = 0.5
    visual_tension: float = 0.3
    focal_point: str = ""
    focal_strength: float = 0.5
    rare_feature: str = ""
    palette_strategy: str = ""
    wing_shape: str = ""
    tail_type: str = ""
    main_effect: str = ""
    accent_effect: str = ""


@dataclass
class ConfigBundle:
    """Загруженные и скомпилированные конфигурации."""

    lexicon: dict[str, Any]
    phrases: dict[str, Any]
    themes: dict[str, Any]
    mapping: dict[str, Any]
    versions: dict[str, str]
    hashes: dict[str, str]
    lexicon_exact: dict[str, dict[str, float]] = field(default_factory=dict)
    lexicon_prefixes: list[tuple[str, dict[str, float]]] = field(default_factory=list)
    phrase_patterns: list["Pattern"] = field(default_factory=list)
    intent_patterns: list["Pattern"] = field(default_factory=list)
    archetypes: dict[str, Any] = field(default_factory=dict)
    palettes: dict[str, Any] = field(default_factory=dict)
    compositions: dict[str, Any] = field(default_factory=dict)
    materials: dict[str, Any] = field(default_factory=dict)
    lemma_exact: dict[str, str] = field(default_factory=dict)
    lemma_prefixes: list[tuple[str, str]] = field(default_factory=list)
    theme_elements: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Pattern:
    """Скомпилированный шаблон фразы или маркера намерения."""

    text: str
    elements: tuple[str, ...]
    data: dict[str, Any]
    kind: str = "phrase"

    @property
    def size(self) -> int:
        return sum(1 for element in self.elements if element != "~")


@dataclass
class TokenStream:
    """Слова текста в нижнем регистре и границы предложений/сегментов."""

    words: list[str]
    clause_ids: list[int]
    segment_ids: list[int]


@dataclass
class GenerationOptions:
    """Параметры одного запуска генерации."""

    output_path: Path | None = None
    analysis_path: Path | None = None
    width: int = 1200
    height: int = 1200
    seed: int | None = None
    show_keywords: bool = True
    include_source_text: bool = False
    max_comments: int = 1000
    max_characters: int = 20000
    supersampling: int = 2
    background: str = "auto"
    debug: bool = False
    input_description: dict[str, Any] = field(default_factory=dict)
    overwrite: bool = False
    progress: Callable[[Image.Image, str], None] | None = None
    keyword_font: Path | None = None
    style: str = "none"
    candidates: int | None = None
    art_overrides: dict[str, str] = field(default_factory=dict)
    avoid_recent_similarity: bool = False
    recent_path: Path = Path("output/recent_specs.json")


@dataclass
class GenerationResult:
    """Результат генерации."""

    prepared: PreparedInput
    analysis: TextAnalysis
    diagnostics: AnalysisDiagnostics
    bird_spec: BirdSpec
    seed: int
    seed_source: str
    image_path: Path
    analysis_path: Path
    warnings: list[str]
    document: dict[str, Any]


@dataclass
class Shape:
    """Фигура в локальных координатах птицы (ось y направлена вверх)."""

    points: list[tuple[float, float]]
    fill: tuple[int, int, int, int] | None = None
    outline: tuple[int, int, int, int] | None = None
    width: float = 0.0
    closed: bool = True


# ============================================================================
# 4. Константы
# ============================================================================

SCHEMA_VERSION = "1.0"
PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "config"
DEFAULT_INPUT_PATH = Path("data/input/comments.csv")
DEFAULT_IMAGE_DIR = Path("output/images")
DEFAULT_ANALYSIS_DIR = Path("output/analysis")
DEFAULT_HISTORY_PATH = Path("data/history/requests.csv")
HISTORY_FIELDS: tuple[str, ...] = ("comment", "generated_at", "image", "analysis", "intent_type", "keywords")
NO_SOURCE_MESSAGE = (
    "Не указан источник текста. Передайте --text, --input или поместите "
    "CSV-файл в data/input/comments.csv."
)

CONFIG_FILES: dict[str, str] = {
    "emotion_lexicon_ru": "emotion_lexicon_ru.json",
    "phrase_rules_ru": "phrase_rules_ru.json",
    "themes_ru": "themes_ru.json",
    "artistic_mapping": "artistic_mapping.json",
    "visual_archetypes": "visual_archetypes.json",
    "palettes": "palettes.json",
    "compositions": "compositions.json",
    "materials": "materials.json",
}

BIPOLAR_PARAMS: tuple[str, ...] = ("valence", "arousal", "intensity", "dominance")
EMOTIONS: tuple[str, ...] = ("hope", "joy", "sadness", "anxiety", "anger", "tenderness", "surprise")
ALL_PARAMS: tuple[str, ...] = BIPOLAR_PARAMS + EMOTIONS
NEGATIVE_EMOTIONS: tuple[str, ...] = ("sadness", "anxiety", "anger")
POSITIVE_EMOTIONS: tuple[str, ...] = ("joy", "tenderness", "hope", "surprise")
INTENT_TYPES: tuple[str, ...] = ("acquire", "release", "mixed", "unclear")
BACKGROUND_MODES: tuple[str, ...] = ("auto", "solid", "gradient")
MIN_CANVAS = 512
MAX_CANVAS = 4096
MIN_TOTAL_CHARACTERS = 10
SEED_SEPARATOR = "\n\u241e\n"

EMOTION_NAMES_RU: dict[str, str] = {
    "hope": "надежда",
    "joy": "радость",
    "sadness": "грусть",
    "anxiety": "тревога",
    "anger": "гнев",
    "tenderness": "нежность",
    "surprise": "удивление",
}

SUPPORTED_MOTIFS: tuple[str, ...] = (
    "waves", "horizon", "open_wings", "stars", "leaves", "home", "music", "technology",
    "love", "fire", "clouds", "path", "separated_particles", "open_space", "faded_layers",
    "echo_lines", "warm_core", "vertical_shapes", "dark_field", "color_fragments",
    "organic_lines", "paired_shapes", "warm_glow", "sharp_geometry",
)

CLAUSE_BREAK_WORDS: frozenset[str] = frozenset({"но", "однако", "зато"})

TOKEN_RE = re.compile(r"[0-9a-zа-яё]+(?:-[0-9a-zа-яё]+)*|[.!?;…]+|[,:()\"«»—–\-]")
REPEATED_LETTERS_RE = re.compile(r"([a-zа-яё])\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")
UPPER_WORD_RE = re.compile(r"\b[A-ZА-ЯЁ]{3,}\b")
REPEATED_CHAR_RE = re.compile(r"([^\W\d_])\1{2,}")

RUSSIAN_STOPWORDS: frozenset[str] = frozenset(
    """
    я мне меня мной мой моя мое мои мою моей моих моим ты тебя тебе твой твоя он она оно мы нас нам
    вы вас они их им ним ней них тех те его ее это этот эта эти этого этом этой эту тот та те того тем том
    и а но или да же ли бы в во на по с со к ко у о об обо от до из за для при про над под без через
    что чтобы чтоб как так там тут здесь где когда уже еще все всё весь вся всех всем всему всего
    свою свой своя свои своей своих себя себе сам сама сами самой очень просто только тоже также
    быть был была было были будет буду будут есть нет не ни чем чего кто кого снова опять всегда
    никогда больше меньше хочу хочется хотелось хотел хотела хотим могу можно нужно надо пусть
    ведь вот ну даже лишь раз ничего какой какая какие какое сейчас потом наконец равно ещё
    который которая которые которого чем-то что-то где-то кто-то this the and to of in a is it i
    останусь останется остаться остаюсь остался осталась остались стану станет станешь станем стать
    становится стал стала стали буду будешь будем будете будь мог могла могли может можешь можем
    могут делать делаю сделать иметь имею являться является вообще все-таки именно какой-то как-то
    что-нибудь поэтому потому почему тогда теперь либо ради между вместо около после перед всего
    свое своего своим своему собой самое самый самая такой такая такое такие этих этим тому
    """.split()
)

# Каллиграфические шрифты для слов внутри птицы (только локальные, ничего не скачивается).
# Запись "путь#индекс" выбирает начертание внутри .ttc-коллекции.
CALLIGRAPHY_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Trattatello.ttf",        # широкое перо: каллиграфия и разборчивость
    "/System/Library/Fonts/Supplemental/Savoye LET.ttc#0",       # Savoye — тонкое перо
    "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc#1",   # Snell Roundhand Bold — английское перо
    "/System/Library/Fonts/Supplemental/Brush Script.ttf",       # кисть
    "C:/Windows/Fonts/segoesc.ttf",                              # Segoe Script
    "C:/Windows/Fonts/Gabriola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/System/Library/Fonts/Palatino.ttc#1",                      # Palatino Italic — запасной курсив
    "/System/Library/Fonts/Supplemental/Baskerville.ttc#2",
    "C:/Windows/Fonts/georgiai.ttf",
)

FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
)

LOGGER = logging.getLogger("bird_generator")


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Ограничить значение диапазоном [minimum, maximum]; NaN превращается в minimum."""
    if value != value:  # NaN
        return minimum
    return max(minimum, min(maximum, float(value)))


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


# ============================================================================
# 5. Конфигурация
# ============================================================================


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _load_json_config(path: Path, name: str) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise ConfigurationError(f"Конфигурационный файл не найден: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigurationError(f"Не удалось прочитать конфигурацию {path}: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"Конфигурация {path.name} не в кодировке UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Некорректный JSON в {path.name}: строка {exc.lineno}, столбец {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Конфигурация {path.name} должна быть JSON-объектом")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ConfigurationError(
            f"Неподдерживаемая версия схемы в {path.name}: {version!r}, ожидается {SCHEMA_VERSION!r}"
        )
    return data, _file_hash(raw)


def _require_section(data: dict[str, Any], section: str, kind: type, file_name: str) -> Any:
    if section not in data:
        raise ConfigurationError(f"В {file_name} отсутствует обязательный раздел '{section}'")
    value = data[section]
    if not isinstance(value, kind):
        raise ConfigurationError(
            f"Раздел '{section}' в {file_name} должен иметь тип {kind.__name__}"
        )
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_param_dict(entry: Any, where: str, allow_extra: Sequence[str] = ()) -> None:
    if not isinstance(entry, dict):
        raise ConfigurationError(f"{where}: ожидается объект с параметрами")
    for key, value in entry.items():
        if key in allow_extra:
            continue
        if key not in ALL_PARAMS:
            raise ConfigurationError(f"{where}: неизвестный параметр '{key}'")
        if not _is_number(value) or not -1.0 <= float(value) <= 1.0:
            raise ConfigurationError(f"{where}: значение '{key}' должно быть числом от -1 до 1")


def _validate_pattern_text(text: str, where: str) -> tuple[str, ...]:
    elements = tuple(normalize_for_matching(text).split())
    if not elements or elements[0] == "~" or elements[-1] == "~":
        raise ConfigurationError(f"{where}: некорректный шаблон '{text}'")
    for index, element in enumerate(elements):
        if element == "~" and elements[index + 1] == "~":
            raise ConfigurationError(f"{where}: два пропуска подряд в шаблоне '{text}'")
    return elements


def validate_lexicon(data: dict[str, Any]) -> None:
    """Проверить структуру эмоционального словаря."""
    words = _require_section(data, "words", dict, "emotion_lexicon_ru.json")
    if not words:
        raise ConfigurationError("Раздел 'words' в emotion_lexicon_ru.json пуст")
    for word, entry in words.items():
        _validate_param_dict(entry, f"emotion_lexicon_ru.json → '{word}'")


def validate_phrase_rules(data: dict[str, Any]) -> None:
    """Проверить структуру правил фраз и маркеров."""
    name = "phrase_rules_ru.json"
    phrases = _require_section(data, "phrases", dict, name)
    for phrase, entry in phrases.items():
        _validate_param_dict(entry, f"{name} → '{phrase}'", ("category", "modifier", "intent", "release_strength", "weight"))
        _validate_pattern_text(phrase, f"{name} → phrases")
        if "modifier" in entry and (not _is_number(entry["modifier"]) or not 0 < entry["modifier"] <= 3):
            raise ConfigurationError(f"{name} → '{phrase}': modifier должен быть числом от 0 до 3")
        if "intent" in entry and entry["intent"] not in ("acquire", "release"):
            raise ConfigurationError(f"{name} → '{phrase}': intent должен быть acquire или release")
    for section in ("intensifiers", "diminishers"):
        values = _require_section(data, section, dict, name)
        for word, factor in values.items():
            if not _is_number(factor) or not 0 < factor <= 3:
                raise ConfigurationError(f"{name} → {section} → '{word}': множитель должен быть числом от 0 до 3")
    for section in ("negations", "overcome_prefixes"):
        values = _require_section(data, section, list, name)
        if not all(isinstance(item, str) for item in values):
            raise ConfigurationError(f"{name} → {section}: ожидается список строк")
    markers = _require_section(data, "intent_markers", dict, name)
    for intent in ("acquire", "release"):
        group = markers.get(intent)
        if not isinstance(group, dict) or not group:
            raise ConfigurationError(f"{name} → intent_markers: отсутствует непустой раздел '{intent}'")
        for pattern, entry in group.items():
            _validate_pattern_text(pattern, f"{name} → intent_markers → {intent}")
            if not isinstance(entry, dict) or not _is_number(entry.get("weight")):
                raise ConfigurationError(f"{name} → '{pattern}': требуется числовой weight")
            if intent == "release" and "release_strength" in entry and not _is_number(entry["release_strength"]):
                raise ConfigurationError(f"{name} → '{pattern}': release_strength должен быть числом")
    emoji = _require_section(data, "emoji", dict, name)
    for symbol, entry in emoji.items():
        _validate_param_dict(entry, f"{name} → emoji")


def validate_themes(data: dict[str, Any]) -> None:
    """Проверить структуру тем."""
    name = "themes_ru.json"
    themes = _require_section(data, "themes", dict, name)
    if not themes:
        raise ConfigurationError(f"Раздел 'themes' в {name} пуст")
    for theme_id, theme in themes.items():
        where = f"{name} → {theme_id}"
        if not isinstance(theme, dict):
            raise ConfigurationError(f"{where}: ожидается объект")
        if not isinstance(theme.get("name"), str):
            raise ConfigurationError(f"{where}: требуется строковое поле name")
        words = theme.get("words")
        if not isinstance(words, list) or not words or not all(isinstance(w, str) for w in words):
            raise ConfigurationError(f"{where}: words должен быть непустым списком строк")
        motifs = theme.get("motifs")
        if not isinstance(motifs, list) or not all(isinstance(m, str) for m in motifs):
            raise ConfigurationError(f"{where}: motifs должен быть списком строк")
        for motif in motifs:
            if motif not in SUPPORTED_MOTIFS:
                raise ConfigurationError(f"{where}: неизвестный мотив '{motif}'")
        hue = theme.get("hue_range")
        if (
            not isinstance(hue, list) or len(hue) != 2 or not all(_is_number(h) for h in hue)
            or not all(0 <= h <= 360 for h in hue)
        ):
            raise ConfigurationError(f"{where}: hue_range должен быть списком из двух чисел 0–360")
    lemmas = _require_section(data, "lemmas", dict, name)
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in lemmas.items()):
        raise ConfigurationError(f"{name} → lemmas: ожидаются строковые ключи и значения")


def validate_mapping(data: dict[str, Any]) -> None:
    """Проверить структуру художественного маппинга."""
    name = "artistic_mapping.json"
    limits = _require_section(data, "limits", dict, name)
    for key in (
        "maximum_feather_count", "minimum_feather_count", "maximum_particle_count",
        "maximum_keyword_count", "maximum_theme_count", "maximum_motif_count",
        "maximum_rendered_keywords", "maximum_background_elements", "maximum_intent_targets",
    ):
        if not isinstance(limits.get(key), int) or isinstance(limits.get(key), bool) or limits[key] < 0:
            raise ConfigurationError(f"{name} → limits → {key}: требуется неотрицательное целое")
    if limits["minimum_feather_count"] > limits["maximum_feather_count"]:
        raise ConfigurationError(f"{name} → limits: minimum_feather_count больше maximum_feather_count")
    for section in ("random_variation", "aggregation", "analysis", "safe_ranges", "colors"):
        values = _require_section(data, section, dict, name)
        for key, value in values.items():
            if isinstance(value, (dict, list)):
                continue
            if not _is_number(value):
                raise ConfigurationError(f"{name} → {section} → {key}: ожидается число")
    if not 0 <= data["random_variation"].get("maximum", -1) <= 0.25:
        raise ConfigurationError(f"{name} → random_variation → maximum: допустимо 0–0.25")
    colors = data["colors"]
    hues = colors.get("emotion_hues")
    if not isinstance(hues, dict) or any(e not in hues or not _is_number(hues[e]) for e in EMOTIONS):
        raise ConfigurationError(f"{name} → colors → emotion_hues: нужны числовые оттенки для всех эмоций")
    parameters = _require_section(data, "parameters", dict, name)
    required = [
        "body_scale", "body_elongation", "body_roundness", "body_tilt", "body_opacity", "wing_span",
        "wing_sweep", "wing_roundness", "wing_asymmetry", "tail_length", "tail_spread", "tail_drop",
        "head_scale", "head_lift", "beak_length", "beak_sharpness", "eye_scale", "feather_density",
        "fragmentation", "glow_intensity", "line_width", "open_space", "feather_detail", "particle_amount",
    ]
    for key in required:
        rule = parameters.get(key)
        if not isinstance(rule, dict) or not _is_number(rule.get("bias")) or not isinstance(rule.get("terms"), dict):
            raise ConfigurationError(f"{name} → parameters → {key}: требуется bias и terms")
        if not all(_is_number(v) for v in rule["terms"].values()):
            raise ConfigurationError(f"{name} → parameters → {key}: коэффициенты должны быть числами")
    for section in ("emotion_motifs", "intent_motifs"):
        _require_section(data, section, dict, name)
    for emotion, rule in data["emotion_motifs"].items():
        if emotion not in EMOTIONS or not isinstance(rule, dict) or not _is_number(rule.get("threshold")):
            raise ConfigurationError(f"{name} → emotion_motifs → {emotion}: некорректное правило")
        for motif in rule.get("motifs", []):
            if motif not in SUPPORTED_MOTIFS:
                raise ConfigurationError(f"{name} → emotion_motifs: неизвестный мотив '{motif}'")
    for intent, motifs in data["intent_motifs"].items():
        if intent not in INTENT_TYPES or not isinstance(motifs, list):
            raise ConfigurationError(f"{name} → intent_motifs → {intent}: некорректное правило")
        for motif in motifs:
            if motif not in SUPPORTED_MOTIFS:
                raise ConfigurationError(f"{name} → intent_motifs: неизвестный мотив '{motif}'")


def validate_art_configs(archetypes: dict[str, Any], palettes: dict[str, Any], compositions: dict[str, Any],
                         materials: dict[str, Any]) -> None:
    """Проверить конфигурации художественной системы: разделы, типы и перекрёстные ссылки."""
    material_table = _require_section(materials, "materials", dict, "materials.json")
    for name, item in material_table.items():
        for key in ("alpha", "readability", "detail"):
            if not _is_number(item.get(key)):
                raise ConfigurationError(f"materials.json → {name}: '{key}' должен быть числом")
        for key in ("feather_style", "edge_style", "main_effect"):
            if not isinstance(item.get(key), str):
                raise ConfigurationError(f"materials.json → {name}: '{key}' должен быть строкой")
        if not isinstance(item.get("accent_effects"), list) or not item["accent_effects"]:
            raise ConfigurationError(f"materials.json → {name}: нужен непустой список accent_effects")
        if not isinstance(item.get("textures"), list) or not all(t in TEXTURE_STYLES for t in item["textures"]):
            raise ConfigurationError(f"materials.json → {name}: textures — список из {', '.join(TEXTURE_STYLES)}")
    for pair in _require_section(materials, "incompatible_pairs", list, "materials.json"):
        if not isinstance(pair, list) or len(pair) != 2 or any(m not in material_table for m in pair):
            raise ConfigurationError("materials.json → incompatible_pairs: пары существующих материалов")
    comp_table = _require_section(compositions, "compositions", dict, "compositions.json")
    silhouettes = _require_section(compositions, "silhouette_affinity", dict, "compositions.json")
    for name, item in comp_table.items():
        for key in ("tilt_degrees", "wing_sweep", "head_lift", "negative_space", "center"):
            value = item.get(key)
            if not isinstance(value, list) or len(value) != 2 or not all(_is_number(v) for v in value):
                raise ConfigurationError(f"compositions.json → {name}: '{key}' — список из двух чисел")
        if not isinstance(item.get("affinity"), dict) or not isinstance(item.get("silhouettes"), list):
            raise ConfigurationError(f"compositions.json → {name}: нужны affinity и silhouettes")
        if any(s not in silhouettes for s in item["silhouettes"]):
            raise ConfigurationError(f"compositions.json → {name}: неизвестный силуэт")
    arch_table = _require_section(archetypes, "archetypes", dict, "visual_archetypes.json")
    _require_section(archetypes, "mixing", dict, "visual_archetypes.json")
    rare = _require_section(archetypes, "rare_features", dict, "visual_archetypes.json")
    if len(arch_table) < 3:
        raise ConfigurationError("visual_archetypes.json: нужно не меньше трёх архетипов")
    for name, item in arch_table.items():
        if not _is_number(item.get("base")) or not isinstance(item.get("affinity"), dict):
            raise ConfigurationError(f"visual_archetypes.json → {name}: нужны base и affinity")
        if any(m not in material_table for m in item.get("materials", [])) or not item.get("materials"):
            raise ConfigurationError(f"visual_archetypes.json → {name}: неизвестный материал")
        if any(c not in comp_table for c in item.get("compositions", [])):
            raise ConfigurationError(f"visual_archetypes.json → {name}: неизвестная композиция")
        if any(m not in SUPPORTED_MOTIFS for m in item.get("motifs", [])):
            raise ConfigurationError(f"visual_archetypes.json → {name}: неизвестный мотив")
    if not _is_number(rare.get("probability")) or not isinstance(rare.get("features"), dict):
        raise ConfigurationError("visual_archetypes.json → rare_features: нужны probability и features")
    strategies = _require_section(palettes, "strategies", dict, "palettes.json")
    expected = {"analogous", "complementary_accent", "monochrome", "dark_field_light_accent", "warm_cool_contrast"}
    if not expected <= set(strategies):
        raise ConfigurationError(f"palettes.json: нужны стратегии {', '.join(sorted(expected))}")
    _require_section(palettes, "contrast", dict, "palettes.json")


def _compile_patterns(entries: dict[str, dict[str, Any]], kind: str) -> list[Pattern]:
    patterns = [
        Pattern(text=text, elements=tuple(normalize_for_matching(text).split()), data=dict(data), kind=kind)
        for text, data in entries.items()
    ]
    patterns.sort(key=lambda p: (-p.size, p.text))
    return patterns


def _split_exact_prefix(entries: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    exact: dict[str, Any] = {}
    prefixes: list[tuple[str, Any]] = []
    for key, value in entries.items():
        normalized = normalize_for_matching(key)
        if normalized.endswith("*"):
            prefixes.append((normalized[:-1], value))
        else:
            exact[normalized] = value
    prefixes.sort(key=lambda item: (-len(item[0]), item[0]))
    return exact, prefixes


def load_configs(config_dir: Path | None = None) -> ConfigBundle:
    """Загрузить, проверить и скомпилировать все конфигурационные файлы."""
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    loaded: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for key, file_name in CONFIG_FILES.items():
        loaded[key], hashes[key] = _load_json_config(directory / file_name, key)
    validate_lexicon(loaded["emotion_lexicon_ru"])
    validate_phrase_rules(loaded["phrase_rules_ru"])
    validate_themes(loaded["themes_ru"])
    validate_mapping(loaded["artistic_mapping"])
    validate_art_configs(loaded["visual_archetypes"], loaded["palettes"], loaded["compositions"], loaded["materials"])

    bundle = ConfigBundle(
        lexicon=loaded["emotion_lexicon_ru"],
        phrases=loaded["phrase_rules_ru"],
        themes=loaded["themes_ru"],
        mapping=loaded["artistic_mapping"],
        versions={key: str(loaded[key]["schema_version"]) for key in CONFIG_FILES},
        hashes=hashes,
    )
    bundle.archetypes = loaded["visual_archetypes"]
    bundle.palettes = loaded["palettes"]
    bundle.compositions = loaded["compositions"]
    bundle.materials = loaded["materials"]
    bundle.lexicon_exact, bundle.lexicon_prefixes = _split_exact_prefix(bundle.lexicon["words"])

    phrase_entries = {k: v for k, v in bundle.phrases["phrases"].items()}
    bundle.phrase_patterns = _compile_patterns(phrase_entries, "phrase")

    intent_entries: list[Pattern] = []
    markers = bundle.phrases["intent_markers"]
    for intent in ("acquire", "release"):
        for pattern in _compile_patterns(markers[intent], intent):
            intent_entries.append(pattern)
    for text, data in phrase_entries.items():
        if "intent" in data:
            intent_entries.append(
                Pattern(text=text, elements=tuple(normalize_for_matching(text).split()),
                        data={"weight": data.get("weight", 1.0), "release_strength": data.get("release_strength", 0.8)},
                        kind=data["intent"])
            )
    intent_entries.sort(key=lambda p: (-p.size, p.kind, p.text))
    bundle.intent_patterns = intent_entries

    bundle.lemma_exact, bundle.lemma_prefixes = _split_exact_prefix(bundle.themes["lemmas"])
    bundle.theme_elements = {
        theme_id: [normalize_for_matching(w) for w in theme["words"]]
        for theme_id, theme in bundle.themes["themes"].items()
    }
    global _ACTIVE_CONFIGS
    _ACTIVE_CONFIGS = bundle
    return bundle


# ============================================================================
# 6. Чтение входа
# ============================================================================


def _decode_bytes(raw: bytes, path: Path) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputDataError(
            f"Файл {path.name} не удалось прочитать как UTF-8 (позиция {exc.start}). "
            "Сохраните файл в кодировке UTF-8 или UTF-8-SIG."
        ) from exc


def read_txt(path: Path) -> list[str]:
    """Прочитать TXT: каждая непустая строка — отдельный комментарий."""
    text = _decode_bytes(path.read_bytes(), path)
    return [line for line in text.splitlines()]


def read_csv(path: Path, column: str = "comment") -> list[str]:
    """Прочитать CSV (UTF-8/UTF-8-SIG, кавычки, переносы строк внутри полей)."""
    text = _decode_bytes(path.read_bytes(), path)
    if not text.strip():
        return []
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        reader.fieldnames = fieldnames
        chosen = None
        for candidate in (column, "comment", "text"):
            if candidate in fieldnames:
                chosen = candidate
                break
        if chosen is None:
            raise InputDataError(
                f"В CSV-файле {path.name} нет колонки '{column}'. Доступные колонки: "
                f"{', '.join(fieldnames) if fieldnames else 'нет заголовка'}. Укажите --column."
            )
        if chosen != column:
            LOGGER.warning("Колонка '%s' не найдена, используется колонка '%s'", column, chosen)
        rows = []
        for row in reader:
            value = row.get(chosen)
            if isinstance(value, str):
                rows.append(value)
        return rows
    except csv.Error as exc:
        raise InputDataError(f"Ошибка разбора CSV-файла {path.name}: {exc}") from exc


def _extract_json_text(item: Any, text_field: str) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in (text_field, "comment", "text"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return None


def read_json(path: Path, text_field: str = "comment") -> list[str]:
    """Прочитать JSON: массив строк, массив объектов или объект с полем comments."""
    text = _decode_bytes(path.read_bytes(), path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputDataError(
            f"Некорректный JSON в {path.name}: строка {exc.lineno}, столбец {exc.colno}: {exc.msg}"
        ) from exc
    if isinstance(data, dict):
        if isinstance(data.get("comments"), list):
            items = data["comments"]
        else:
            single = _extract_json_text(data, text_field)
            if single is None:
                raise InputDataError(
                    f"JSON-объект в {path.name} должен содержать массив 'comments' "
                    f"или поле '{text_field}'"
                )
            items = [single]
    elif isinstance(data, list):
        items = data
    else:
        raise InputDataError(f"JSON в {path.name} должен быть массивом или объектом")
    result: list[str] = []
    skipped = 0
    for item in items:
        value = _extract_json_text(item, text_field)
        if value is None:
            skipped += 1
        else:
            result.append(value)
    if skipped:
        LOGGER.warning("Пропущено JSON-записей без текстового поля: %d", skipped)
    if items and not result:
        raise InputDataError(
            f"В JSON-файле {path.name} не найдено текстовое поле '{text_field}', 'comment' или 'text'"
        )
    return result


def read_comments(path: Path, column: str = "comment", text_field: str = "comment") -> list[str]:
    """Прочитать комментарии из TXT, CSV или JSON по расширению файла."""
    path = Path(path)
    if not path.exists():
        raise InputDataError(f"Входной файл не найден: {path}")
    if not path.is_file():
        raise InputDataError(f"Входной путь не является файлом: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".txt":
            return read_txt(path)
        if suffix == ".csv":
            return read_csv(path, column)
        if suffix == ".json":
            return read_json(path, text_field)
    except OSError as exc:
        raise InputDataError(f"Не удалось прочитать файл {path.name}: {exc.strerror}") from exc
    raise InputDataError(
        f"Неподдерживаемый формат файла '{suffix or 'без расширения'}'. Поддерживаются .txt, .csv, .json"
    )


# ============================================================================
# 7. Нормализация
# ============================================================================


def normalize_text(text: str) -> str:
    """Удалить крайние пробелы и сжать пробельные последовательности (регистр сохраняется)."""
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_for_matching(text: str) -> str:
    """Версия текста для словарного анализа: нижний регистр, ё→е, без растянутых букв."""
    lowered = text.lower().replace("ё", "е")
    return REPEATED_LETTERS_RE.sub(r"\1", lowered)


def tokenize(text: str) -> TokenStream:
    """Разбить текст (уже нормализованный для анализа) на слова и границы."""
    words: list[str] = []
    clause_ids: list[int] = []
    segment_ids: list[int] = []
    clause = 0
    segment = 0
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        if token[0].isalnum():
            if token in CLAUSE_BREAK_WORDS and words:
                clause += 1
                segment += 1
            words.append(token)
            clause_ids.append(clause)
            segment_ids.append(segment)
        elif token[0] in ".!?;…":
            clause += 1
            segment += 1
        else:
            segment += 1
    return TokenStream(words, clause_ids, segment_ids)


# ============================================================================
# 8. Дубли и лимиты
# ============================================================================


def _truncate_at_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    return cut.rstrip()


def prepare_comments(raw_comments: Sequence[str], max_comments: int = 1000, max_characters: int = 20000) -> PreparedInput:
    """Удалить пустые записи, нормализовать, удалить дубли и применить лимиты."""
    if max_comments < 1:
        raise InputDataError("max-comments должен быть положительным целым числом")
    if max_characters < MIN_TOTAL_CHARACTERS:
        raise InputDataError(f"max-characters должен быть не меньше {MIN_TOTAL_CHARACTERS}")
    warnings: list[str] = []
    normalized = [normalize_text(c) for c in raw_comments if isinstance(c, str)]
    non_empty = [c for c in normalized if c]
    source_count = len(non_empty)
    unique: list[str] = []
    seen: set[str] = set()
    for comment in non_empty:
        if comment not in seen:
            seen.add(comment)
            unique.append(comment)
    duplicate_count = source_count - len(unique)
    truncated = False

    limited = unique
    if len(limited) > max_comments:
        limited = limited[:max_comments]
        truncated = True
        warnings.append(
            f"Количество комментариев ограничено до {max_comments} (исключено {len(unique) - max_comments})"
        )

    selected: list[str] = []
    total = 0
    excluded = 0
    for comment in limited:
        if total + len(comment) <= max_characters:
            selected.append(comment)
            total += len(comment)
        elif not selected:
            cut = _truncate_at_boundary(comment, max_characters)
            selected.append(cut)
            total += len(cut)
            truncated = True
            warnings.append(f"Первый комментарий обрезан до {max_characters} символов")
        else:
            excluded += 1
    if excluded:
        truncated = True
        warnings.append(
            f"Из-за лимита {max_characters} символов целиком исключено комментариев: {excluded}"
        )

    if source_count == 0:
        raise InputDataError("Входные данные не содержат непустых комментариев")
    if total < MIN_TOTAL_CHARACTERS:
        raise InputDataError(
            f"Слишком короткий текст: {total} символов, требуется не меньше {MIN_TOTAL_CHARACTERS}"
        )
    return PreparedInput(
        comments=selected,
        source_comment_count=source_count,
        unique_comment_count=len(unique),
        duplicate_comment_count=duplicate_count,
        analyzed_comment_count=len(selected),
        truncated=truncated,
        warnings=warnings,
    )


def compute_seed(comments: Sequence[str]) -> int:
    """Детерминированный seed: SHA-256 от нормализованных уникальных комментариев.

    Встроенный ``hash()`` не используется, поэтому результат не зависит
    от PYTHONHASHSEED и процесса.
    """
    unique: list[str] = []
    seen: set[str] = set()
    for comment in comments:
        normalized = normalize_text(comment)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    digest = hashlib.sha256(SEED_SEPARATOR.join(unique).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ============================================================================
# 9. Определение языка
# ============================================================================


def detect_language(texts: Sequence[str]) -> str:
    """Определить язык по доле кириллических и латинских букв."""
    cyrillic = 0
    latin = 0
    for text in texts:
        for char in text:
            if "а" <= char.lower() <= "я" or char in "ёЁ":
                cyrillic += 1
            elif "a" <= char.lower() <= "z":
                latin += 1
    total = cyrillic + latin
    if total < 3:
        return "unknown"
    share = cyrillic / total
    if share >= 0.8:
        return "ru"
    if share <= 0.2:
        return "en"
    return "mixed"


# ============================================================================
# 10. Определение намерения (сопоставление шаблонов)
# ============================================================================


def _element_matches(word: str, element: str) -> bool:
    if element.endswith("*"):
        return word.startswith(element[:-1])
    return word == element


def match_pattern(stream: TokenStream, start: int, elements: Sequence[str], claimed: set[int]) -> list[int] | None:
    """Сопоставить шаблон начиная с позиции start; вернуть индексы совпавших слов."""
    words = stream.words
    indices: list[int] = []
    position = start
    k = 0
    while k < len(elements):
        element = elements[k]
        if element == "~":
            target = elements[k + 1]
            previous = indices[-1]
            found = None
            for j in range(position, min(position + 4, len(words))):
                if stream.clause_ids[j] != stream.clause_ids[previous]:
                    break
                if j not in claimed and _element_matches(words[j], target):
                    found = j
                    break
            if found is None:
                return None
            indices.append(found)
            position = found + 1
            k += 2
            continue
        if position >= len(words) or position in claimed:
            return None
        if indices and stream.segment_ids[position] != stream.segment_ids[indices[-1]]:
            return None
        if not _element_matches(words[position], element):
            return None
        indices.append(position)
        position += 1
        k += 1
    return indices


def find_pattern_matches(stream: TokenStream, patterns: Sequence[Pattern], claimed: set[int]) -> list[tuple[Pattern, list[int]]]:
    """Найти непересекающиеся совпадения; длинные шаблоны имеют приоритет."""
    matches: list[tuple[Pattern, list[int]]] = []
    for pattern in patterns:
        for start in range(len(stream.words)):
            if start in claimed:
                continue
            found = match_pattern(stream, start, pattern.elements, claimed)
            if found:
                matches.append((pattern, found))
                claimed.update(found)
    matches.sort(key=lambda item: item[1][0])
    return matches


def detect_intent_markers(stream: TokenStream, configs: ConfigBundle) -> tuple[list[tuple[Pattern, list[int]]], set[int]]:
    """Найти маркеры приобретения и отпускания в одном комментарии.

    Слабые маркеры желания («хочу», «хочется») не учитываются как
    приобретение, если сразу за ними следует маркер отпускания.
    """
    claimed: set[int] = set()
    matches = find_pattern_matches(stream, configs.intent_patterns, claimed)
    release_starts = [idx[0] for p, idx in matches if p.kind == "release"]
    kept: list[tuple[Pattern, list[int]]] = []
    for pattern, indices in matches:
        if pattern.kind == "acquire" and pattern.data.get("weak"):
            end = indices[-1]
            if any(
                end < start <= end + 3 and stream.clause_ids[start] == stream.clause_ids[end]
                for start in release_starts
            ):
                continue
        kept.append((pattern, indices))
    return kept, claimed


# ============================================================================
# 11. Анализ одного комментария
# ============================================================================


class _Scorer:
    """Накопитель вкладов в параметры одного комментария."""

    def __init__(self) -> None:
        self.raw = {p: 0.0 for p in ALL_PARAMS}
        self.evidence = {p: 0.0 for p in ALL_PARAMS}
        self.positive = {p: 0.0 for p in ALL_PARAMS}
        self.negative = {p: 0.0 for p in ALL_PARAMS}
        self.contributions: dict[str, dict[str, float]] = {}

    def add(self, param: str, value: float, source: str) -> None:
        if param not in self.raw or value == 0:
            return
        self.raw[param] += value
        if abs(value) >= 0.03:
            self.evidence[param] += min(1.0, abs(value) / 0.3)
        if value > 0:
            self.positive[param] += value
        else:
            self.negative[param] += -value
        bucket = self.contributions.setdefault(param, {})
        bucket[source] = bucket.get(source, 0.0) + value


def lookup_lexicon(word: str, configs: ConfigBundle) -> dict[str, float] | None:
    """Найти слово в словаре: сначала точное совпадение, затем самый длинный префикс."""
    entry = configs.lexicon_exact.get(word)
    if entry is not None:
        return entry
    for prefix, value in configs.lexicon_prefixes:
        if word.startswith(prefix) and len(word) >= len(prefix):
            return value
    return None


def lemmatize(word: str, configs: ConfigBundle) -> str:
    """Привести слово к канонической форме по небольшой таблице лемм (если есть)."""
    lemma = configs.lemma_exact.get(word)
    if lemma is not None:
        return lemma
    for prefix, value in configs.lemma_prefixes:
        if word.startswith(prefix):
            return value
    return word


def _is_theme_word(word: str, configs: ConfigBundle, exclude: Sequence[str] = ()) -> bool:
    return any(
        _element_matches(word, el)
        for theme_id, elements in configs.theme_elements.items() if theme_id not in exclude
        for el in elements
    )


def _apply_formal_signals(text: str, scorer: _Scorer, diagnostics: AnalysisDiagnostics, configs: ConfigBundle) -> int:
    """Учесть пунктуацию, заглавные буквы, повторы и эмодзи. Возвращает число сигналов."""
    signals = 0
    for match in re.finditer(r"[!?]+", text):
        run = match.group(0)
        exclamations = min(run.count("!"), 3)
        questions = min(run.count("?"), 3)
        if exclamations and questions:
            scorer.add("surprise", 0.3, "punctuation")
            scorer.add("arousal", 0.15, "punctuation")
            signals += 1
        elif exclamations:
            scorer.add("arousal", 0.15 * exclamations, "punctuation")
            scorer.add("intensity", 0.12 * exclamations, "punctuation")
            signals += 1
        elif questions:
            scorer.add("surprise", 0.08 * questions, "punctuation")
            scorer.add("anxiety", 0.05 * questions, "punctuation")
            scorer.add("dominance", -0.05 * questions, "punctuation")
            signals += 1
    ellipses = len(re.findall(r"\.{3,}|…", text))
    for _ in range(min(ellipses, 3)):
        scorer.add("dominance", -0.08, "punctuation")
        scorer.add("arousal", -0.05, "punctuation")
        scorer.add("anxiety", 0.04, "punctuation")
        signals += 1
    diagnostics.punctuation_signal_count += signals

    upper = min(len(UPPER_WORD_RE.findall(text)), 3)
    for _ in range(upper):
        scorer.add("intensity", 0.15, "uppercase")
        scorer.add("arousal", 0.12, "uppercase")
    diagnostics.uppercase_signal_count += upper

    repeated_chars = min(len(REPEATED_CHAR_RE.findall(text)), 3)
    for _ in range(repeated_chars):
        scorer.add("intensity", 0.1, "repetition")
        scorer.add("arousal", 0.05, "repetition")

    emoji_weight = float(configs.mapping["analysis"]["emoji_weight"])
    cleaned = text.replace("\ufe0f", "")
    emoji_hits = 0
    for symbol, entry in configs.phrases["emoji"].items():
        count = min(cleaned.count(symbol.replace("\ufe0f", "")), 3)
        for _ in range(count):
            for param, value in entry.items():
                scorer.add(param, float(value) * emoji_weight, "emoji")
            emoji_hits += 1
    diagnostics.emoji_signal_count += emoji_hits
    return signals + upper + repeated_chars + emoji_hits


def analyze_comment(comment: str, configs: ConfigBundle) -> CommentAnalysis:
    """Эвристически проанализировать один нормализованный комментарий."""
    settings = configs.mapping["analysis"]
    phrases_cfg = configs.phrases
    intensifiers = {normalize_for_matching(k): float(v) for k, v in phrases_cfg["intensifiers"].items()}
    diminishers = {normalize_for_matching(k): float(v) for k, v in phrases_cfg["diminishers"].items()}
    negations = {normalize_for_matching(w) for w in phrases_cfg["negations"]}
    overcome_prefixes = {normalize_for_matching(w) for w in phrases_cfg["overcome_prefixes"]}

    lower = normalize_for_matching(comment)
    stream = tokenize(lower)
    words = stream.words
    scorer = _Scorer()
    diagnostics = AnalysisDiagnostics()
    recognized: set[int] = set()

    # --- намерение -------------------------------------------------------
    intent_matches, intent_claimed = detect_intent_markers(stream, configs)
    acquire_weight = 0.0
    release_weight = 0.0
    release_strengths: list[float] = []
    release_scope: set[int] = set()
    marker_indices: set[int] = set()
    for pattern, indices in intent_matches:
        weight = float(pattern.data.get("weight", 1.0))
        marker_indices.update(indices)
        recognized.update(indices)
        if pattern.kind == "release":
            release_weight += weight
            release_strengths.append(float(pattern.data.get("release_strength", 0.8)))
            scorer.add("hope", float(settings["release_marker_hope_bonus"]) * weight, "intent")
            scorer.add("dominance", 0.1 * weight, "intent")
            scorer.add("valence", 0.05 * weight, "intent")
            clause = stream.clause_ids[indices[0]]
            for j in range(indices[0], min(indices[-1] + 6, len(words))):
                if stream.clause_ids[j] == clause:
                    release_scope.add(j)
        else:
            acquire_weight += weight
            scorer.add("hope", 0.1 * weight, "intent")
            scorer.add("intensity", 0.05 * weight, "intent")

    # --- устойчивые фразы (раньше отдельных слов) --------------------------
    phrase_claimed: set[int] = set()
    phrase_matches = find_pattern_matches(stream, configs.phrase_patterns, phrase_claimed)
    modifiers: dict[int, float] = {}
    for pattern, indices in phrase_matches:
        diagnostics.matched_phrase_count += 1
        recognized.update(indices)
        category = pattern.data.get("category")
        if category == "intensifier":
            diagnostics.intensifier_count += 1
        elif category == "diminisher":
            diagnostics.diminisher_count += 1
        in_release = indices[0] in release_scope
        for param, value in pattern.data.items():
            if param not in ALL_PARAMS:
                continue
            value = float(value)
            if in_release and param in NEGATIVE_EMOTIONS:
                value *= float(settings["release_scope_emotion_factor"])
            scorer.add(param, value, "phrases")
        if "modifier" in pattern.data:
            modifiers[indices[-1]] = float(pattern.data["modifier"])

    for index, word in enumerate(words):
        if index in phrase_claimed:
            continue
        if word in intensifiers:
            modifiers[index] = intensifiers[word]
            diagnostics.intensifier_count += 1
            scorer.add("intensity", 0.2 * (intensifiers[word] - 1.0) * 2, "modifiers")
            scorer.add("arousal", 0.05, "modifiers")
            recognized.add(index)
        elif word in diminishers:
            modifiers[index] = diminishers[word]
            diagnostics.diminisher_count += 1
            scorer.add("intensity", -0.15, "modifiers")
            scorer.add("dominance", -0.05, "modifiers")
            recognized.add(index)

    # --- отдельные слова -------------------------------------------------
    auto_intensity = float(settings["auto_intensity_factor"])
    release_emotion_factor = float(settings["release_scope_emotion_factor"])
    for index, word in enumerate(words):
        if index in phrase_claimed:
            continue
        entry = lookup_lexicon(word, configs)
        if entry is None:
            continue
        diagnostics.matched_word_count += 1
        recognized.add(index)
        segment = stream.segment_ids[index]
        factor = 1.0
        for j in (index - 1, index - 2):
            if j >= 0 and stream.segment_ids[j] == segment and j in modifiers:
                factor *= modifiers[j]
        negated = False
        overcome = False
        for j in (index - 1, index - 2, index - 3):
            if j >= 0 and stream.segment_ids[j] == segment and words[j] in negations and j not in phrase_claimed:
                negated = True
                overcome = j - 1 >= 0 and words[j - 1] in overcome_prefixes
                break
        in_release = index in release_scope
        is_negative_word = any(entry.get(e, 0) > 0.3 for e in NEGATIVE_EMOTIONS)
        if negated and not in_release:
            diagnostics.negation_count += 1

        for param, raw_value in entry.items():
            value = float(raw_value) * factor
            if in_release:
                if param in NEGATIVE_EMOTIONS:
                    value *= release_emotion_factor
                elif param == "valence" and value < 0:
                    value *= float(settings["release_scope_negative_valence_factor"])
                elif param == "dominance" and value < 0:
                    value *= float(settings["release_scope_negative_dominance_factor"])
                elif param == "arousal" and value > 0 and is_negative_word:
                    value *= 0.5
                elif param in ("joy", "tenderness") and not is_negative_word:
                    value *= 0.5
            elif negated:
                if overcome:
                    if param in EMOTIONS:
                        value *= float(settings["overcome_emotion_factor"])
                    elif param in ("valence", "dominance") and value < 0:
                        value = -value * 0.5
                    else:
                        value *= 0.3
                else:
                    if param in EMOTIONS:
                        value *= float(settings["negation_emotion_factor"])
                    elif param == "valence":
                        value = -value * 0.5
                    elif param == "dominance":
                        value = -value * 0.4
                    else:
                        value *= 0.4
            scorer.add(param, value, "lexicon")

        strongest = max((float(entry.get(e, 0.0)) for e in EMOTIONS), default=0.0)
        if strongest > 0:
            damp = 0.5 if (negated or in_release) else 1.0
            scorer.add("intensity", auto_intensity * strongest * factor * damp, "lexicon")
        if is_negative_word and in_release:
            scorer.add("hope", 0.1, "release_context")
        elif is_negative_word and negated and overcome:
            scorer.add("hope", 0.1, "negation")
            scorer.add("dominance", 0.2, "negation")

    # --- повторы слов ----------------------------------------------------
    content_counts = Counter(w for w in words if w not in RUSSIAN_STOPWORDS and len(w) > 2)
    repeated_words = sum(1 for count in content_counts.values() if count >= 3)
    repeated_words += sum(
        1 for i in range(1, len(words)) if words[i] == words[i - 1] and len(words[i]) > 1
    )
    for _ in range(min(repeated_words, 2)):
        scorer.add("intensity", 0.1, "repetition")

    formal_signals = _apply_formal_signals(comment, scorer, diagnostics, configs)

    # --- итоговые значения -----------------------------------------------
    bip_gain = float(settings["bipolar_gain"])
    emo_gain = float(settings["emotion_gain"])
    hope_base = float(settings["hope_baseline"])
    values: dict[str, float] = {}
    for param in BIPOLAR_PARAMS:
        values[param] = clamp(0.5 + 0.5 * math.tanh(bip_gain * scorer.raw[param]))
    for emotion in EMOTIONS:
        base = hope_base if emotion == "hope" else 0.0
        values[emotion] = clamp(1.0 - (1.0 - base) * math.exp(-emo_gain * max(0.0, scorer.raw[emotion])))

    conflict = {}
    for param in ALL_PARAMS:
        pos, neg = scorer.positive[param], scorer.negative[param]
        conflict[param] = (min(pos, neg) / (pos + neg)) if (pos + neg) > 0 else 0.0

    # --- цели намерения и ключевые слова ----------------------------------
    service_words = set(intensifiers) | set(diminishers) | negations
    targets: list[str] = []
    for pattern, indices in intent_matches:
        end = indices[-1]
        for j in range(end + 1, min(end + 5, len(words))):
            if stream.segment_ids[j] != stream.segment_ids[end]:
                break
            candidate = words[j]
            if candidate in RUSSIAN_STOPWORDS or candidate in service_words or len(candidate) < 3:
                continue
            if j in marker_indices and not _is_theme_word(candidate, configs):
                continue
            lemma = lemmatize(candidate, configs)
            if lemma not in targets:
                targets.append(lemma)
            break

    keyword_candidates: list[str] = []
    for index, word in enumerate(words):
        if len(word) < 3 or word in RUSSIAN_STOPWORDS or word in service_words or word.isdigit():
            continue
        if index in marker_indices and not _is_theme_word(word, configs, exclude=("release",)):
            continue
        keyword_candidates.append(lemmatize(word, configs))

    theme_counts: dict[str, int] = {}
    for theme_id, elements in configs.theme_elements.items():
        count = sum(1 for w in words if any(_element_matches(w, el) for el in elements))
        if count:
            theme_counts[theme_id] = count
    for index, word in enumerate(words):
        if index not in recognized and _is_theme_word(word, configs):
            recognized.add(index)

    diagnostics.recognized_token_ratio = (len(recognized) / len(words)) if words else 0.0
    diagnostics.parameter_contributions = {
        param: {src: _round(val) for src, val in sources.items()}
        for param, sources in scorer.contributions.items()
    }
    signal_count = (
        diagnostics.matched_word_count + diagnostics.matched_phrase_count + len(intent_matches)
        + formal_signals + diagnostics.intensifier_count + diagnostics.diminisher_count
    )
    if release_weight > 0:
        mean_strength = sum(release_strengths) / len(release_strengths)
        share = release_weight / (release_weight + acquire_weight)
        release_strength = clamp(share * mean_strength * min(1.0, 0.8 + 0.15 * len(release_strengths)))
    else:
        release_strength = 0.0
    return CommentAnalysis(
        values=values,
        evidence=dict(scorer.evidence),
        conflict=conflict,
        signal_count=signal_count,
        word_count=len(words),
        acquire_weight=acquire_weight,
        release_weight=release_weight,
        release_strength=release_strength,
        marker_count=len(intent_matches),
        targets=targets,
        keyword_candidates=keyword_candidates,
        theme_counts=theme_counts,
        recognized_ratio=diagnostics.recognized_token_ratio,
        diagnostics=diagnostics,
    )


# ============================================================================
# 12. Агрегация
# ============================================================================


def comment_weight(item: CommentAnalysis, aggregation: dict[str, Any]) -> float:
    """Вес комментария: учитывает сигналы и (слабо) длину, ограничен сверху."""
    weight = (
        float(aggregation["base_comment_weight"])
        + float(aggregation["signal_weight"]) * item.signal_count
        + float(aggregation["length_weight"]) * math.log1p(item.word_count)
    )
    return clamp(weight, float(aggregation["minimum_comment_weight"]), float(aggregation["maximum_comment_weight"]))


def aggregate_intent(items: Sequence[CommentAnalysis], weights: Sequence[float], aggregation: dict[str, Any]) -> tuple[str, float, float]:
    """Объединить намерения: вернуть (intent_type, intent_confidence, release_strength)."""
    acquire = release = unclear = 0.0
    markers = 0
    strengths: list[float] = []
    for item, weight in zip(items, weights):
        total = item.acquire_weight + item.release_weight
        markers += item.marker_count
        if total <= 0:
            unclear += weight
            strengths.append(0.0)
            continue
        acquire += weight * item.acquire_weight / total
        release += weight * item.release_weight / total
        strengths.append(item.release_strength)
    overall = acquire + release + unclear
    directed = acquire + release
    if overall <= 0 or directed < float(aggregation["unclear_intent_threshold"]) * overall:
        confidence = clamp(0.1 + 0.1 * min(markers, 2))
        return "unclear", _round(confidence), _round(0.0)
    minor_share = min(acquire, release) / directed
    if minor_share >= float(aggregation["mixed_intent_threshold"]):
        intent = "mixed"
        clarity = 0.85
    elif release > acquire:
        intent = "release"
        clarity = release / directed
    else:
        intent = "acquire"
        clarity = acquire / directed
    evidence = 1.0 - math.exp(-0.7 * markers)
    confidence = clamp((0.35 + 0.6 * evidence) * clarity * (1.0 - 0.5 * unclear / overall))
    mean_strength = sum(s * w for s, w in zip(strengths, weights)) / sum(weights)
    release_strength = 0.0 if intent == "acquire" and release == 0 else clamp(
        0.7 * mean_strength + 0.3 * max(strengths)
    )
    return intent, _round(confidence), _round(release_strength)


# ============================================================================
# 13. Эмоциональный анализ (итоговая сборка и уверенность)
# ============================================================================


LANGUAGE_CONFIDENCE = {"ru": 1.0, "mixed": 0.7, "en": 0.5, "unknown": 0.35}


def _score_confidence(items: Sequence[CommentAnalysis], language: str, settings: dict[str, Any]) -> dict[str, float]:
    base = float(settings["confidence_base"])
    gain = float(settings["confidence_gain"])
    rate = float(settings["confidence_rate"])
    penalty = float(settings["conflict_penalty"])
    coverage = sum(i.recognized_ratio for i in items) / len(items)
    coverage_factor = 0.75 + 0.25 * min(1.0, coverage * 3)
    count_bonus = min(0.1, 0.02 * (len(items) - 1))
    language_factor = LANGUAGE_CONFIDENCE.get(language, 0.35)
    result = {}
    for param in ALL_PARAMS:
        evidence = sum(i.evidence[param] for i in items)
        conflict = sum(i.conflict[param] for i in items) / len(items)
        value = base + gain * (1.0 - math.exp(-rate * evidence))
        value = value * coverage_factor * (1.0 - penalty * conflict) * language_factor
        if evidence > 0:
            value += count_bonus
        result[param] = _round(clamp(value, 0.05, 0.95))
    return result


def effective_scores(analysis: TextAnalysis) -> dict[str, float]:
    """Эффективные значения с учётом уверенности (низкая уверенность → нейтральность)."""
    result: dict[str, float] = {}
    for param in BIPOLAR_PARAMS:
        conf = analysis.score_confidence.get(param, 0.0)
        result[param] = clamp(0.5 + conf * (getattr(analysis, param) - 0.5))
    for emotion in EMOTIONS:
        conf = analysis.score_confidence.get(emotion, 0.0)
        result[emotion] = clamp(conf * analysis.emotions.get(emotion, 0.0))
    return result


# ============================================================================
# 14. Ключевые слова
# ============================================================================


def extract_keywords(items: Sequence[CommentAnalysis], configs: ConfigBundle) -> list[str]:
    """Выбрать до N ключевых слов по частоте, частоте по комментариям и тематичности."""
    limit = int(configs.mapping["limits"]["maximum_keyword_count"])
    frequency: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    order = 0
    for item in items:
        frequency.update(item.keyword_candidates)
        for word in dict.fromkeys(item.keyword_candidates):
            document_frequency[word] += 1
        for word in item.keyword_candidates:
            if word not in first_seen:
                first_seen[word] = order
                order += 1
    scores: dict[str, float] = {}
    for word, count in frequency.items():
        score = count + 1.5 * document_frequency[word]
        if _is_theme_word(word, configs):
            score += 1.0
        if lookup_lexicon(word, configs) is not None:
            score += 0.3
        scores[word] = score
    ranked = sorted(scores, key=lambda w: (-scores[w], first_seen[w], w))
    return ranked[:limit]


# ============================================================================
# 15. Темы и мотивы
# ============================================================================


def detect_themes(items: Sequence[CommentAnalysis], weights: Sequence[float], configs: ConfigBundle) -> list[str]:
    """Выбрать темы по взвешенному числу совпадений; равенства — по порядку конфигурации."""
    limit = int(configs.mapping["limits"]["maximum_theme_count"])
    theme_order = list(configs.themes["themes"])
    scores: dict[str, float] = {}
    for item, weight in zip(items, weights):
        for theme_id, count in item.theme_counts.items():
            scores[theme_id] = scores.get(theme_id, 0.0) + weight * min(count, 2)
    ranked = sorted(scores, key=lambda t: (-scores[t], theme_order.index(t)))
    return ranked[:limit]


def select_motifs(themes: Sequence[str], intent_type: str, effective: dict[str, float], configs: ConfigBundle) -> list[str]:
    """Собрать мотивы по кругу из групп: намерение, темы, эмоции."""
    mapping = configs.mapping
    limit = int(mapping["limits"]["maximum_motif_count"])
    groups: list[list[str]] = [list(mapping["intent_motifs"].get(intent_type, []))]
    for theme_id in themes:
        groups.append(list(configs.themes["themes"][theme_id]["motifs"]))
    for emotion in EMOTIONS:
        rule = mapping["emotion_motifs"].get(emotion)
        if rule and effective.get(emotion, 0.0) >= float(rule["threshold"]):
            groups.append(list(rule.get("motifs", [])))
    motifs: list[str] = []
    depth = max((len(g) for g in groups), default=0)
    for level in range(depth):
        for group in groups:
            if level < len(group) and group[level] not in motifs:
                motifs.append(group[level])
    return motifs[:limit]


# ============================================================================
# 16. TextAnalysis
# ============================================================================


def analyze_comments(prepared: PreparedInput, configs: ConfigBundle) -> tuple[TextAnalysis, AnalysisDiagnostics, list[str]]:
    """Проанализировать комментарии по отдельности и объединить в общий профиль."""
    warnings: list[str] = []
    comments = prepared.comments
    language = detect_language(comments)
    if language != "ru":
        warnings.append(
            f"Язык текста определён как '{language}': полноценный анализ гарантируется только для русского"
        )
    items = [analyze_comment(comment, configs) for comment in comments]
    aggregation = configs.mapping["aggregation"]
    weights = [comment_weight(item, aggregation) for item in items]
    total_weight = sum(weights)
    mean_w = float(aggregation["emotion_mean_weight"])
    max_w = float(aggregation["emotion_max_weight"])

    values: dict[str, float] = {}
    for param in BIPOLAR_PARAMS:
        values[param] = sum(i.values[param] * w for i, w in zip(items, weights)) / total_weight
    for emotion in EMOTIONS:
        mean = sum(i.values[emotion] * w for i, w in zip(items, weights)) / total_weight
        values[emotion] = mean_w * mean + max_w * max(i.values[emotion] for i in items)

    score_confidence = _score_confidence(items, language, configs.mapping["analysis"])
    intent_type, intent_confidence, release_strength = aggregate_intent(items, weights, aggregation)

    target_counts: Counter[str] = Counter()
    target_order: dict[str, int] = {}
    for item in items:
        for target in item.targets:
            target_counts[target] += 1
            target_order.setdefault(target, len(target_order))
    max_targets = int(configs.mapping["limits"]["maximum_intent_targets"])
    targets = sorted(target_counts, key=lambda t: (-target_counts[t], target_order[t]))[:max_targets]

    keywords = extract_keywords(items, configs)
    themes = detect_themes(items, weights, configs)

    diagnostics = AnalysisDiagnostics()
    contributions: dict[str, dict[str, float]] = {}
    for item in items:
        d = item.diagnostics
        diagnostics.matched_word_count += d.matched_word_count
        diagnostics.matched_phrase_count += d.matched_phrase_count
        diagnostics.intensifier_count += d.intensifier_count
        diagnostics.diminisher_count += d.diminisher_count
        diagnostics.negation_count += d.negation_count
        diagnostics.punctuation_signal_count += d.punctuation_signal_count
        diagnostics.uppercase_signal_count += d.uppercase_signal_count
        diagnostics.emoji_signal_count += d.emoji_signal_count
        for param, sources in d.parameter_contributions.items():
            bucket = contributions.setdefault(param, {})
            for source, value in sources.items():
                bucket[source] = _round(bucket.get(source, 0.0) + value)
    total_words = sum(i.word_count for i in items)
    diagnostics.recognized_token_ratio = _round(
        sum(i.recognized_ratio * i.word_count for i in items) / total_words if total_words else 0.0
    )
    diagnostics.parameter_contributions = {k: contributions[k] for k in sorted(contributions)}

    mean_conf = sum(score_confidence.values()) / len(score_confidence)
    total_signals = sum(i.signal_count for i in items)
    coverage = min(1.0, diagnostics.recognized_token_ratio * 3)
    confidence = clamp(
        LANGUAGE_CONFIDENCE.get(language, 0.35)
        * (0.4 * mean_conf + 0.3 * min(1.0, total_signals / 8) + 0.3 * coverage)
    )

    analysis = TextAnalysis(
        language=language,
        source_comment_count=prepared.source_comment_count,
        unique_comment_count=prepared.unique_comment_count,
        duplicate_comment_count=prepared.duplicate_comment_count,
        analyzed_comment_count=prepared.analyzed_comment_count,
        word_count=total_words,
        truncated=prepared.truncated,
        intent_type=intent_type,
        intent_confidence=intent_confidence,
        intent_targets=targets,
        release_strength=release_strength,
        valence=_round(clamp(values["valence"])),
        arousal=_round(clamp(values["arousal"])),
        intensity=_round(clamp(values["intensity"])),
        dominance=_round(clamp(values["dominance"])),
        emotions={e: _round(clamp(values[e])) for e in EMOTIONS},
        score_confidence=score_confidence,
        keywords=keywords,
        themes=themes,
        visual_motifs=[],
        confidence=_round(confidence),
    )
    analysis.visual_motifs = select_motifs(themes, intent_type, effective_scores(analysis), configs)
    if not keywords:
        warnings.append("Не удалось выделить ключевые слова")
    if confidence < 0.3:
        warnings.append("Низкая уверенность анализа: птица будет ближе к нейтральной")
    return analysis, diagnostics, warnings


# ============================================================================
# 17. BirdSpec
# ============================================================================


def mapping_signals(analysis: TextAnalysis) -> dict[str, float]:
    """Сигналы для линейного маппинга: эффективные значения и сила намерения."""
    eff = effective_scores(analysis)
    signals: dict[str, float] = {}
    for param in BIPOLAR_PARAMS:
        signals[param] = eff[param] - 0.5
    for emotion in EMOTIONS:
        signals[emotion] = eff[emotion]
    conf = analysis.intent_confidence
    strength = analysis.release_strength
    signals["acquire"] = conf * (1.0 - strength) if analysis.intent_type in ("acquire", "mixed") else 0.0
    if analysis.intent_type == "release":
        signals["release"] = conf * max(strength, 0.5)
    elif analysis.intent_type == "mixed":
        signals["release"] = conf * max(strength, 0.3)
    else:
        signals["release"] = 0.0
    signals["mixed"] = conf if analysis.intent_type == "mixed" else 0.0
    signals["detail"] = min(1.0, math.log1p(analysis.word_count) / math.log(60))
    for theme_id in analysis.themes:
        signals[f"theme_{theme_id}"] = 1.0
    for motif in analysis.visual_motifs:
        signals[f"motif_{motif}"] = 1.0
    return signals


def _linear(rule: dict[str, Any], signals: dict[str, float]) -> float:
    total = float(rule["bias"])
    for name, coefficient in rule["terms"].items():
        total += float(coefficient) * signals.get(name, 0.0)
    return total


def hsl_to_rgb(hue: float, saturation: float, lightness: float) -> tuple[int, int, int]:
    """Преобразовать HSL (оттенок в градусах) в RGB 0–255."""
    r, g, b = colorsys.hls_to_rgb((hue % 360.0) / 360.0, clamp(lightness), clamp(saturation))
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def rgb_lightness(color: Sequence[int]) -> float:
    """Светлота HSL цвета RGB."""
    return colorsys.rgb_to_hls(color[0] / 255, color[1] / 255, color[2] / 255)[1]


def _circular_mean(hues: Sequence[tuple[float, float]]) -> float | None:
    x = sum(math.cos(math.radians(h)) * w for h, w in hues)
    y = sum(math.sin(math.radians(h)) * w for h, w in hues)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return None
    return math.degrees(math.atan2(y, x)) % 360.0


def build_palette(analysis: TextAnalysis, signals: dict[str, float], colors: dict[str, Any],
                  theme_hues: dict[str, list[float]], motifs: Sequence[str], rng: random.Random) -> dict[str, tuple[int, int, int]]:
    """Построить согласованную палитру из темы, эмоций и seed-вариации."""
    jitter = float(colors.get("hue_jitter", 0.0))
    emotion_hues = colors["emotion_hues"]
    # Базовый оттенок задаёт главная тема (усреднение далёких оттенков тем дало бы
    # «грязный» промежуточный цвет); эмоции сдвигают его лишь ограниченно.
    emotion_samples: list[tuple[float, float]] = []
    for emotion in EMOTIONS:
        weight = signals[emotion] * float(colors["emotion_hue_weight"])
        if emotion == "hope":
            weight *= 0.5
        if weight > 0.02:
            emotion_samples.append((float(emotion_hues[emotion]), weight))
    emotion_hue = _circular_mean(emotion_samples)
    emotion_weight = sum(w for _, w in emotion_samples)
    if analysis.themes:
        low, high = theme_hues.get(analysis.themes[0], [colors["neutral_hue"], colors["neutral_hue"]])
        base_hue = low + (high - low) * rng.random()
        if emotion_hue is not None:
            difference = (emotion_hue - base_hue + 180.0) % 360.0 - 180.0
            base_hue += difference * clamp(float(colors["emotion_hue_shift_rate"]) * emotion_weight, 0.0, float(colors["emotion_hue_shift_max"]))
    else:
        base_hue = emotion_hue if emotion_hue is not None else float(colors["neutral_hue"])
    base_hue = (base_hue + rng.uniform(-jitter, jitter)) % 360.0

    v, a, i = signals["valence"], signals["arousal"], signals["intensity"]
    s_min, s_max = float(colors["saturation_min"]), float(colors["saturation_max"])
    l_min, l_max = float(colors["lightness_min"]), float(colors["lightness_max"])
    saturation = clamp(
        0.45 + 0.6 * i + 0.2 * signals["joy"] + 0.1 * signals["anger"]
        - 0.3 * signals["sadness"] - 0.25 * signals["tenderness"] + 0.3 * v,
        s_min, s_max,
    )
    lightness = clamp(
        0.5 + 0.45 * v + 0.12 * signals["hope"] + 0.12 * signals["tenderness"]
        - 0.1 * signals["sadness"] - 0.06 * signals["anger"],
        l_min, l_max,
    )
    dark_background = v < -0.03 or "dark_field" in motifs
    min_contrast = float(colors["minimum_background_contrast"])
    if dark_background:
        bg_light = clamp(0.1 + 0.05 * (v + 0.5), 0.07, 0.16)
        lightness = max(lightness, bg_light + min_contrast + 0.08)
        bg_hue = (base_hue * 0.4 + 225 * 0.6) if signals["sadness"] > 0.2 else base_hue
        background = hsl_to_rgb(bg_hue, 0.35 * saturation + 0.1, bg_light)
    else:
        bg_light = clamp(0.93 + 0.03 * v, 0.88, 0.97)
        lightness = min(lightness, bg_light - min_contrast - 0.02)
        background = hsl_to_rgb(base_hue + 15, 0.25 + 0.2 * saturation, bg_light)

    primary = hsl_to_rgb(base_hue, saturation, lightness)
    # Грусть охлаждает основу: смешивание в RGB с холодным синим (а не поворот
    # оттенка по кругу, который из тёплого цвета дал бы пурпурный).
    cool_mix = clamp(float(colors["sadness_cool_mix"]) * signals["sadness"], 0.0, 0.55)
    if cool_mix > 0:
        primary = mix(primary, hsl_to_rgb(float(emotion_hues["sadness"]), saturation, lightness), cool_mix)
    if a > 0.1 or signals["anger"] > 0.3 or signals["surprise"] > 0.3:
        secondary_hue = base_hue + 180
        secondary = hsl_to_rgb(secondary_hue, saturation * 0.75, clamp(lightness + 0.06, l_min, l_max))
    else:
        direction = 1 if rng.random() < 0.5 else -1
        secondary_hue = base_hue + 32 * direction
        secondary = hsl_to_rgb(secondary_hue, saturation * 0.9, clamp(lightness + 0.1, l_min, l_max + 0.08))

    strongest = max(EMOTIONS, key=lambda e: (signals[e], -EMOTIONS.index(e)))
    if signals[strongest] < 0.12:
        strongest = "hope"
    accent_hue = float(emotion_hues[strongest])
    accent_light = 0.72 if strongest in ("hope", "joy", "tenderness") else 0.58
    accent = hsl_to_rgb(accent_hue + rng.uniform(-jitter / 2, jitter / 2), 0.78, accent_light)

    anger_boost = signals["anger"]
    if dark_background:
        outline = hsl_to_rgb(base_hue, saturation * 0.5, clamp(0.82 + 0.1 * anger_boost, 0, 0.95))
    else:
        outline = hsl_to_rgb(base_hue, clamp(saturation * 0.7), clamp(0.2 - 0.1 * anger_boost, 0.06, 0.3))
    return {
        "background": background,
        "primary": primary,
        "secondary": secondary,
        "accent": accent,
        "outline": outline,
    }


def analysis_to_bird_spec(
    analysis: TextAnalysis,
    seed: int,
    mapping_config: dict[str, object],
    theme_hues: dict[str, list[float]] | None = None,
) -> BirdSpec:
    """Преобразовать анализ в художественную спецификацию птицы.

    80–90% значения каждого параметра определяется анализом, небольшая
    вариация добавляется локальным генератором ``random.Random(seed)``.
    """
    rng = random.Random(seed)
    mapping: dict[str, Any] = mapping_config  # type: ignore[assignment]
    variation = float(mapping["random_variation"]["maximum"])
    signals = mapping_signals(analysis)
    parameters = mapping["parameters"]
    limits = mapping["limits"]

    values: dict[str, float] = {}
    for name in (
        "body_scale", "body_elongation", "body_roundness", "body_tilt", "body_opacity", "wing_span",
        "wing_sweep", "wing_roundness", "wing_asymmetry", "tail_length", "tail_spread", "tail_drop",
        "head_scale", "head_lift", "beak_length", "beak_sharpness", "eye_scale", "feather_density",
        "fragmentation", "glow_intensity", "line_width", "open_space", "feather_detail", "particle_amount",
    ):
        values[name] = _round(clamp(_linear(parameters[name], signals) + rng.uniform(-variation, variation)))

    min_feathers = int(limits["minimum_feather_count"])
    max_feathers = int(limits["maximum_feather_count"])
    feather_count = int(round(min_feathers + (max_feathers - min_feathers) * values["feather_detail"]))
    particle_count = int(round(int(limits["maximum_particle_count"]) * values["particle_amount"]))
    motifs = list(analysis.visual_motifs[: int(limits["maximum_motif_count"])])
    if "separated_particles" in motifs:
        particle_count = max(particle_count, 24)
    particle_count = min(particle_count, int(limits["maximum_particle_count"]))

    colors = dict(mapping["colors"])
    colors["hue_jitter"] = float(mapping["random_variation"].get("hue_degrees", 10.0))
    palette = build_palette(analysis, signals, colors, theme_hues or {}, motifs, rng)

    return BirdSpec(
        seed=seed,
        background_color=palette["background"],
        primary_color=palette["primary"],
        secondary_color=palette["secondary"],
        accent_color=palette["accent"],
        outline_color=palette["outline"],
        body_scale=values["body_scale"],
        body_elongation=values["body_elongation"],
        body_roundness=values["body_roundness"],
        body_tilt=values["body_tilt"],
        body_opacity=values["body_opacity"],
        wing_span=values["wing_span"],
        wing_sweep=values["wing_sweep"],
        wing_roundness=values["wing_roundness"],
        wing_asymmetry=values["wing_asymmetry"],
        tail_length=values["tail_length"],
        tail_spread=values["tail_spread"],
        tail_drop=values["tail_drop"],
        head_scale=values["head_scale"],
        head_lift=values["head_lift"],
        beak_length=values["beak_length"],
        beak_sharpness=values["beak_sharpness"],
        eye_scale=values["eye_scale"],
        feather_count=feather_count,
        feather_density=values["feather_density"],
        fragmentation=values["fragmentation"],
        glow_intensity=values["glow_intensity"],
        line_width=values["line_width"],
        particle_count=particle_count,
        open_space=values["open_space"],
        decorative_motifs=motifs,
        keywords=list(analysis.keywords[: int(limits["maximum_rendered_keywords"])]),
        intent_type=analysis.intent_type,
        intent_strength=_round(analysis.intent_confidence),
    )


# ============================================================================
# 18. Цветовые функции
# ============================================================================


def mix(color_a: Sequence[int], color_b: Sequence[int], t: float) -> tuple[int, int, int]:
    """Линейно смешать два RGB-цвета."""
    t = clamp(t)
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(color_a[:3], color_b[:3]))  # type: ignore[return-value]


def rgba(color: Sequence[int], alpha: float) -> tuple[int, int, int, int]:
    """Добавить альфа-канал (0..1) к RGB-цвету."""
    return (int(color[0]), int(color[1]), int(color[2]), int(round(clamp(alpha) * 255)))


def shift_hue(color: Sequence[int], degrees: float, lightness_delta: float = 0.0, saturation_scale: float = 1.0) -> tuple[int, int, int]:
    """Сдвинуть оттенок и светлоту RGB-цвета."""
    h, l, s = colorsys.rgb_to_hls(color[0] / 255, color[1] / 255, color[2] / 255)
    return hsl_to_rgb(h * 360 + degrees, s * saturation_scale, clamp(l + lightness_delta))


# ============================================================================
# 19. Геометрические функции
# ============================================================================

Point = tuple[float, float]


def cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, steps: int = 24) -> list[Point]:
    """Точки кубической кривой Безье."""
    points = []
    for index in range(steps + 1):
        t = index / steps
        u = 1 - t
        x = u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0]
        y = u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1]
        points.append((x, y))
    return points


def rotate_point(point: Point, angle: float, origin: Point = (0.0, 0.0)) -> Point:
    """Повернуть точку против часовой стрелки на angle радиан вокруг origin."""
    x, y = point[0] - origin[0], point[1] - origin[1]
    c, s = math.cos(angle), math.sin(angle)
    return (origin[0] + x * c - y * s, origin[1] + x * s + y * c)


def ellipse_points(center: Point, rx: float, ry: float, angle: float = 0.0, steps: int = 28) -> list[Point]:
    """Точки повёрнутого эллипса."""
    points = []
    for index in range(steps):
        t = 2 * math.pi * index / steps
        points.append(rotate_point((center[0] + rx * math.cos(t), center[1] + ry * math.sin(t)), angle, center))
    return points


def feather_points(base: Point, angle: float, length: float, width: float, kind: str) -> list[Point]:
    """Полигон пера заданной формы, растущего из base в направлении angle."""
    local: list[Point]
    if kind == "ellipse":
        local = [((length / 2) * (1 + math.cos(t)), (width / 2) * math.sin(t))
                 for t in (2 * math.pi * i / 20 for i in range(20))]
    elif kind == "drop":
        local = []
        for i in range(11):
            s = i / 10
            local.append((length * s, (width / 2) * math.sin(math.pi * s ** 0.6)))
        for i in range(10, -1, -1):
            s = i / 10
            local.append((length * s, -(width / 2) * math.sin(math.pi * s ** 0.6)))
    elif kind == "leaf":
        local = []
        for i in range(11):
            s = i / 10
            local.append((length * s, (width / 2) * math.sin(math.pi * s)))
        for i in range(9, 0, -1):
            s = i / 10
            local.append((length * s, -(width / 2) * math.sin(math.pi * s)))
    elif kind == "triangle":
        local = [(0.0, width / 2), (length, 0.0), (0.0, -width / 2)]
    elif kind == "arch":  # стрельчатая арка (готика)
        half = width / 2
        side = [(length * (0.55 + 0.45 * math.sin(i / 8 * math.pi / 2)), half * math.cos(i / 8 * math.pi / 2)) for i in range(9)]
        local = [(0.0, half)] + side + [(x, -y) for x, y in reversed(side[:-1])] + [(0.0, -half)]
    elif kind == "block":  # прямоугольный блок (брутализм)
        local = [(0.0, width / 2), (length, width / 2), (length, -width / 2), (0.0, -width / 2)]
    else:  # lance / polygon
        local = [(0.0, width / 3), (length * 0.65, width / 2), (length, 0.0),
                 (length * 0.65, -width / 2), (0.0, -width / 3)]
    return [rotate_point((base[0] + x, base[1] + y), angle, base) for x, y in local]


def style_kinds(spec: "BirdSpec") -> tuple[str, ...]:
    """Формы перьев, заданные стилем птицы (пусто — форма по округлости)."""
    style = STYLES.get(spec.style)
    return style.feather_kinds if style else ()


def feather_kind(roundness: float, rng: random.Random, kinds: Sequence[str] = ()) -> str:
    """Выбрать форму пера: из форм стиля, если они заданы, иначе по округлости."""
    if kinds:
        return kinds[0] if len(kinds) == 1 else kinds[rng.randrange(len(kinds))]
    if roundness > 0.62:
        return "ellipse" if rng.random() < 0.55 else "drop"
    if roundness > 0.38:
        return "leaf" if rng.random() < 0.6 else "drop"
    return "triangle" if rng.random() < 0.5 else "lance"


def _normalize(vector: Point) -> Point:
    length = math.hypot(vector[0], vector[1]) or 1.0
    return (vector[0] / length, vector[1] / length)


# ============================================================================
# 19a. Художественно-архитектурные стили
# ============================================================================

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class StyleProfile:
    """Художественный стиль: палитра, форма перьев, контур, фон, орнамент и фактура.

    Стиль меняет «почерк» птицы, но не её эмоциональную геометрию: размах,
    наклон, хвост, асимметрия и частицы по-прежнему задаются анализом текста.
    Эмоциональный цвет сохраняется частично (``palette_strength`` < 1).
    """

    name: str
    title: str
    description: str
    palette: tuple[RGB, ...]
    palette_strength: float
    background: RGB | None
    outline: RGB | None
    line_scale: float
    feather_kinds: tuple[str, ...]
    roundness_shift: float
    feather_scale: float
    particle_scale: float
    glow: bool
    chest_feathers: bool
    background_mode: str
    background_motifs: bool
    ornament: str
    post: str
    saturation_scale: float = 1.0


STYLES: dict[str, StyleProfile] = {
    "art_nouveau": StyleProfile(
        "art_nouveau", "Модерн (ар-нуво)",
        "Растительные перья-листья, текучие линии «удар бича», приглушённые оливковые и золотые тона, арочная рамка",
        ((128, 140, 70), (205, 165, 70), (60, 120, 115), (190, 120, 110), (236, 226, 196)), 0.55,
        (238, 228, 200), (70, 60, 40), 1.4, ("leaf",), 0.25, 1.0, 0.8, True, True, "gradient", True, "whiplash", ""),
    "art_deco": StyleProfile(
        "art_deco", "Ар-деко",
        "Золото на чёрном, копьевидные симметричные перья, лучи «солнечного взрыва», ступенчатая рамка",
        ((212, 175, 55), (26, 26, 30), (20, 95, 100), (240, 230, 205), (125, 30, 48)), 0.6,
        (18, 22, 28), (212, 175, 55), 1.3, ("lance",), -0.1, 1.1, 0.8, True, True, "solid", False, "sunburst", ""),
    "bauhaus": StyleProfile(
        "bauhaus", "Баухаус",
        "Чистые красный, жёлтый и синий, круги и треугольники, толстый чёрный контур, никакой декоративности",
        ((220, 50, 40), (245, 200, 40), (30, 70, 160), (30, 30, 30), (245, 242, 235)), 0.9,
        (242, 238, 228), (20, 20, 20), 1.9, ("ellipse", "triangle"), 0.0, 0.6, 0.5, False, False, "solid", False, "bauhaus_shapes", ""),
    "gothic": StyleProfile(
        "gothic", "Готика (витраж)",
        "Перья — стрельчатые арки, свинцовые переплёты, драгоценные цвета витража, стрельчатое окно за птицей",
        ((160, 25, 45), (30, 60, 150), (25, 115, 75), (225, 155, 35), (95, 45, 125)), 0.65,
        (16, 14, 24), (12, 10, 14), 3.0, ("arch",), -0.2, 1.0, 0.8, True, True, "gradient", False, "lancet_window", "", 1.2),
    "constructivism": StyleProfile(
        "constructivism", "Конструктивизм",
        "Красный, чёрный и кремовый, острые треугольные перья, диагональная композиция, типографская зернистость",
        ((200, 30, 30), (22, 22, 22), (235, 225, 200), (120, 120, 120)), 0.85,
        (232, 222, 196), (20, 20, 20), 2.0, ("triangle",), -0.35, 0.8, 0.7, False, False, "solid", False, "constructivist", "grain"),
    "ukiyo_e": StyleProfile(
        "ukiyo_e", "Японская гравюра (укиё-э)",
        "Плоские приглушённые цвета, тонкий контур, перья-капли, красное солнце и стилизованные волны, фактура бумаги",
        ((40, 60, 100), (30, 85, 125), (215, 80, 50), (225, 200, 160), (70, 100, 70)), 0.6,
        (236, 224, 196), (50, 45, 60), 0.9, ("drop",), 0.15, 0.9, 0.7, False, True, "solid", False, "rising_sun_waves", "paper"),
    "byzantine": StyleProfile(
        "byzantine", "Византийская мозаика",
        "Золотой фон, нимб вокруг головы, лазурь и пурпур, всё изображение выложено смальтой",
        ((212, 175, 55), (30, 60, 150), (120, 30, 60), (20, 110, 80), (240, 235, 220)), 0.55,
        (185, 145, 55), (60, 40, 20), 1.2, ("ellipse",), 0.1, 0.9, 0.6, False, True, "gradient", False, "halo", "mosaic"),
    "minimalism": StyleProfile(
        "minimalism", "Минимализм",
        "Мало деталей, приглушённые цвета, тонкая линия, много пустого пространства",
        (), 0.0,
        (245, 244, 240), None, 0.5, ("leaf",), 0.1, 0.45, 0.35, False, False, "solid", False, "", "", 0.55),
    "baroque": StyleProfile(
        "baroque", "Барокко",
        "Пышное оперение, золотые завитки-волюты, глубокие насыщенные цвета, драматическая виньетка",
        ((125, 20, 32), (200, 160, 70), (30, 62, 42), (240, 228, 200), (92, 56, 30)), 0.5,
        (30, 20, 18), (200, 160, 70), 1.3, ("drop",), 0.2, 1.4, 1.0, True, True, "gradient", True, "scrolls", "vignette"),
    "brutalism": StyleProfile(
        "brutalism", "Брутализм",
        "Бетонные серые тона, прямоугольные блочные перья, тяжёлые контуры, фактура необработанного бетона",
        ((150, 150, 145), (105, 105, 102), (192, 190, 182), (150, 82, 52)), 0.85,
        (170, 168, 160), (45, 45, 45), 2.6, ("block",), -0.4, 0.7, 0.5, False, False, "solid", False, "concrete_blocks", "concrete", 0.3),
}
STYLE_NAMES: tuple[str, ...] = tuple(STYLES)
STYLE_ALIASES: dict[str, str] = {
    **{name: name for name in STYLES},
    "модерн": "art_nouveau", "ар-нуво": "art_nouveau", "ардеко": "art_deco", "ар-деко": "art_deco",
    "баухаус": "bauhaus", "готика": "gothic", "витраж": "gothic", "конструктивизм": "constructivism",
    "укиё-э": "ukiyo_e", "укиёэ": "ukiyo_e", "японский": "ukiyo_e", "мозаика": "byzantine", "византия": "byzantine",
    "минимализм": "minimalism", "барокко": "baroque", "брутализм": "brutalism",
}


def choose_style(requested: str | None, seed: int) -> tuple[str, str]:
    """Выбрать стиль: имя/синоним → он; 'seed' → по seed (воспроизводимо); 'random' → случайно.

    Случайный выбор использует отдельный системный генератор и не влияет
    на остальную (seed-зависимую) случайность птицы.
    """
    key = (requested or "none").strip().lower()
    if key in ("none", "art", "нет"):
        return "", "none"
    if key in STYLE_ALIASES:
        return STYLE_ALIASES[key], "cli"
    if key == "seed":
        return STYLE_NAMES[random.Random(seed ^ 0x57A1E).randrange(len(STYLE_NAMES))], "seed"
    if key == "random":
        return random.SystemRandom().choice(STYLE_NAMES), "random"
    raise InputDataError(f"Неизвестный стиль '{requested}'. Доступны: none, {', '.join(STYLE_NAMES)}, random, seed")


def _hue_distance(color_a: Sequence[int], color_b: Sequence[int]) -> float:
    ha, la, sa = colorsys.rgb_to_hls(*(c / 255 for c in color_a[:3]))
    hb, lb, sb = colorsys.rgb_to_hls(*(c / 255 for c in color_b[:3]))
    if sb < 0.15 or sa < 0.15:  # серые и чёрно-белые сравниваем по светлоте
        return abs(la - lb) * 2 + (0.5 if sb >= 0.15 else 0.0)
    delta = abs(ha - hb) % 1.0
    return min(delta, 1 - delta) * 2 + abs(la - lb) * 0.5


def _stylize_color(color: Sequence[int], style: StyleProfile) -> RGB:
    result: RGB = (int(color[0]), int(color[1]), int(color[2]))
    if style.palette and style.palette_strength > 0:
        nearest = min(style.palette, key=lambda candidate: _hue_distance(result, candidate))
        result = mix(result, nearest, style.palette_strength)
    if style.saturation_scale != 1.0:
        h, l, s = colorsys.rgb_to_hls(*(c / 255 for c in result))
        result = hsl_to_rgb(h * 360, clamp(s * style.saturation_scale), l)
    return result


def _ensure_contrast(color: RGB, background: RGB, minimum: float = 0.22) -> RGB:
    lc, lb = rgb_lightness(color), rgb_lightness(background)
    if abs(lc - lb) >= minimum:
        return color
    h, l, s = colorsys.rgb_to_hls(*(c / 255 for c in color))
    target = lb + minimum if lb < 0.5 else lb - minimum
    return hsl_to_rgb(h * 360, s, clamp(target))


def apply_style_to_spec(spec: BirdSpec, style_name: str, limits: dict[str, Any] | None = None) -> BirdSpec:
    """Применить стиль к BirdSpec: цвета, округлость, число перьев и частиц, свечение."""
    style = STYLES[style_name]
    limits = limits or {}
    background = style.background or spec.background_color
    primary = _ensure_contrast(_stylize_color(spec.primary_color, style), background)
    secondary = _ensure_contrast(_stylize_color(spec.secondary_color, style), background, 0.15)
    accent = _stylize_color(spec.accent_color, style)
    outline = style.outline or mix(primary, (0, 0, 0) if rgb_lightness(background) > 0.5 else (255, 255, 255), 0.45)
    max_feathers = int(limits.get("maximum_feather_count", 84))
    min_feathers = int(limits.get("minimum_feather_count", 16))
    max_particles = int(limits.get("maximum_particle_count", 220))
    return dataclasses.replace(
        spec,
        style=style_name,
        background_color=background,
        primary_color=primary,
        secondary_color=secondary,
        accent_color=accent,
        outline_color=outline,
        body_roundness=_round(clamp(spec.body_roundness + style.roundness_shift)),
        wing_roundness=_round(clamp(spec.wing_roundness + style.roundness_shift)),
        feather_count=int(clamp(round(spec.feather_count * style.feather_scale), min_feathers // 2, max_feathers)),
        particle_count=int(clamp(round(spec.particle_count * style.particle_scale), 0, max_particles)),
        glow_intensity=spec.glow_intensity if style.glow else 0.0,
    )


def _noise(width: int, height: int, seed: int, scale: int = 1) -> np.ndarray:
    """Детерминированный шум (локальный генератор NumPy, без глобального состояния)."""
    generator = np.random.default_rng(seed & 0xFFFFFFFF)
    small = generator.standard_normal((max(1, height // scale), max(1, width // scale))).astype(np.float32)
    if scale == 1:
        return small
    image = Image.fromarray(((np.clip(small, -3, 3) + 3) / 6 * 255).astype(np.uint8), "L")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    return (np.asarray(image, dtype=np.float32) / 255 * 6 - 3)


def draw_style_ornament(draw: ImageDraw.ImageDraw, width: int, height: int, spec: BirdSpec, rng: random.Random) -> None:
    """Фоновый орнамент стиля (рисуется до птицы, не конкурирует с ней)."""
    style = STYLES.get(spec.style)
    if style is None or not style.ornament:
        return
    unit = min(width, height)
    line = max(1, int(unit * 0.004))
    outline = spec.outline_color
    ornament = style.ornament
    if ornament == "whiplash":
        inset = unit * 0.04
        draw.rounded_rectangle((inset, inset, width - inset, height - inset), radius=unit * 0.18,
                               outline=rgba(outline, 0.35), width=line)
        for k in range(4):
            x0 = width * (0.05 + 0.3 * k)
            curve = cubic_bezier((x0, height * 0.98), (x0 + width * rng.uniform(-0.3, 0.3), height * 0.6),
                                 (x0 + width * rng.uniform(-0.3, 0.3), height * 0.3), (x0 + width * rng.uniform(-0.1, 0.2), height * 0.02), 60)
            for index in range(len(curve) - 1):
                thickness = max(1, int(line * (2.2 - 1.8 * index / len(curve))))
                draw.line((curve[index], curve[index + 1]), fill=rgba(mix(outline, spec.secondary_color, 0.4), 0.25), width=thickness)
    elif ornament == "sunburst":
        center = (width / 2, height * 1.05)
        rays = 28
        for index in range(rays):
            a0 = math.pi + math.pi * index / rays
            a1 = math.pi + math.pi * (index + 0.5) / rays
            radius = unit * 1.5
            points = [center, (center[0] + math.cos(a0) * radius, center[1] + math.sin(a0) * radius),
                      (center[0] + math.cos(a1) * radius, center[1] + math.sin(a1) * radius)]
            draw.polygon(points, fill=rgba(outline, 0.07 if index % 2 else 0.12))
        for k, inset in enumerate((0.03, 0.045, 0.06)):
            pad = unit * inset
            draw.rectangle((pad, pad, width - pad, height - pad), outline=rgba(outline, 0.5 - 0.12 * k), width=line)
    elif ornament == "bauhaus_shapes":
        r = unit * 0.16
        draw.ellipse((width * 0.1, height * 0.08, width * 0.1 + 2 * r, height * 0.08 + 2 * r), fill=rgba((245, 200, 40), 0.85))
        draw.rectangle((width * 0.7, height * 0.66, width * 0.94, height * 0.9), fill=rgba((30, 70, 160), 0.8))
        draw.polygon([(width * 0.06, height * 0.94), (width * 0.3, height * 0.94), (width * 0.18, height * 0.72)],
                     fill=rgba((220, 50, 40), 0.8))
        draw.rectangle((width * 0.62, height * 0.12, width * 0.95, height * 0.12 + unit * 0.025), fill=rgba((25, 25, 25), 0.9))
    elif ornament == "lancet_window":
        left, right = width * 0.14, width * 0.86
        top, bottom = height * 0.05, height * 0.96
        mid = (left + right) / 2
        spring = top + (right - left) * 0.8
        arch = cubic_bezier((left, spring), (left, top + (spring - top) * 0.35), (mid - (right - left) * 0.2, top), (mid, top), 30)
        arch += cubic_bezier((mid, top), (mid + (right - left) * 0.2, top), (right, top + (spring - top) * 0.35), (right, spring), 30)[1:]
        window = [(left, bottom)] + arch + [(right, bottom)]
        draw.polygon(window, fill=rgba(mix(spec.secondary_color, spec.background_color, 0.75), 0.35))
        lead = rgba((5, 5, 8), 0.9)
        draw.line(window + [window[0]], fill=lead, width=line * 3, joint="curve")
        draw.line((mid, top, mid, bottom), fill=lead, width=line * 2)
        for k in range(1, 5):
            y = spring + (bottom - spring) * k / 5
            draw.line((left, y, right, y), fill=lead, width=line * 2)
        rose = (right - left) * 0.16
        draw.ellipse((mid - rose, spring - rose * 1.4, mid + rose, spring + rose * 0.6), outline=lead, width=line * 2)
    elif ornament == "constructivist":
        draw.polygon([(0, height * 0.82), (width, height * 0.18), (width, height * 0.36), (0, height * 1.0)],
                     fill=rgba((200, 30, 30), 0.75))
        r = unit * 0.13
        draw.ellipse((width * 0.72 - r, height * 0.7 - r, width * 0.72 + r, height * 0.7 + r), fill=rgba((22, 22, 22), 0.85))
        for k in range(3):
            offset = unit * 0.03 * k
            draw.line((width * 0.05, height * 0.12 + offset, width * 0.45, height * 0.02 + offset), fill=rgba((22, 22, 22), 0.8), width=line * 2)
    elif ornament == "rising_sun_waves":
        r = unit * 0.17
        cx, cy = width * 0.78, height * 0.2
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=rgba((215, 80, 50), 0.85))
        wave_color = rgba((40, 60, 100), 0.55)
        step = unit * 0.06
        for row in range(4):
            y = height * 0.82 + row * step * 0.55
            shift = (row % 2) * step / 2
            x = -step + shift
            while x < width + step:
                draw.arc((x, y, x + step, y + step), 180, 360, fill=wave_color, width=line)
                draw.arc((x + step * 0.2, y + step * 0.2, x + step * 0.8, y + step * 0.8), 180, 360, fill=wave_color, width=max(1, line // 2))
                x += step
    elif ornament == "scrolls":
        gold = rgba((200, 160, 70), 0.55)
        for cx, cy, direction in ((0.12, 0.14, 1), (0.88, 0.14, -1), (0.12, 0.86, -1), (0.88, 0.86, 1)):
            points = []
            for index in range(120):
                t = index / 119 * 3.2 * math.pi
                radius = unit * 0.012 * math.exp(0.18 * t)
                points.append((width * cx + direction * radius * math.cos(t), height * cy + radius * math.sin(t)))
            draw.line(points, fill=gold, width=line, joint="curve")
        pad = unit * 0.035
        draw.rounded_rectangle((pad, pad, width - pad, height - pad), radius=unit * 0.05, outline=gold, width=line)
    elif ornament == "concrete_blocks":
        tones = [(150, 150, 145), (130, 130, 126), (185, 183, 176), (115, 115, 112)]
        for index in range(6):
            x0 = width * rng.uniform(-0.1, 0.8)
            y0 = height * rng.uniform(-0.1, 0.8)
            w, h = width * rng.uniform(0.2, 0.5), height * rng.uniform(0.15, 0.45)
            draw.rectangle((x0, y0, x0 + w, y0 + h), fill=rgba(tones[index % len(tones)], 0.55))


def apply_style_post(image: Image.Image, spec: BirdSpec) -> Image.Image:
    """Финальная фактура стиля: мозаика, бумага, зерно, бетон или виньетка."""
    style = STYLES.get(spec.style)
    if style is None or not style.post:
        return image
    width, height = image.size
    array = np.asarray(image, dtype=np.float32)
    if style.post == "mosaic":
        tile = max(4, min(width, height) // 110)
        cols, rows = math.ceil(width / tile), math.ceil(height / tile)
        small = image.resize((cols, rows), Image.Resampling.BOX)
        jitter = _noise(cols, rows, spec.seed) * 10
        small_arr = np.clip(np.asarray(small, dtype=np.float32) + jitter[..., None], 0, 255)
        tiles = Image.fromarray(small_arr.astype(np.uint8), "RGB").resize((cols * tile, rows * tile), Image.Resampling.NEAREST)
        array = np.asarray(tiles.crop((0, 0, width, height)), dtype=np.float32).copy()
        grout = np.array(mix(spec.background_color, (40, 30, 20), 0.6), dtype=np.float32)
        gap = max(1, tile // 6)
        for offset in range(gap):
            array[offset::tile, :, :] = array[offset::tile, :, :] * 0.35 + grout * 0.65
            array[:, offset::tile, :] = array[:, offset::tile, :] * 0.35 + grout * 0.65
    elif style.post == "paper":
        fine = _noise(width, height, spec.seed, 1) * 2.2
        fibres = _noise(width, height, spec.seed + 1, 3) * 2.0  # короткие волокна
        array = array + (fine + fibres)[..., None]
    elif style.post == "grain":
        array = array + _noise(width, height, spec.seed, 1)[..., None] * 9
    elif style.post == "concrete":
        array = array + _noise(width, height, spec.seed, 1)[..., None] * 7 + _noise(width, height, spec.seed + 7, 40)[..., None] * 5
    elif style.post == "vignette":
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        distance = np.sqrt(((xs - width / 2) / (width / 2)) ** 2 + ((ys - height / 2) / (height / 2)) ** 2)
        array = array * np.clip(1.15 - 0.55 * distance ** 2, 0.25, 1.0)[..., None]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


# ============================================================================
# 17a. Художественная режиссура: архетипы, композиция, материал, палитра
# ============================================================================
#
# Генерация разделена на три слоя:
#   1. Семантика текста (TextAnalysis) — характер птицы.
#   2. Художественная режиссура (этот раздел) — визуальный язык: архетипы,
#      композиция, материал, силуэт, палитровая стратегия, фокус.
#   3. Вариативность — иерархическая случайность из производных seed:
#      макроуровень (архетип, композиция, материал, силуэт), средний уровень
#      (форма крыла, хвост, текстура, палитра), микроуровень (углы перьев,
#      частицы, отклонения контура — в рендере).
# Случайность всегда взвешена анализом, поэтому не разрушает смысл.

ART_SEED_MACRO = 0x4D41_4352
ART_SEED_MESO = 0x4D45_534F
ART_SEED_RARE = 0x5241_5245

FOCAL_POINTS: tuple[str, ...] = ("eye", "luminous_core", "wing_tip", "tail", "break_zone", "chest_accent")
WING_SHAPES: tuple[str, ...] = ("pointed", "rounded", "broad", "sickle", "wave")
TAIL_TYPES: tuple[str, ...] = ("fan", "streamer", "forked", "short", "spiral", "split")
TEXTURE_STYLES: tuple[str, ...] = ("fine_grain", "soft_film", "paper", "digital_noise", "marble")
ARCHETYPE_FOCAL: dict[str, str] = {
    "light": "luminous_core", "crystal": "wing_tip", "typographic": "chest_accent", "flow": "tail",
    "particles": "break_zone", "organic": "chest_accent", "digital": "eye", "memory": "eye",
}
FOCAL_NAMES_RU: dict[str, str] = {
    "eye": "глаз", "luminous_core": "световое ядро", "wing_tip": "кончик крыла", "tail": "хвост",
    "break_zone": "зона распада", "chest_accent": "акцент на груди",
}
SIGNAL_NAMES_RU: dict[str, str] = {
    **EMOTION_NAMES_RU, "arousal": "энергия", "calm": "спокойствие", "dominance": "уверенность",
    "vulnerability": "уязвимость", "intensity": "сила переживания", "acquire": "желание обрести",
    "release": "желание отпустить", "mixed": "смешанное намерение", "base": "базовый вес",
}

_ACTIVE_CONFIGS: "ConfigBundle | None" = None


def active_configs() -> "ConfigBundle":
    """Последние загруженные конфигурации (для рендера и repair_bird_spec)."""
    global _ACTIVE_CONFIGS
    if _ACTIVE_CONFIGS is None:
        _ACTIVE_CONFIGS = load_configs()
    return _ACTIVE_CONFIGS


def art_signals(analysis: TextAnalysis) -> dict[str, float]:
    """Сигналы семантического слоя для художественной режиссуры (все 0..1)."""
    eff = effective_scores(analysis)
    mapping = mapping_signals(analysis)
    signals: dict[str, float] = {emotion: eff[emotion] for emotion in EMOTIONS}
    signals.update({
        "arousal": clamp((eff["arousal"] - 0.5) * 2),
        "calm": clamp((0.5 - eff["arousal"]) * 2),
        "dominance": clamp((eff["dominance"] - 0.5) * 2),
        "vulnerability": clamp((0.5 - eff["dominance"]) * 2),
        "intensity": clamp((eff["intensity"] - 0.5) * 2),
        "valence": eff["valence"],
        "acquire": clamp(mapping["acquire"]),
        "release": clamp(mapping["release"]),
        "mixed": clamp(mapping["mixed"]),
        "base": 1.0,
    })
    for rank, theme_id in enumerate(analysis.themes):
        signals[f"theme_{theme_id}"] = max(0.5, 1.0 - 0.2 * rank)
    return signals


def affinity_score(table: dict[str, Any], signals: dict[str, float]) -> float:
    """Сумма коэффициентов × сигналы."""
    return sum(float(coefficient) * signals.get(name, 0.0) for name, coefficient in table.items())


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    """Взвешенный детерминированный выбор (порядок ключей фиксирован конфигурацией)."""
    items = [(key, max(0.0, float(value))) for key, value in weights.items()]
    total = sum(value for _, value in items)
    if total <= 0:
        return items[0][0]
    threshold = rng.random() * total
    running = 0.0
    for key, value in items:
        running += value
        if running >= threshold:
            return key
    return items[-1][0]


def _jitter(rng: random.Random, weights: dict[str, float], sigma: float, power: float = 2.0) -> dict[str, float]:
    return {key: (max(value, 1e-3) * math.exp(rng.gauss(0.0, sigma))) ** power for key, value in weights.items()}


def archetype_raw_weights(signals: dict[str, float], configs: "ConfigBundle") -> dict[str, float]:
    """Веса архетипов из анализа до seed-вариации."""
    return {
        name: max(0.02, float(spec["base"]) + affinity_score(spec["affinity"], signals))
        for name, spec in configs.archetypes["archetypes"].items()
    }


def mix_archetypes(signals: dict[str, float], configs: "ConfigBundle", rng: random.Random,
                   forced_primary: str | None = None) -> tuple[str, str, str, dict[str, float]]:
    """Выбрать основной, вторичный и (необязательно) акцентный архетип и их доли.

    Доли: основной 55–75%, вторичный 20–35%, акцентный 0–15%, сумма 1.0.
    """
    mixing = configs.archetypes["mixing"]
    raw = archetype_raw_weights(signals, configs)
    jittered = {name: value * math.exp(rng.gauss(0.0, float(mixing["seed_jitter"]))) for name, value in raw.items()}
    primary = forced_primary or weighted_choice(rng, {k: v ** 2 for k, v in jittered.items()})
    rest = {k: v for k, v in jittered.items() if k != primary}
    secondary = weighted_choice(rng, {k: v ** 1.5 for k, v in rest.items()})
    rest.pop(secondary)
    accent = weighted_choice(rng, rest) if rng.random() < float(mixing["accent_probability"]) else ""
    lo_p, hi_p = mixing["primary_range"]
    lo_s, hi_s = mixing["secondary_range"]
    _, hi_a = mixing["accent_range"]
    relative = jittered[primary] / (jittered[primary] + jittered[secondary])
    share_p = lo_p + (hi_p - lo_p) * clamp((relative - 0.35) / 0.5)
    share_a = clamp(0.05 + 0.10 * jittered[accent] / jittered[secondary], 0.05, hi_a) if accent else 0.0
    share_p = clamp(share_p, max(lo_p, 1 - hi_s - share_a), min(hi_p, 1 - lo_s - share_a))
    share_p, share_a = round(share_p, 3), round(share_a, 3)
    weights = {primary: share_p, secondary: round(1.0 - share_p - share_a, 3)}
    if accent:
        weights[accent] = share_a
    return primary, secondary, accent, weights


def choose_rare_feature(signals: dict[str, float], seed: int, configs: "ConfigBundle") -> str:
    """Редкая особенность (~12% генераций), детерминированно из базового seed."""
    table = configs.archetypes["rare_features"]
    rng = random.Random(seed ^ ART_SEED_RARE)
    if rng.random() >= float(table["probability"]):
        return ""
    weights = {name: max(0.01, affinity_score(item["affinity"], signals)) for name, item in table["features"].items()}
    return weighted_choice(rng, weights)


def _palette_colors(strategy: str, base_hue: float, saturation: float, lightness: float, signals: dict[str, float],
                    configs: "ConfigBundle", rng: random.Random, dark: bool) -> dict[str, RGB]:
    """Цвета из палитровой стратегии: фон, базовый, вторичный, акцент, свет/контур."""
    table = configs.palettes
    params = table["strategies"][strategy]
    contrast = table["contrast"]
    hue = (base_hue + rng.uniform(-6, 6)) % 360
    sat = clamp(saturation, 0.28, 0.80)
    if strategy == "dark_field_light_accent":
        dark = True
    lo, hi = contrast["dark_background_lightness"] if dark else contrast["light_background_lightness"]
    bg_light = lo + (hi - lo) * rng.random()
    bg_hue = (hue * 0.5 + 228 * 0.5) if dark and signals.get("sadness", 0) > 0.25 else hue + 8
    background = hsl_to_rgb(bg_hue, 0.22 if dark else 0.16, bg_light)
    min_diff = float(contrast["minimum_lightness_difference"])
    if dark:
        body_light = clamp(max(lightness, bg_light + min_diff + 0.12), 0.45, 0.78)
    else:
        body_light = clamp(min(lightness, bg_light - min_diff - 0.05), 0.28, 0.6)
    shift = 1 if dark else -1
    if strategy == "analogous":
        secondary = hsl_to_rgb(hue + params["secondary_offset"], sat * 0.9, clamp(body_light + 0.08 * shift))
        accent = hsl_to_rgb(hue + params["accent_offset"], params["accent_saturation"], params["accent_lightness"])
    elif strategy == "complementary_accent":
        secondary = hsl_to_rgb(hue + params["secondary_offset"], sat, clamp(body_light - 0.07 * shift))
        accent = hsl_to_rgb(hue + params["accent_offset"], params["accent_saturation"], params["accent_lightness"])
    elif strategy == "monochrome":
        sat = sat * 0.7
        secondary = hsl_to_rgb(hue, sat * 0.85, clamp(body_light + 0.16 * shift))
        accent = hsl_to_rgb(hue, params["accent_saturation"], params["accent_lightness"] if dark else 0.42)
    elif strategy == "dark_field_light_accent":
        accent_hue = float(params.get("accent_hue", 48)) if signals.get("hope", 0) >= signals.get("anger", 0) else hue + 150
        secondary = hsl_to_rgb(hue + params["secondary_offset"], sat * 0.8, clamp(body_light + 0.06))
        accent = hsl_to_rgb(accent_hue, params["accent_saturation"], params["accent_lightness"])
    else:  # warm_cool_contrast
        cool_lo, cool_hi = params["cool_hue_range"]
        warm_lo, warm_hi = params["warm_hue_range"]
        cool = clamp((hue - cool_lo) / (cool_hi - cool_lo)) * (cool_hi - cool_lo) + cool_lo
        warm = warm_lo + (warm_hi - warm_lo) * rng.random()
        warm_first = 0 <= hue < 90 or hue > 320
        hue, second = (warm, cool) if warm_first else (cool, warm)
        secondary = hsl_to_rgb(second, sat * 0.85, clamp(body_light + 0.05 * shift))
        accent = hsl_to_rgb(warm if not warm_first else warm + 10, params["accent_saturation"], params["accent_lightness"])
    primary = hsl_to_rgb(hue, sat, body_light)
    glow = hsl_to_rgb(hue, 0.55, 0.86) if dark else hsl_to_rgb(hue, 0.45, 0.18)
    return {"background": background, "primary": primary, "secondary": secondary, "accent": accent, "outline": glow}


def build_art_candidate(analysis: TextAnalysis, base_spec: BirdSpec, candidate_seed: int, signals: dict[str, float],
                        rare_feature: str, configs: "ConfigBundle", overrides: dict[str, str] | None = None) -> BirdSpec:
    """Один кандидат BirdSpec: макро- и средний уровни выбора из производного seed."""
    overrides = overrides or {}
    macro = random.Random(candidate_seed ^ ART_SEED_MACRO)
    meso = random.Random(candidate_seed ^ ART_SEED_MESO)
    archetypes = configs.archetypes["archetypes"]
    compositions = configs.compositions["compositions"]
    materials = configs.materials["materials"]
    limits = configs.mapping["limits"]

    # --- макроуровень --------------------------------------------------------
    primary, secondary, accent, weights = mix_archetypes(signals, configs, macro, overrides.get("archetype"))
    comp_weights = {}
    for name, comp in compositions.items():
        value = max(0.02, affinity_score(comp["affinity"], signals))
        value += 0.35 * (name in archetypes[primary]["compositions"]) + 0.15 * (name in archetypes[secondary]["compositions"])
        comp_weights[name] = value
    composition = overrides.get("composition") or weighted_choice(macro, _jitter(macro, comp_weights, 0.3))
    comp = compositions[composition]

    material_weights = {m: 1.0 / (index + 1) ** 0.8 for index, m in enumerate(archetypes[primary]["materials"])}
    material = overrides.get("material") or weighted_choice(macro, _jitter(macro, material_weights, 0.35, 1.0))
    incompatible = {tuple(sorted(pair)) for pair in configs.materials["incompatible_pairs"]}
    second_options = [m for m in archetypes[secondary]["materials"] + (archetypes[accent]["materials"] if accent else [])
                      if m != material and tuple(sorted((m, material))) not in incompatible]
    secondary_material = ""
    if second_options and (macro.random() < 0.8 or rare_feature == "asymmetric_materials"):
        secondary_material = second_options[macro.randrange(len(second_options))]

    silhouette_weights = {}
    for name, table in configs.compositions["silhouette_affinity"].items():
        silhouette_weights[name] = max(0.02, affinity_score(table, signals)) + 0.5 * (name in comp["silhouettes"])
    silhouette = weighted_choice(macro, _jitter(macro, silhouette_weights, 0.3))

    # --- средний уровень ------------------------------------------------------
    w = lambda name: weights.get(name, 0.0)  # noqa: E731
    wing_weights = {
        "pointed": 0.2 + 0.4 * signals["arousal"] + 0.3 * signals["anger"] + 0.3 * w("crystal"),
        "rounded": 0.2 + 0.4 * signals["tenderness"] + 0.2 * signals["calm"] + 0.3 * w("organic"),
        "broad": 0.15 + 0.3 * signals["dominance"] + 0.3 * signals.get("theme_freedom", 0) + 0.2 * w("light"),
        "sickle": 0.1 + 0.3 * signals["arousal"] + 0.2 * signals["hope"],
        "wave": 0.05 + 0.6 * signals.get("theme_sea", 0) + 0.4 * signals.get("theme_music", 0) + 0.4 * w("flow"),
    }
    wing_shape = weighted_choice(meso, _jitter(meso, wing_weights, 0.3))
    tail_weights = {
        "fan": 0.3 + 0.2 * signals["joy"],
        "streamer": 0.1 + 0.4 * (silhouette == "elongated") + 0.3 * w("flow"),
        "forked": 0.1 + 0.2 * signals["arousal"] + 0.2 * signals["release"],
        "short": 0.05 + 0.5 * (composition == "compressed_core") + 0.2 * signals["vulnerability"],
        "spiral": 0.03 + 0.7 * (composition == "open_spiral"),
    }
    tail_type = "split" if rare_feature == "split_tail" else weighted_choice(meso, _jitter(meso, tail_weights, 0.3))
    strategy_weights = {name: max(0.02, affinity_score(item["affinity"], signals))
                        for name, item in configs.palettes["strategies"].items()}
    palette_strategy = overrides.get("palette") or weighted_choice(meso, _jitter(meso, strategy_weights, 0.3))
    mat = materials[material]
    texture_style = mat["textures"][meso.randrange(len(mat["textures"]))]
    accent_pool = list(materials[secondary_material]["accent_effects"] if secondary_material else mat["accent_effects"])
    accent_effect = accent_pool[meso.randrange(len(accent_pool))]
    if accent_effect == mat["main_effect"]:
        accent_effect = mat["accent_effects"][0]

    # --- числовые параметры ---------------------------------------------------
    abstraction = sum(weights[name] * float(archetypes[name]["abstraction"]) for name in weights)
    abstraction += 0.08 * signals["release"] + 0.05 * signals["anxiety"] - 0.08 * signals["dominance"]
    abstraction = clamp(abstraction + meso.uniform(-0.06, 0.06), 0.25, 0.70)
    layer_count = int(clamp(round(2 + 3 * (w("light") + w("memory") + 0.5 * w("flow")) + 3 * abstraction), 2, 7))
    ns_lo, ns_hi = comp["negative_space"]
    negative_space = ns_lo + (ns_hi - ns_lo) * clamp(0.5 * signals["release"] + 0.3 * signals["hope"] + 0.3 * meso.random())
    openness = clamp(0.3 + 0.35 * signals["release"] + 0.2 * signals["hope"] - 0.2 * signals["acquire"]
                     + 0.15 * (composition in ("open_spiral", "fragmented_escape", "dual_direction"))
                     - 0.2 * (composition == "compressed_core") + meso.uniform(-0.05, 0.05))
    inner_void = clamp(0.3 * signals["release"] + 0.15 * w("memory") - 0.15 * signals["acquire"]
                       + (0.4 if rare_feature == "hollow_core" else 0.0), 0.0, 0.65)
    tension = clamp(0.15 + 0.35 * signals["arousal"] + 0.3 * signals["anxiety"] + 0.25 * signals["anger"]
                    - 0.2 * signals["calm"] + 0.1 * (primary in ("crystal", "digital")) + meso.uniform(-0.05, 0.05))
    focal_point = ARCHETYPE_FOCAL[primary]
    if composition == "fragmented_escape":
        focal_point = "break_zone"
    elif meso.random() < 0.2:
        focal_point = ARCHETYPE_FOCAL[secondary]
    if rare_feature == "luminous_eye":
        focal_point = "eye"
    focal_strength = clamp(0.5 + 0.3 * signals["intensity"] + meso.uniform(-0.08, 0.08), 0.35, 0.95)
    texture_strength = clamp(0.02 + 0.06 * abstraction + 0.04 * tension + meso.uniform(-0.01, 0.01), 0.02, 0.12)

    # --- композиция задаёт геометрию -----------------------------------------
    lerp = lambda rng_pair, t: rng_pair[0] + (rng_pair[1] - rng_pair[0]) * clamp(t)  # noqa: E731
    tilt_degrees = lerp(comp["tilt_degrees"], base_spec.body_tilt)
    body_elongation = base_spec.body_elongation
    body_roundness = base_spec.body_roundness
    body_scale = base_spec.body_scale
    asymmetry = base_spec.wing_asymmetry
    line_width = base_spec.line_width
    wing_span = clamp(base_spec.wing_span * float(comp["wing_length_scale"]))
    if silhouette == "elongated":
        body_elongation += 0.25
    elif silhouette == "round":
        body_roundness += 0.25
        body_elongation -= 0.15
    elif silhouette == "triangular":
        body_roundness -= 0.3
    elif silhouette == "wide":
        wing_span += 0.15
    elif silhouette == "fragile":
        body_scale -= 0.12
        line_width -= 0.2
    elif silhouette == "symmetric":
        asymmetry *= 0.3
    elif silhouette == "asymmetric":
        asymmetry += 0.2
    max_particles = int(limits["maximum_particle_count"])
    particle_count = base_spec.particle_count + int(max_particles * (0.45 * w("particles") + 0.2 * negative_space * signals["release"]))
    particle_count += 40 * (rare_feature == "dissolving_edge") + 30 * (composition == "fragmented_escape")

    colors = _palette_colors(
        palette_strategy, *_hue_sat_light(base_spec.primary_color), signals, configs, meso,
        dark=rgb_lightness(base_spec.background_color) < 0.5,
    )
    motifs = list(base_spec.decorative_motifs[: int(limits["maximum_motif_count"])])
    return dataclasses.replace(
        base_spec,
        seed=candidate_seed,
        background_color=colors["background"], primary_color=colors["primary"], secondary_color=colors["secondary"],
        accent_color=colors["accent"], outline_color=colors["outline"],
        body_scale=_round(clamp(body_scale)), body_elongation=_round(clamp(body_elongation)),
        body_roundness=_round(clamp(body_roundness)), body_tilt=_round(clamp((tilt_degrees + 10) / 40)),
        wing_span=_round(clamp(wing_span)), wing_sweep=_round(lerp(comp["wing_sweep"], base_spec.wing_sweep)),
        wing_asymmetry=_round(clamp(asymmetry)), head_lift=_round(lerp(comp["head_lift"], base_spec.head_lift)),
        tail_length=_round(clamp(base_spec.tail_length * float(comp["tail_length_scale"]))),
        line_width=_round(clamp(line_width)), particle_count=int(clamp(particle_count, 0, max_particles)),
        glow_intensity=_round(clamp(base_spec.glow_intensity + 0.25 * w("light"))),
        open_space=_round(clamp(max(base_spec.open_space, negative_space))),
        decorative_motifs=motifs,
        primary_archetype=primary, secondary_archetype=secondary, accent_archetype=accent,
        archetype_weights=weights, composition_type=composition, abstraction_level=_round(abstraction),
        material_type=material, secondary_material_type=secondary_material, silhouette_type=silhouette,
        feather_style=str(mat["feather_style"]), edge_style=str(mat["edge_style"]), texture_style=texture_style,
        texture_strength=_round(texture_strength), layer_count=layer_count, negative_space=_round(negative_space),
        inner_void=_round(inner_void), silhouette_openness=_round(openness), visual_tension=_round(tension),
        focal_point=focal_point, focal_strength=_round(focal_strength), rare_feature=rare_feature,
        palette_strategy=palette_strategy, wing_shape=wing_shape, tail_type=tail_type,
        main_effect=str(mat["main_effect"]), accent_effect=accent_effect,
    )


def _hue_sat_light(color: Sequence[int]) -> tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(*(c / 255 for c in color[:3]))
    return h * 360, s, l


# --- валидация и ремонт -----------------------------------------------------

ART_FLOAT_LIMITS: dict[str, tuple[float, float]] = {
    "abstraction_level": (0.25, 0.70),
    "texture_strength": (0.02, 0.12),
}


def validate_bird_spec(spec: BirdSpec) -> list[str]:
    """Проверить читаемость и художественные ограничения. Возвращает список проблем (пусто — всё хорошо)."""
    issues: list[str] = []
    for item in dataclasses.fields(spec):
        value = getattr(spec, item.name)
        if item.type in ("float", float) and not 0.0 <= value <= 1.0:
            issues.append(f"{item.name} вне диапазона 0–1")
    if spec.body_scale < 0.3:
        issues.append("корпус слишком мал")
    if spec.wing_span < 0.35:
        issues.append("крылья незаметны")
    if spec.beak_length < 0.2:
        issues.append("клюв не читается")
    if spec.eye_scale > 0.8:
        issues.append("глаз слишком большой")
    if not 0.25 <= spec.head_scale <= 0.9:
        issues.append("голова непропорциональна корпусу")
    if abs(rgb_lightness(spec.primary_color) - rgb_lightness(spec.background_color)) < 0.25:
        issues.append("недостаточный контраст птицы и фона")
    if spec.wing_asymmetry > 0.6:
        issues.append("асимметрия разрушает силуэт")
    if spec.fragmentation > 0.55:
        issues.append("фрагментация уничтожает форму")
    if spec.feather_count * max(1, spec.layer_count) > 84 * 5:
        issues.append("слишком много деталей")
    if len(spec.decorative_motifs) > 3:
        issues.append("больше трёх активных мотивов")
    if spec.primary_archetype:
        for name, (low, high) in ART_FLOAT_LIMITS.items():
            if not low - 1e-9 <= getattr(spec, name) <= high + 1e-9:
                issues.append(f"{name} вне диапазона {low}–{high}")
        weights = spec.archetype_weights
        if len(weights) > 3 or abs(sum(weights.values()) - 1.0) > 0.011:
            issues.append("веса архетипов некорректны")
        if not 0.55 - 1e-9 <= weights.get(spec.primary_archetype, 0) <= 0.75 + 1e-9:
            issues.append("доля основного архетипа вне 55–75%")
        if not 0.20 - 1e-9 <= weights.get(spec.secondary_archetype, 0) <= 0.35 + 1e-9:
            issues.append("доля вторичного архетипа вне 20–35%")
        if spec.accent_archetype and weights.get(spec.accent_archetype, 0) > 0.15 + 1e-9:
            issues.append("доля акцентного архетипа больше 15%")
        if spec.secondary_material_type == spec.material_type:
            issues.append("два одинаковых материала")
        if spec.material_type and spec.secondary_material_type:
            pairs = {tuple(sorted(p)) for p in active_configs().materials["incompatible_pairs"]}
            if tuple(sorted((spec.material_type, spec.secondary_material_type))) in pairs:
                issues.append("несовместимые материалы")
        if not spec.main_effect or not spec.accent_effect or spec.main_effect == spec.accent_effect:
            issues.append("нужны ровно один основной и один акцентный эффект")
        if spec.focal_point not in FOCAL_POINTS:
            issues.append("нет визуального фокуса")
        if not 2 <= spec.layer_count <= 7:
            issues.append("число слоёв вне 2–7")
    return issues


def repair_bird_spec(spec: BirdSpec, configs: "ConfigBundle | None" = None) -> BirdSpec:
    """Детерминированно исправить опасные сочетания (без случайности)."""
    configs = configs or active_configs()
    changes: dict[str, Any] = {}
    for item in dataclasses.fields(spec):
        value = getattr(spec, item.name)
        if item.type in ("float", float):
            low, high = ART_FLOAT_LIMITS.get(item.name, (0.0, 1.0)) if spec.primary_archetype else (0.0, 1.0)
            if not low <= value <= high:
                changes[item.name] = _round(clamp(value, low, high))
    changes["body_scale"] = max(changes.get("body_scale", spec.body_scale), 0.3)
    changes["wing_span"] = max(changes.get("wing_span", spec.wing_span), 0.35)
    changes["beak_length"] = max(changes.get("beak_length", spec.beak_length), 0.2)
    changes["eye_scale"] = min(changes.get("eye_scale", spec.eye_scale), 0.8)
    changes["head_scale"] = clamp(changes.get("head_scale", spec.head_scale), 0.25, 0.9)
    changes["wing_asymmetry"] = min(changes.get("wing_asymmetry", spec.wing_asymmetry), 0.6)
    changes["fragmentation"] = min(changes.get("fragmentation", spec.fragmentation), 0.55)
    if spec.feather_count * max(1, spec.layer_count) > 84 * 5:
        changes["feather_count"] = (84 * 5) // max(1, spec.layer_count)
    changes["decorative_motifs"] = list(spec.decorative_motifs[:3])
    background = spec.background_color
    if abs(rgb_lightness(spec.primary_color) - rgb_lightness(background)) < 0.25:
        changes["primary_color"] = _ensure_contrast(spec.primary_color, background, 0.3)
    if spec.primary_archetype:
        changes["layer_count"] = int(clamp(spec.layer_count, 2, 7))
        if spec.focal_point not in FOCAL_POINTS:
            changes["focal_point"] = ARCHETYPE_FOCAL.get(spec.primary_archetype, "eye")
        pairs = {tuple(sorted(p)) for p in configs.materials["incompatible_pairs"]}
        if spec.secondary_material_type and (
            spec.secondary_material_type == spec.material_type
            or tuple(sorted((spec.material_type, spec.secondary_material_type))) in pairs
        ):
            changes["secondary_material_type"] = ""
        material = configs.materials["materials"].get(spec.material_type)
        if material and (not spec.accent_effect or spec.accent_effect == spec.main_effect):
            changes["accent_effect"] = next(e for e in material["accent_effects"] if e != spec.main_effect)
    return dataclasses.replace(spec, **changes)


# --- оценка и отбор кандидатов ---------------------------------------------

CRITERIA_NAMES_RU: dict[str, str] = {
    "silhouette_readability": "читаемость силуэта",
    "color_contrast": "цветовой контраст",
    "semantic_alignment": "соответствие смыслу текста",
    "composition_balance": "баланс композиции",
    "visual_novelty": "визуальная новизна",
    "motif_coherence": "согласованность мотивов",
    "detail_balance": "баланс деталей",
}


def _color_distance(a: Sequence[int], b: Sequence[int]) -> float:
    ha, sa, la = _hue_sat_light(a)
    hb, sb, lb = _hue_sat_light(b)
    hue = min(abs(ha - hb), 360 - abs(ha - hb)) / 180
    return hue * min(sa, sb) + abs(la - lb)


def expected_choices(signals: dict[str, float], configs: "ConfigBundle") -> dict[str, str]:
    """Самые «ожидаемые» выборы без seed — точка отсчёта для новизны."""
    raw = archetype_raw_weights(signals, configs)
    primary = max(raw, key=raw.get)
    comps = {n: affinity_score(c["affinity"], signals) for n, c in configs.compositions["compositions"].items()}
    sils = {n: affinity_score(t, signals) for n, t in configs.compositions["silhouette_affinity"].items()}
    pals = {n: affinity_score(p["affinity"], signals) for n, p in configs.palettes["strategies"].items()}
    return {
        "primary_archetype": primary,
        "composition_type": max(comps, key=comps.get),
        "material_type": configs.archetypes["archetypes"][primary]["materials"][0],
        "silhouette_type": max(sils, key=sils.get),
        "palette_strategy": max(pals, key=pals.get),
    }


def score_bird_spec(spec: BirdSpec, signals: dict[str, float], configs: "ConfigBundle",
                    recent: Sequence[dict[str, Any]] = ()) -> dict[str, float]:
    """Художественная оценка кандидата без рендера (0..1 по каждому критерию)."""
    materials = configs.materials["materials"]
    compositions = configs.compositions["compositions"]
    archetypes = configs.archetypes["archetypes"]
    readability = float(materials.get(spec.material_type, {}).get("readability", 0.9))
    silhouette = clamp(1 - 0.9 * max(0.0, spec.abstraction_level - 0.35) - 0.45 * spec.fragmentation
                       - 0.4 * max(0.0, spec.wing_asymmetry - 0.25) - 0.3 * max(0.0, 0.35 - spec.body_scale))
    silhouette *= 0.6 + 0.4 * readability
    contrast = 0.7 * clamp(abs(rgb_lightness(spec.primary_color) - rgb_lightness(spec.background_color)) / 0.45)
    contrast += 0.3 * clamp(_color_distance(spec.accent_color, spec.primary_color) / 0.35)
    raw = archetype_raw_weights(signals, configs)
    top = max(raw.values())
    comp_aff = {n: affinity_score(c["affinity"], signals) for n, c in compositions.items()}
    pal_aff = {n: affinity_score(p["affinity"], signals) for n, p in configs.palettes["strategies"].items()}
    semantic = 0.6 * sum(w * raw[a] / top for a, w in spec.archetype_weights.items())
    semantic += 0.25 * clamp(comp_aff[spec.composition_type] / max(max(comp_aff.values()), 1e-6))
    semantic += 0.15 * clamp(pal_aff[spec.palette_strategy] / max(max(pal_aff.values()), 1e-6))
    comp = compositions[spec.composition_type]
    ns_lo, ns_hi = comp["negative_space"]
    ns_center = (ns_lo + ns_hi) / 2
    balance = 0.5 * (1.0 if spec.silhouette_type in comp["silhouettes"] else 0.4)
    balance += 0.3 * clamp(1 - abs(spec.negative_space - ns_center) / 0.3)
    balance += 0.2 * clamp(1 - abs(spec.visual_tension - 0.45))
    expected = expected_choices(signals, configs)
    unexpected = sum(getattr(spec, key) != value for key, value in expected.items())
    novelty = 0.8 * unexpected / len(expected) + 0.2 * bool(spec.accent_archetype)
    if recent:
        summary = art_summary(spec)
        similarity = max(sum(summary.get(k) == r.get(k) for k in summary) / len(summary) for r in recent)
        novelty = clamp(novelty - 0.6 * similarity)
    motif_pool = {m: 0.0 for m in spec.decorative_motifs}
    for name, weight in spec.archetype_weights.items():
        for motif in archetypes[name]["motifs"]:
            if motif in motif_pool:
                motif_pool[motif] = max(motif_pool[motif], 0.5 + weight)
    motifs = clamp(sum(motif_pool.values()) / len(motif_pool)) if motif_pool else 0.7
    detail = clamp(spec.feather_count / 84 * 0.5 + spec.layer_count / 7 * 0.3 + spec.particle_count / 220 * 0.2)
    target = 0.25 + 0.45 * (1 - spec.abstraction_level)
    detail_score = clamp(1 - abs(detail - target) * 1.5)
    scores = {
        "silhouette_readability": silhouette, "color_contrast": clamp(contrast), "semantic_alignment": clamp(semantic),
        "composition_balance": clamp(balance), "visual_novelty": clamp(novelty), "motif_coherence": motifs,
        "detail_balance": detail_score,
    }
    return {k: _round(v) for k, v in scores.items()}


def art_summary(spec: BirdSpec) -> dict[str, Any]:
    """Краткая сводка художественных решений (без текста) — для сравнения новизны."""
    hue = int(_hue_sat_light(spec.primary_color)[0] // 30)
    return {
        "primary_archetype": spec.primary_archetype, "composition_type": spec.composition_type,
        "material_type": spec.material_type, "silhouette_type": spec.silhouette_type,
        "wing_shape": spec.wing_shape, "palette_strategy": spec.palette_strategy, "hue_sector": hue,
    }


def derive_seed(seed: int, index: int) -> int:
    """Производный seed кандидата (SHA-256, без встроенного hash())."""
    digest = hashlib.sha256(f"{seed}:candidate:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def direct_art(analysis: TextAnalysis, base_spec: BirdSpec, seed: int, configs: "ConfigBundle",
               candidates: int | None = None, overrides: dict[str, str] | None = None,
               recent: Sequence[dict[str, Any]] = ()) -> tuple[BirdSpec, dict[str, Any]]:
    """Слой художественной режиссуры: создать кандидатов, проверить, оценить и выбрать лучшего."""
    settings = configs.mapping.get("art_direction", {})
    count = int(candidates or settings.get("candidates", 5))
    weights = settings.get("score_weights", {
        "silhouette_readability": 0.25, "color_contrast": 0.20, "semantic_alignment": 0.15,
        "composition_balance": 0.15, "visual_novelty": 0.10, "motif_coherence": 0.10, "detail_balance": 0.05,
    })
    signals = art_signals(analysis)
    rare_feature = choose_rare_feature(signals, seed, configs)
    scored: list[tuple[float, int, BirdSpec, dict[str, float], list[str]]] = []
    for index in range(count):
        candidate = build_art_candidate(analysis, base_spec, derive_seed(seed, index), signals, rare_feature, configs, overrides)
        problems = validate_bird_spec(candidate)
        if problems:
            candidate = repair_bird_spec(candidate, configs)
        criteria = score_bird_spec(candidate, signals, configs, recent)
        total = sum(float(weights[k]) * v for k, v in criteria.items())
        scored.append((total, index, candidate, criteria, problems))
    best_total, best_index, best, best_criteria, best_problems = max(scored, key=lambda item: (item[0], -item[1]))
    mean_total = sum(item[0] for item in scored) / len(scored)
    reasons = [f"итоговая оценка {best_total:.3f} при средней {mean_total:.3f} по {count} кандидатам"]
    for key, value in best_criteria.items():
        others = [item[3][key] for item in scored if item[1] != best_index]
        if others and value >= max(others) and value > sum(others) / len(others) + 0.02:
            reasons.append(f"лучшая {CRITERIA_NAMES_RU[key]} среди кандидатов ({value:.2f})")
    if best_problems:
        reasons.append("до отбора исправлено: " + "; ".join(best_problems))
    report = {
        "candidates": count,
        "winner_index": best_index,
        "winner_seed": best.seed,
        "winner_score": _round(best_total),
        "criteria": best_criteria,
        "score_weights": dict(weights),
        "reasons": reasons[:6],
        "candidate_scores": [
            {"index": item[1], "score": _round(item[0]), "primary_archetype": item[2].primary_archetype,
             "composition_type": item[2].composition_type, "material_type": item[2].material_type}
            for item in sorted(scored, key=lambda x: x[1])
        ],
    }
    return best, report


# ============================================================================
# 20. Рендер фона
# ============================================================================


def choose_background_mode(spec: BirdSpec, requested: str) -> str:
    """Выбрать режим фона для auto."""
    if requested in ("solid", "gradient"):
        return requested
    motifs = spec.decorative_motifs
    if spec.glow_intensity > 0.3 or spec.open_space > 0.35 or "dark_field" in motifs or "horizon" in motifs:
        return "gradient"
    return "solid"


def render_background(width: int, height: int, spec: BirdSpec, mode: str) -> Image.Image:
    """Создать фон: однотонный или мягкий радиальный градиент (через NumPy)."""
    background = spec.background_color
    if mode == "solid":
        return Image.new("RGB", (width, height), background)
    small_w = max(64, width // 4)
    small_h = max(64, height // 4)
    ys, xs = np.mgrid[0:small_h, 0:small_w].astype(np.float32)
    cx, cy = small_w / 2, small_h * 0.46
    distance = np.sqrt(((xs - cx) / (small_w / 2)) ** 2 + ((ys - cy) / (small_h / 2)) ** 2)
    t = np.clip(distance / 1.25, 0.0, 1.0) ** 1.4
    dark = rgb_lightness(background) < 0.5
    center = mix(background, spec.primary_color, 0.12 + 0.1 * spec.open_space)
    center = mix(center, (255, 255, 255), 0.06 if dark else 0.35)
    edge = mix(background, (0, 0, 0), 0.35 if dark or "dark_field" in spec.decorative_motifs else 0.06)
    center_arr = np.array(center, dtype=np.float32)
    edge_arr = np.array(edge, dtype=np.float32)
    pixels = center_arr[None, None, :] * (1 - t[..., None]) + edge_arr[None, None, :] * t[..., None]
    small = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")
    return small.resize((width, height), Image.Resampling.BICUBIC)


def draw_background_motifs(draw: ImageDraw.ImageDraw, width: int, height: int, spec: BirdSpec,
                           rng: random.Random, max_elements: int) -> None:
    """Абстрактные фоновые мотивы с низкой непрозрачностью."""
    motifs = spec.decorative_motifs
    unit = min(width, height)
    budget = max(8, int(max_elements * (1.0 - 0.5 * spec.open_space)))
    light = rgb_lightness(spec.background_color) < 0.5
    soft = mix(spec.secondary_color, (255, 255, 255) if light else spec.background_color, 0.3)
    line_w = max(1, int(unit * 0.0035))

    if "dark_field" in motifs or "stars" in motifs:
        count = min(budget, 70) if "stars" in motifs else min(budget, 25)
        star_color = mix(spec.accent_color, (255, 255, 255), 0.6)
        for _ in range(count):
            x, y = rng.uniform(0, width), rng.uniform(0, height * 0.85)
            r = unit * rng.uniform(0.0015, 0.004)
            alpha = rng.uniform(0.25, 0.65)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba(star_color, alpha))
            if rng.random() < 0.18:
                arm = r * 3.2
                draw.line((x - arm, y, x + arm, y), fill=rgba(star_color, alpha * 0.6), width=max(1, line_w // 2))
                draw.line((x, y - arm, x, y + arm), fill=rgba(star_color, alpha * 0.6), width=max(1, line_w // 2))

    if "horizon" in motifs:
        y = height * rng.uniform(0.66, 0.72)
        draw.rectangle((0, y - unit * 0.012, width, y + unit * 0.012), fill=rgba(soft, 0.12))
        draw.line((0, y, width, y), fill=rgba(soft, 0.35), width=line_w)

    if "waves" in motifs:
        for k in range(4):
            base_y = height * (0.76 + 0.05 * k)
            amplitude = unit * (0.008 + 0.004 * k)
            frequency = rng.uniform(2.5, 4.5) * math.pi * 2 / width
            phase = rng.uniform(0, math.pi * 2)
            points = [(x, base_y + amplitude * math.sin(frequency * x + phase)) for x in np.linspace(0, width, 90)]
            draw.line(points, fill=rgba(soft, 0.32 - 0.05 * k), width=line_w, joint="curve")

    if "clouds" in motifs:
        cloud_color = mix(spec.background_color, (255, 255, 255), 0.5)
        for _ in range(3):
            cx, cy = rng.uniform(0.1, 0.9) * width, rng.uniform(0.08, 0.35) * height
            for _ in range(5):
                rx, ry = unit * rng.uniform(0.04, 0.08), unit * rng.uniform(0.02, 0.04)
                ox, oy = rng.uniform(-rx, rx), rng.uniform(-ry, ry) * 0.6
                draw.ellipse((cx + ox - rx, cy + oy - ry, cx + ox + rx, cy + oy + ry), fill=rgba(cloud_color, 0.12))

    if "path" in motifs:
        start = (width * 0.04, height * 0.95)
        end = (width * 0.4, height * 0.72)
        curve = cubic_bezier(start, (width * 0.2, height * 0.8), (width * 0.25, height * 0.9), end, 40)
        for index, (x, y) in enumerate(curve):
            if index % 2 == 0:
                r = unit * 0.003 * (1 + index / 40)
                draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba(soft, 0.4))

    if "vertical_shapes" in motifs:
        count = rng.randint(5, 8)
        for index in range(count):
            x = width * (0.08 + 0.84 * (index + rng.uniform(0.2, 0.8)) / count)
            bar_w = unit * rng.uniform(0.008, 0.02)
            top = height * rng.uniform(0.45, 0.8)
            draw.rounded_rectangle((x - bar_w, top, x + bar_w, height * 0.98), radius=bar_w, fill=rgba(soft, 0.14))

    if "technology" in motifs:
        step = unit * 0.035
        x0, y0 = width * 0.06, height * 0.08
        for i in range(6):
            for j in range(4):
                x, y = x0 + i * step, y0 + j * step
                r = unit * 0.0022
                draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba(soft, 0.45))
                if rng.random() < 0.3:
                    draw.line((x, y, x + step, y), fill=rgba(soft, 0.3), width=max(1, line_w // 2))
                if rng.random() < 0.2:
                    draw.line((x, y, x, y + step), fill=rgba(soft, 0.3), width=max(1, line_w // 2))

    if "music" in motifs:
        base_y = height * rng.uniform(0.18, 0.28)
        frequency = rng.uniform(1.5, 2.5) * math.pi * 2 / width
        for k in range(5):
            y = base_y + k * unit * 0.012
            points = [(x, y + unit * 0.03 * math.sin(frequency * x + 0.3)) for x in np.linspace(0, width, 90)]
            draw.line(points, fill=rgba(soft, 0.22), width=max(1, line_w // 2), joint="curve")

    if "organic_lines" in motifs:
        for _ in range(3):
            p0 = (rng.uniform(0, 0.2) * width, rng.uniform(0.2, 0.9) * height)
            p3 = (rng.uniform(0.8, 1.0) * width, rng.uniform(0.2, 0.9) * height)
            p1 = (rng.uniform(0.2, 0.5) * width, rng.uniform(0, 1) * height)
            p2 = (rng.uniform(0.5, 0.8) * width, rng.uniform(0, 1) * height)
            draw.line(cubic_bezier(p0, p1, p2, p3, 60), fill=rgba(soft, 0.18), width=line_w, joint="curve")


# ============================================================================
# 21. Рендер птицы
# ============================================================================


@dataclass
class BirdGeometry:
    """Фигуры птицы в локальных координатах и опорные точки."""

    behind: list[Shape]
    body: list[Shape]
    front: list[Shape]
    particles: list[Shape]
    glow: list[tuple[Point, float, float, tuple[int, int, int], float]]
    anchors: dict[str, Point]
    # Внутренние линии для ключевых слов: имя → (точки пути, цвет области, толщина области)
    text_paths: dict[str, tuple[list[Point], tuple[int, int, int], float]] = field(default_factory=dict)


def _body_contour(a: float, b: float, roundness: float, wobble: float, rng: random.Random) -> list[Point]:
    exponent = 1.3 + 0.9 * roundness
    phase = rng.uniform(0, math.pi * 2)
    points = []
    for index in range(72):
        t = 2 * math.pi * index / 72
        c, s = math.cos(t), math.sin(t)
        x = a * math.copysign(abs(c) ** (2 / exponent), c)
        y = b * math.copysign(abs(s) ** (2 / exponent), s)
        if x < 0:
            y *= 1 - 0.38 * (-x / a)
        else:
            y *= 1 + 0.08 * (x / a)
        factor = 1 + wobble * math.sin(7 * t + phase)
        points.append((x * factor, y * factor))
    return points


def _wing(spec: BirdSpec, root: Point, angle: float, length: float, roundness: float,
          fill: tuple[int, int, int, int], feather_fill: Callable[[float], tuple[int, int, int, int]],
          outline: tuple[int, int, int, int], line: float, feathers: int, rng: random.Random,
          midline_out: list[tuple[Point, float]] | None = None) -> tuple[list[Shape], list[tuple[Point, float, float]]]:
    """Крыло: контур из двух кривых Безье и отдельные перья вдоль задней кромки.

    Если передан ``midline_out``, в него записывается внутренняя линия крыла
    (точка, локальная толщина) — по ней потом пишется ключевое слово.
    """
    axis = (math.cos(angle), math.sin(angle))
    front_normal = (axis[1], -axis[0])
    tip = (root[0] + axis[0] * length, root[1] + axis[1] * length)
    root_front = (root[0] + 0.09, root[1] + 0.01)
    root_back = (root[0] - 0.13, root[1] - 0.02)
    bulge = length * (0.1 + 0.22 * roundness)
    trailing_bulge = length * (0.06 + 0.12 * roundness)
    tip_pull = 0.25 + 0.35 * roundness
    leading = cubic_bezier(
        root_front,
        (root_front[0] + axis[0] * length * 0.35 + front_normal[0] * bulge, root_front[1] + axis[1] * length * 0.35 + front_normal[1] * bulge),
        (tip[0] - axis[0] * length * tip_pull * 0.4 + front_normal[0] * bulge * 0.6, tip[1] - axis[1] * length * tip_pull * 0.4 + front_normal[1] * bulge * 0.6),
        tip, 28,
    )
    trailing = cubic_bezier(
        tip,
        (tip[0] - axis[0] * length * 0.3 - front_normal[0] * trailing_bulge, tip[1] - axis[1] * length * 0.3 - front_normal[1] * trailing_bulge),
        (root_back[0] + axis[0] * length * 0.3 - front_normal[0] * trailing_bulge, root_back[1] + axis[1] * length * 0.3 - front_normal[1] * trailing_bulge),
        root_back, 28,
    )
    shapes = [Shape(points=leading + trailing[1:], fill=fill, outline=outline, width=line)]
    if midline_out is not None:
        trailing_reversed = trailing[::-1]
        for index in range(3, len(leading) - 5):
            lead, trail = leading[index], trailing_reversed[index]
            point = (lead[0] * 0.58 + trail[0] * 0.42, lead[1] * 0.58 + trail[1] * 0.42)
            midline_out.append((point, math.hypot(lead[0] - trail[0], lead[1] - trail[1])))
    feather_anchors: list[tuple[Point, float, float]] = []
    count = max(3, feathers)
    shapes_feathers: list[Shape] = []
    for index in range(count):
        t = 0.08 + 0.9 * index / max(1, count - 1)
        point = trailing[min(len(trailing) - 1, int((1 - t) * (len(trailing) - 1)))]
        direction = _normalize((
            -front_normal[0] * (1.0 - 0.55 * t) + axis[0] * (0.25 + 0.75 * t),
            -front_normal[1] * (1.0 - 0.55 * t) + axis[1] * (0.25 + 0.75 * t),
        ))
        jitter = (spec.fragmentation * 0.45 + spec.wing_asymmetry * 0.2) * rng.uniform(-1, 1)
        feather_angle = math.atan2(direction[1], direction[0]) + jitter
        feather_length = length * (0.16 + 0.26 * t) * (0.85 + 0.3 * rng.random()) * (0.8 + 0.4 * spec.feather_density)
        feather_width = feather_length * (0.22 + 0.2 * roundness)
        gap = 0.0
        if t > 0.55:
            gap = spec.open_space * feather_length * 0.35 * rng.random()
        base = (point[0] - direction[0] * feather_length * 0.25 + direction[0] * gap,
                point[1] - direction[1] * feather_length * 0.25 + direction[1] * gap)
        kind = feather_kind(roundness, rng, style_kinds(spec))
        color = feather_fill(t)
        if rng.random() < spec.fragmentation * 0.6:
            first = feather_points(base, feather_angle, feather_length * 0.55, feather_width, kind)
            second_base = (base[0] + math.cos(feather_angle) * feather_length * 0.68,
                           base[1] + math.sin(feather_angle) * feather_length * 0.68)
            second = feather_points(second_base, feather_angle + rng.uniform(-0.25, 0.25), feather_length * 0.34, feather_width * 0.8, kind)
            shapes_feathers.append(Shape(first, fill=color, outline=outline, width=line * 0.6))
            shapes_feathers.append(Shape(second, fill=color, outline=outline, width=line * 0.6))
        else:
            shapes_feathers.append(Shape(feather_points(base, feather_angle, feather_length, feather_width, kind),
                                         fill=color, outline=outline, width=line * 0.6))
        tip_point = (base[0] + math.cos(feather_angle) * feather_length, base[1] + math.sin(feather_angle) * feather_length)
        feather_anchors.append((tip_point, feather_angle, t))
    return shapes_feathers + shapes, feather_anchors


def build_bird_geometry(spec: BirdSpec, rng: random.Random) -> BirdGeometry:
    """Построить все элементы птицы в локальных координатах (без растеризации)."""
    primary, secondary, accent, outline_rgb = spec.primary_color, spec.secondary_color, spec.accent_color, spec.outline_color
    dark_bg = rgb_lightness(spec.background_color) < 0.5
    periphery = mix(primary, (255, 255, 255), 0.55)
    opacity = 0.55 + 0.45 * spec.body_opacity
    style = STYLES.get(spec.style)
    line = (0.002 + 0.006 * spec.line_width) * (style.line_scale if style else 1.0)
    outline = rgba(outline_rgb, 0.55 + 0.4 * spec.line_width if not style or not style.outline else 0.95)
    behind: list[Shape] = []
    body_shapes: list[Shape] = []
    front: list[Shape] = []
    particles: list[Shape] = []
    glow: list[tuple[Point, float, float, tuple[int, int, int], float]] = []
    anchors: dict[str, Point] = {}
    motifs = spec.decorative_motifs
    release_like = spec.intent_type in ("release", "mixed")
    acquire_like = spec.intent_type in ("acquire", "mixed")

    # --- корпус ----------------------------------------------------------
    a = 0.25 * (0.85 + 0.35 * spec.body_elongation)
    b = a * (0.62 - 0.25 * spec.body_elongation)
    wobble = 0.03 * spec.fragmentation + 0.015 * spec.wing_asymmetry
    body = _body_contour(a, b, spec.body_roundness, wobble, rng)

    # --- голова ----------------------------------------------------------
    head_r = 0.085 * (0.75 + 0.6 * spec.head_scale)
    head_c = (a * 0.8 + head_r * 0.3, b * 0.35 + head_r * (0.25 + 1.05 * spec.head_lift))
    anchors["head"] = head_c

    # --- крылья ----------------------------------------------------------
    wing_length = 0.36 * (0.62 + 0.8 * spec.wing_span)
    near_angle = math.radians(185 - 95 * spec.wing_sweep)
    spread = math.radians(12 + 38 * spec.wing_span)
    asym = spec.wing_asymmetry
    far_angle = near_angle - spread + math.radians(rng.uniform(-1, 1) * 25 * asym)
    far_length = wing_length * (0.92 - 0.25 * asym * rng.random())
    near_count = max(3, int(spec.feather_count * 0.38))
    far_count = max(3, int(spec.feather_count * 0.25))
    tail_count = max(3, 3 + int(round(4 * spec.tail_spread)))

    def near_feather_color(t: float) -> tuple[int, int, int, int]:
        base = mix(secondary, primary, 0.35)
        if release_like:
            base = mix(base, periphery, spec.open_space * t)
        elif acquire_like:
            base = mix(base, accent, 0.25 * (1 - t))
        alpha = opacity * (1 - (0.45 * spec.open_space * t if release_like else 0.1 * t))
        return rgba(base, alpha)

    def far_feather_color(t: float) -> tuple[int, int, int, int]:
        base = mix(secondary, spec.background_color, 0.25)
        if release_like:
            base = mix(base, periphery, spec.open_space * t)
        return rgba(base, opacity * 0.75 * (1 - 0.4 * spec.open_space * t))

    far_root = (0.03, b * 0.72)
    near_root = (-0.04, b * 0.62)
    far_wing, far_anchors = _wing(
        spec, far_root, far_angle, far_length, spec.wing_roundness,
        rgba(mix(secondary, spec.background_color, 0.3), opacity * 0.7), far_feather_color,
        rgba(outline_rgb, 0.35), line * 0.8, far_count, rng,
    )
    near_midline: list[tuple[Point, float]] = []
    near_wing_rgb = mix(secondary, primary, 0.45)
    near_wing, near_anchors = _wing(
        spec, near_root, near_angle, wing_length, spec.wing_roundness,
        rgba(near_wing_rgb, opacity * 0.92), near_feather_color,
        outline, line, near_count, rng, midline_out=near_midline,
    )
    anchors["near_wing_tip"] = (near_root[0] + math.cos(near_angle) * wing_length, near_root[1] + math.sin(near_angle) * wing_length)
    anchors["far_wing_tip"] = (far_root[0] + math.cos(far_angle) * far_length, far_root[1] + math.sin(far_angle) * far_length)
    anchors["near_wing_mid"] = (near_root[0] + math.cos(near_angle) * wing_length * 0.5, near_root[1] + math.sin(near_angle) * wing_length * 0.5)
    anchors["near_wing_angle"] = (near_angle, 0.0)

    if "faded_layers" in motifs:
        for k, offset in enumerate((0.05, 0.1)):
            ghost_angle = near_angle + math.radians(8 * (k + 1))
            ghost, _ = _wing(spec, (near_root[0] - offset, near_root[1] - offset * 0.3), ghost_angle, wing_length * (1 - 0.06 * k),
                             spec.wing_roundness, rgba(mix(primary, spec.background_color, 0.55), 0.18 - 0.05 * k),
                             lambda t: rgba(mix(primary, spec.background_color, 0.6), 0.12), None, 0.0, 4, rng)  # type: ignore[arg-type]
            behind.extend(ghost)

    if "open_wings" in motifs:
        for k in range(5):
            ray_angle = (near_angle + far_angle) / 2 + math.radians(-30 + 15 * k)
            start = (0.0, b * 0.8)
            end = (start[0] + math.cos(ray_angle) * wing_length * 1.25, start[1] + math.sin(ray_angle) * wing_length * 1.25)
            behind.append(Shape([start, end], outline=rgba(mix(accent, (255, 255, 255), 0.4), 0.18), width=line * 0.7, closed=False))

    behind.extend(far_wing)

    # --- хвост -----------------------------------------------------------
    tail_root = (-a * 0.9, -b * 0.05)
    tail_angle = math.radians(168 + 48 * spec.tail_drop)
    tail_len = 0.16 + 0.3 * spec.tail_length
    tail_spread = math.radians(14 + 50 * spec.tail_spread)
    tail_shapes: list[Shape] = []
    tail_tips: list[tuple[Point, float]] = []
    for index in range(tail_count):
        offset = (index / max(1, tail_count - 1) - 0.5) * tail_spread
        angle = tail_angle + offset + rng.uniform(-0.05, 0.05) * (1 + 3 * spec.fragmentation)
        length = tail_len * (0.8 + 0.35 * (1 - abs(offset) / max(tail_spread, 1e-6))) * (0.9 + 0.2 * rng.random())
        width = length * (0.16 + 0.14 * spec.body_roundness)
        kind = style_kinds(spec)[0] if style_kinds(spec) else ("leaf" if spec.body_roundness > 0.45 else "lance")
        base_color = mix(secondary, accent, 0.25 + 0.35 * abs(offset) / max(tail_spread, 1e-6))
        if release_like:
            base_color = mix(base_color, periphery, spec.open_space * 0.6)
        tail_shapes.append(Shape(feather_points(tail_root, angle, length, width, kind), fill=rgba(base_color, opacity * 0.9), outline=outline, width=line * 0.6))
        tail_tips.append(((tail_root[0] + math.cos(angle) * length, tail_root[1] + math.sin(angle) * length), angle))
    anchors["tail_tip"] = tail_tips[len(tail_tips) // 2][0]
    behind.extend(tail_shapes)
    middle_tip, middle_angle = tail_tips[len(tail_tips) // 2]
    middle_length = math.hypot(middle_tip[0] - tail_root[0], middle_tip[1] - tail_root[1])
    tail_path = [
        (tail_root[0] + math.cos(middle_angle) * middle_length * t, tail_root[1] + math.sin(middle_angle) * middle_length * t)
        for t in np.linspace(0.92, 0.12, 16)
    ]
    tail_rgb = mix(secondary, accent, 0.25)
    tail_width = middle_length * (0.16 + 0.14 * spec.body_roundness) * 1.6

    if "fire" in motifs:
        fire_colors = [(232, 72, 38), (246, 142, 46), (252, 196, 80)]
        for index, (tip, angle) in enumerate(tail_tips[: 5]):
            for k in range(2):
                flame_angle = angle + rng.uniform(-0.3, 0.3) + (math.pi / 2 - angle) * 0.25
                length = tail_len * rng.uniform(0.25, 0.4)
                front.append(Shape(feather_points(tip, flame_angle, length, length * 0.35, "drop"),
                                   fill=rgba(fire_colors[(index + k) % 3], 0.55)))

    if style and style.ornament == "halo":  # византийский нимб за головой
        gold = (212, 175, 55)
        behind.append(Shape(ellipse_points(head_c, head_r * 2.1, head_r * 2.1, 0, 40), fill=rgba(gold, 0.9),
                            outline=rgba(mix(gold, (60, 40, 20), 0.6), 0.9), width=line * 1.5))
        behind.append(Shape(ellipse_points(head_c, head_r * 1.75, head_r * 1.75, 0, 40), outline=rgba((250, 225, 140), 0.8),
                            width=line, closed=True))

    # --- ядро и свечение -------------------------------------------------
    glow_color = mix(accent, (255, 214, 150), 0.5) if "warm_glow" in motifs else accent
    glow.append(((0.0, 0.02), a * 1.7, b * 2.2, glow_color, spec.glow_intensity))
    if "warm_glow" in motifs or "love" in motifs:
        glow.append((head_c, head_r * 2.4, head_r * 2.4, (255, 190, 150), spec.glow_intensity * 0.7))

    # --- тело ------------------------------------------------------------
    if release_like:
        inner = mix(primary, outline_rgb if not dark_bg else spec.background_color, 0.15)
        body_shapes.append(Shape(body, fill=rgba(mix(primary, periphery, 0.35 * spec.open_space), opacity), outline=outline, width=line))
        core = [(x * 0.72, y * 0.72) for x, y in body]
        body_shapes.append(Shape(core, fill=rgba(inner, opacity * 0.9)))
    else:
        body_shapes.append(Shape(body, fill=rgba(primary, opacity), outline=outline, width=line))

    if "echo_lines" in motifs:
        for k, scale in enumerate((1.12, 1.25, 1.38)):
            echo = [(x * scale - 0.02 * (k + 1), y * scale) for x, y in body]
            behind.insert(0, Shape(echo + [echo[0]], outline=rgba(mix(primary, spec.background_color, 0.3), 0.3 - 0.07 * k), width=line * 0.8, closed=False))

    # мелкие перья-чешуйки на груди
    scale_count = max(0, int(spec.feather_count * 0.2 * (0.5 + spec.feather_density)))
    if style and not style.chest_feathers:
        scale_count = 0
    for index in range(scale_count):
        u = rng.uniform(-0.55, 0.75)
        v = rng.uniform(-0.55, 0.35)
        x, y = a * u, b * v * (1 - 0.3 * max(0.0, -u))
        size = a * rng.uniform(0.1, 0.18) * (0.8 + 0.4 * spec.feather_density)
        kind = feather_kind(spec.body_roundness, rng, style_kinds(spec))
        color = mix(primary, secondary, rng.uniform(0.2, 0.6))
        body_shapes.append(Shape(feather_points((x, y), math.radians(200 + rng.uniform(-20, 20)), size, size * 0.55, kind),
                                 fill=rgba(color, 0.45 * opacity)))

    if "warm_core" in motifs or acquire_like:
        core_strength = 0.35 + 0.35 * (1 if "warm_core" in motifs else 0.5)
        warm = mix(accent, (255, 196, 120), 0.45)
        for k, scale in enumerate((0.55, 0.38, 0.22)):
            body_shapes.append(Shape(ellipse_points((a * 0.2, -b * 0.05), a * scale, b * scale * 1.1),
                                     fill=rgba(mix(warm, (255, 255, 255), 0.2 * k), core_strength * (0.35 + 0.2 * k))))

    if acquire_like:
        # линии, направленные к ядру
        strokes = 6 + int(10 * spec.feather_density)
        for index in range(strokes):
            angle = 2 * math.pi * index / strokes + rng.uniform(-0.1, 0.1)
            outer = (a * 0.2 + math.cos(angle) * a * 1.55, -b * 0.05 + math.sin(angle) * a * 1.25)
            inner_pt = (a * 0.2 + math.cos(angle) * a * 1.2, -b * 0.05 + math.sin(angle) * a * 0.95)
            behind.insert(0, Shape([outer, inner_pt], outline=rgba(mix(accent, primary, 0.3), 0.22), width=line * 0.8, closed=False))

    # --- шея, голова, клюв, глаз -----------------------------------------
    neck = [(a * 0.3, b * 0.75), (head_c[0] - head_r * 0.35, head_c[1] + head_r * 0.9),
            (head_c[0] + head_r * 0.6, head_c[1] - head_r * 0.75), (a * 0.95, b * 0.05)]
    head_color = mix(primary, secondary, 0.15)
    front.append(Shape(neck, fill=rgba(head_color, opacity)))
    front.append(Shape(ellipse_points(head_c, head_r, head_r, 0, 32), fill=rgba(head_color, opacity), outline=outline, width=line))

    beak_base = (head_c[0] + head_r * 0.9, head_c[1] - head_r * 0.12)
    beak_len = head_r * (0.6 + 1.6 * spec.beak_length)
    beak_half = head_r * (0.5 - 0.3 * spec.beak_sharpness)
    droop = beak_len * (0.08 + 0.2 * (1 - spec.head_lift))
    tip = (beak_base[0] + beak_len, beak_base[1] - droop)
    if spec.beak_sharpness > 0.5:
        beak = [(beak_base[0] - head_r * 0.1, beak_base[1] + beak_half), tip, (beak_base[0] - head_r * 0.1, beak_base[1] - beak_half)]
    else:
        upper = cubic_bezier((beak_base[0] - head_r * 0.1, beak_base[1] + beak_half), (beak_base[0] + beak_len * 0.5, beak_base[1] + beak_half), (tip[0] - beak_len * 0.1, tip[1] + beak_half * 0.3), tip, 10)
        lower = cubic_bezier(tip, (tip[0] - beak_len * 0.1, tip[1] - beak_half * 0.3), (beak_base[0] + beak_len * 0.5, beak_base[1] - beak_half), (beak_base[0] - head_r * 0.1, beak_base[1] - beak_half), 10)
        beak = upper + lower[1:]
    beak_color = mix(accent, outline_rgb, 0.25 + 0.3 * spec.beak_sharpness)
    front.append(Shape(beak, fill=rgba(beak_color, 0.95), outline=outline, width=line * 0.8))
    anchors["beak_tip"] = tip

    eye_c = (head_c[0] + head_r * 0.3, head_c[1] + head_r * 0.18)
    eye_r = head_r * (0.14 + 0.3 * spec.eye_scale)
    eye_white = mix(spec.background_color, (255, 255, 255), 0.7)
    front.append(Shape(ellipse_points(eye_c, eye_r, eye_r, 0, 24), fill=rgba(eye_white, 0.95), outline=outline, width=line * 0.6))
    pupil_r = eye_r * 0.55
    front.append(Shape(ellipse_points((eye_c[0] + eye_r * 0.15, eye_c[1]), pupil_r, pupil_r, 0, 20), fill=rgba(mix(outline_rgb, (0, 0, 0), 0.4), 1.0)))
    highlight_r = pupil_r * 0.3
    front.append(Shape(ellipse_points((eye_c[0] + eye_r * 0.3, eye_c[1] + eye_r * 0.25), highlight_r, highlight_r, 0, 12), fill=(255, 255, 255, 220)))

    front.extend(near_wing)

    # --- декоративные мотивы, встроенные в птицу --------------------------
    if "paired_shapes" in motifs or "love" in motifs:
        pair_color = mix(accent, (255, 170, 190) if "love" in motifs else secondary, 0.4)
        center = (a * 0.35, -b * 1.25)
        for k in (-1, 1):
            front.append(Shape(ellipse_points((center[0] + k * head_r * 0.45, center[1]), head_r * 0.6, head_r * 0.45, k * 0.3, 24),
                               fill=rgba(pair_color, 0.3), outline=rgba(pair_color, 0.6), width=line * 0.5))
    if "love" in motifs:
        arc_color = mix(accent, (255, 150, 170), 0.5)
        for k in (0, 1):
            arc = [(head_c[0] - head_r * 2.2 + head_r * 0.9 * k + math.cos(t) * head_r * 0.9,
                    head_c[1] + head_r * 1.6 + math.sin(t) * head_r * 0.6) for t in np.linspace(0.2, math.pi - 0.2, 16)]
            front.append(Shape(arc, outline=rgba(arc_color, 0.55), width=line * 0.8, closed=False))
    if "home" in motifs:
        nest_color = mix(secondary, (170, 120, 80), 0.5)
        for k in range(5):
            depth = b * (1.25 + 0.07 * k)
            nest = [(x, -depth - 0.08 * (1 - (x / (a * 1.1)) ** 2) * a) for x in np.linspace(-a * 1.1, a * 1.1, 24)]
            behind.append(Shape(nest, outline=rgba(nest_color, 0.45 - 0.05 * k), width=line * 1.4, closed=False))
    if "leaves" in motifs:
        leaf_color = shift_hue(secondary, 0.0) if 70 <= colorsys.rgb_to_hls(*(c / 255 for c in secondary))[0] * 360 <= 160 else hsl_to_rgb(115, 0.4, 0.5)
        for index in range(6):
            base = anchors["tail_tip"] if index < 3 else (tail_root[0] + rng.uniform(-0.05, 0.05), tail_root[1] - rng.uniform(0.05, 0.12))
            angle = rng.uniform(0, 2 * math.pi)
            length = rng.uniform(0.04, 0.07)
            front.append(Shape(feather_points(base, angle, length, length * 0.45, "leaf"), fill=rgba(leaf_color, 0.55)))
    if "color_fragments" in motifs:
        for index in range(12):
            anchor = near_anchors[rng.randrange(len(near_anchors))][0] if near_anchors else (0.0, 0.2)
            cx, cy = anchor[0] + rng.uniform(-0.08, 0.08), anchor[1] + rng.uniform(-0.08, 0.08)
            size = rng.uniform(0.01, 0.025)
            sides = rng.randint(3, 5)
            rot = rng.uniform(0, math.pi)
            poly = [(cx + math.cos(rot + 2 * math.pi * k / sides) * size, cy + math.sin(rot + 2 * math.pi * k / sides) * size) for k in range(sides)]
            fragment_color = shift_hue(accent, rng.choice((-40, 0, 40)), 0.05)
            front.append(Shape(poly, fill=rgba(fragment_color, 0.6)))
    if "sharp_geometry" in motifs:
        for index in range(7):
            anchor = far_anchors[rng.randrange(len(far_anchors))][0] if far_anchors else (0.0, 0.3)
            cx, cy = anchor[0] + rng.uniform(-0.06, 0.06), anchor[1] + rng.uniform(-0.03, 0.08)
            length = rng.uniform(0.03, 0.06)
            angle = rng.uniform(0, 2 * math.pi)
            behind.append(Shape(feather_points((cx, cy), angle, length, length * 0.3, "triangle"),
                                fill=rgba(mix(outline_rgb, accent, 0.5), 0.35)))

    # --- частицы отпускания ----------------------------------------------
    emitters: list[tuple[Point, float]] = [(p, ang) for p, ang, t in near_anchors if t > 0.4]
    emitters += [(p, ang) for p, ang, t in far_anchors if t > 0.5]
    emitters += tail_tips
    if spec.particle_count > 0 and emitters:
        particle_light = mix(accent, (255, 255, 255), 0.35)
        spread_distance = 0.06 + 0.3 * spec.open_space
        for index in range(spec.particle_count):
            origin, angle = emitters[index % len(emitters)]
            angle += rng.uniform(-0.6, 0.6)
            distance = spread_distance * rng.random() ** 0.8
            position = (origin[0] + math.cos(angle) * distance, origin[1] + math.sin(angle) * distance)
            progress = distance / spread_distance
            size = 0.004 + 0.012 * (1 - progress) * rng.uniform(0.5, 1.0)
            color = mix(mix(primary, accent, 0.5), particle_light, progress)
            alpha = 0.85 - 0.6 * progress
            if release_like and rng.random() < 0.35:
                particles.append(Shape(feather_points(position, angle, size * 3, size * 1.2, "leaf"), fill=rgba(color, alpha)))
            elif rng.random() < 0.5:
                particles.append(Shape(ellipse_points(position, size, size, 0, 10), fill=rgba(color, alpha)))
            else:
                particles.append(Shape([(position[0] + size, position[1]), (position[0], position[1] + size),
                                        (position[0] - size, position[1]), (position[0], position[1] - size)],
                                       fill=rgba(color, alpha)))

    body_path = [(x * 0.66, y * 0.66 + b * 0.08) for x, y in body[40:70]]
    text_paths = {
        "wing": ([pt for pt, _ in near_midline], near_wing_rgb,
                 sorted(width for _, width in near_midline)[len(near_midline) // 2] if near_midline else 0.05),
        "body": (body_path, primary, b * 0.7),
        "tail": (tail_path, tail_rgb, tail_width),
    }
    return BirdGeometry(behind=behind, body=body_shapes, front=front, particles=particles, glow=glow,
                        anchors=anchors, text_paths=text_paths)


class _Transform:
    """Поворот, масштаб и перенос локальных координат в пиксели холста."""

    def __init__(self, angle: float, scale: float, offset: Point) -> None:
        self.cos = math.cos(angle)
        self.sin = math.sin(angle)
        self.scale = scale
        self.offset = offset

    def rotate(self, point: Point) -> Point:
        return (point[0] * self.cos - point[1] * self.sin, point[0] * self.sin + point[1] * self.cos)

    def __call__(self, point: Point) -> Point:
        x, y = self.rotate(point)
        return (self.offset[0] + x * self.scale, self.offset[1] - y * self.scale)


def fit_transform(shapes: Sequence[Shape], angle: float, width: int, height: int, desired_scale: float, margin: float) -> _Transform:
    """Подобрать масштаб и смещение, чтобы птица целиком помещалась в безопасную область."""
    probe = _Transform(angle, 1.0, (0.0, 0.0))
    xs: list[float] = []
    ys: list[float] = []
    for shape in shapes:
        for point in shape.points:
            x, y = probe.rotate(point)
            xs.append(x)
            ys.append(y)
    if not xs:
        raise RenderingError("Геометрия птицы пуста")
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    bird_w = max(max_x - min_x, 1e-6)
    bird_h = max(max_y - min_y, 1e-6)
    available_w = width * (1 - 2 * margin)
    available_h = height * (1 - 2 * margin)
    scale = min(desired_scale, available_w / bird_w, available_h / bird_h)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    offset = (width / 2 - center_x * scale, height / 2 + center_y * scale)
    return _Transform(angle, scale, offset)


def _draw_shape(draw: ImageDraw.ImageDraw, shape: Shape, transform: _Transform) -> None:
    points = [transform(p) for p in shape.points]
    if len(points) < 2:
        return
    if shape.closed and shape.fill is not None and len(points) >= 3:
        draw.polygon(points, fill=shape.fill)
    if shape.outline is not None and shape.width > 0:
        pixel_width = max(1, int(round(shape.width * transform.scale)))
        outline_points = points + [points[0]] if shape.closed else points
        draw.line(outline_points, fill=shape.outline, width=pixel_width, joint="curve")


def _render_glow(canvas: Image.Image, glows: Sequence[tuple[Point, float, float, tuple[int, int, int], float]], transform: _Transform) -> None:
    active = [g for g in glows if g[4] > 0.05]
    if not active:
        return
    factor = 4
    small_size = (max(1, canvas.width // factor), max(1, canvas.height // factor))
    layer = Image.new("RGBA", small_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for center, rx, ry, color, strength in active:
        cx, cy = transform(center)
        prx, pry = rx * transform.scale / factor, ry * transform.scale / factor
        draw.ellipse((cx / factor - prx, cy / factor - pry, cx / factor + prx, cy / factor + pry),
                     fill=rgba(color, clamp(0.15 + 0.55 * strength, 0, 0.7)))
    radius = max(2.0, min(small_size) * 0.06)
    layer = layer.filter(ImageFilter.GaussianBlur(radius)).resize(canvas.size, Image.Resampling.BICUBIC)
    canvas.paste(layer.convert("RGB"), (0, 0), layer.getchannel("A"))


def render_bird(
    spec: BirdSpec,
    width: int = 1200,
    height: int = 1200,
    supersampling: int = 2,
    background: str = "auto",
    show_keywords: bool = False,
    mapping_config: dict[str, Any] | None = None,
    font_candidates: Sequence[str] | None = None,
    progress: Callable[[Image.Image, str], None] | None = None,
) -> tuple[Image.Image, list[str], dict[str, Any]]:
    """Нарисовать птицу по BirdSpec. Возвращает (изображение RGB, предупреждения, сведения о рендере).

    ``progress(image, label)`` вызывается на промежуточных этапах рисования
    (для окна предпросмотра). Он получает текущий холст и не должен его менять;
    на итоговое изображение вызов не влияет.
    """
    if not MIN_CANVAS <= width <= MAX_CANVAS or not MIN_CANVAS <= height <= MAX_CANVAS:
        raise RenderingError(f"Размер холста должен быть от {MIN_CANVAS} до {MAX_CANVAS} пикселей")
    if supersampling not in (1, 2, 3):
        raise RenderingError("supersampling должен быть 1, 2 или 3")
    if spec.primary_archetype and not spec.style:
        return render_art_bird(spec, width, height, supersampling, background, show_keywords,
                               mapping_config, font_candidates, progress)
    warnings: list[str] = []
    safe = (mapping_config or {}).get("safe_ranges", {})
    limits = (mapping_config or {}).get("limits", {})
    margin = float(safe.get("canvas_margin", 0.07))
    min_fraction = float(safe.get("minimum_bird_fraction", 0.42))
    max_fraction = float(safe.get("maximum_bird_fraction", 0.66))
    min_tilt = float(safe.get("minimum_tilt_degrees", -6.0))
    max_tilt = float(safe.get("maximum_tilt_degrees", 22.0))
    max_bg = int(limits.get("maximum_background_elements", 90))

    try:
        rng = random.Random(spec.seed ^ 0x5EED_B1D)
        big_w, big_h = width * supersampling, height * supersampling
        unit = min(big_w, big_h)
        style = STYLES.get(spec.style)
        if style and background == "auto":
            mode = style.background_mode
        else:
            mode = choose_background_mode(spec, background)
        canvas = render_background(big_w, big_h, spec, mode)
        if progress:
            progress(canvas, "Фон")
        draw = ImageDraw.Draw(canvas, "RGBA")
        if style:
            draw_style_ornament(draw, big_w, big_h, spec, random.Random(spec.seed ^ 0x0A7E))
            if progress:
                progress(canvas, f"Орнамент стиля «{style.title}»")
        if not style or style.background_motifs:
            draw_background_motifs(draw, big_w, big_h, spec, rng, max_bg)
            if progress:
                progress(canvas, "Фоновые мотивы")

        geometry = build_bird_geometry(spec, rng)
        angle = math.radians(min_tilt + (max_tilt - min_tilt) * spec.body_tilt)
        desired = unit * (min_fraction + (max_fraction - min_fraction) * spec.body_scale)
        all_shapes = geometry.behind + geometry.body + geometry.front + geometry.particles
        transform = fit_transform(all_shapes, angle, big_w, big_h, desired, margin)

        _render_glow(canvas, geometry.glow, transform)
        if progress:
            progress(canvas, "Свечение")
        draw = ImageDraw.Draw(canvas, "RGBA")
        stages = (
            ("Дальнее крыло и хвост", geometry.behind),
            ("Корпус и перья груди", geometry.body),
            ("Голова, клюв, глаз и ближнее крыло", geometry.front),
            ("Отделяющиеся частицы", geometry.particles),
        )
        step = max(1, math.ceil(len(all_shapes) / 45))
        drawn = 0
        for label, shapes in stages:
            for shape in shapes:
                _draw_shape(draw, shape, transform)
                drawn += 1
                if progress and drawn % step == 0:
                    progress(canvas, label)
            if progress and shapes:
                progress(canvas, label)

        image = canvas.resize((width, height), Image.Resampling.LANCZOS) if supersampling > 1 else canvas
        image = image.convert("RGB")
        if style and style.post:
            image = apply_style_post(image, spec)
            if progress:
                progress(image, "Фактура стиля")
        font_used = None
        if show_keywords and spec.keywords:
            paths_px: dict[str, tuple[list[Point], tuple[int, int, int], float]] = {}
            for name, (points, color, thickness) in geometry.text_paths.items():
                projected = [transform(point) for point in points]
                paths_px[name] = (
                    [(x / supersampling, y / supersampling) for x, y in projected],
                    color,
                    thickness * transform.scale / supersampling,
                )
            font_used = draw_keywords(image, spec, paths_px, warnings, font_candidates)
            if progress and font_used:
                progress(image, "Ключевые слова из запроса")
    except RenderingError:
        raise
    except (ValueError, MemoryError, OSError) as exc:
        raise RenderingError(f"Не удалось построить изображение: {exc}") from exc
    info = {"background_mode": mode, "font": font_used, "style": spec.style, "bird_scale_px": _round(transform.scale / supersampling, 2)}
    return image, warnings, info


# ============================================================================
# 22. Рендер мотивов — см. draw_background_motifs и build_bird_geometry
# ============================================================================

# ============================================================================
# 23. Рендер ключевых слов
# ============================================================================


def _render_probe(font: ImageFont.FreeTypeFont, text: str) -> bytes:
    image = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(image).text((4, 4), text, font=font, fill=255)
    return image.tobytes()


def font_supports_cyrillic(font: ImageFont.FreeTypeFont) -> bool:
    """Проверить, что шрифт рисует кириллицу, а не пустые/замещающие глифы."""
    try:
        cyrillic = _render_probe(font, "Жщ")
        missing = _render_probe(font, "\U000F0000\U000F0001")
        return any(cyrillic) and cyrillic != missing
    except (OSError, ValueError):
        return False


def load_font(reference: str, size: int) -> ImageFont.FreeTypeFont:
    """Загрузить шрифт по ссылке "путь" или "путь#индекс" (начертание в .ttc)."""
    path, _, index = reference.partition("#")
    return ImageFont.truetype(path, size, index=int(index) if index.isdigit() else 0)


def find_cyrillic_font(size: int, candidates: Sequence[str] | None = None) -> tuple[ImageFont.FreeTypeFont, str] | None:
    """Найти локальный шрифт с кириллицей. Возвращает (шрифт, ссылка). Шрифты не скачиваются."""
    for candidate in candidates if candidates is not None else FONT_CANDIDATES:
        if not Path(candidate.partition("#")[0]).is_file():
            continue
        try:
            font = load_font(candidate, size)
        except (OSError, ValueError):
            continue
        if font_supports_cyrillic(font):
            return font, candidate
    return None


def keyword_font_candidates(custom: str | Path | None = None) -> list[str]:
    """Порядок поиска шрифта для слов: свой шрифт → каллиграфические → обычные."""
    ordered = [str(custom)] if custom else []
    return ordered + list(CALLIGRAPHY_FONT_CANDIDATES) + list(FONT_CANDIDATES)


def _path_lengths(points: Sequence[Point]) -> list[float]:
    lengths = [0.0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(x1 - x0, y1 - y0))
    return lengths


def _point_on_path(points: Sequence[Point], lengths: Sequence[float], distance: float) -> tuple[Point, float]:
    """Точка и угол касательной на расстоянии distance вдоль ломаной."""
    distance = clamp(distance, 0.0, lengths[-1])
    for index in range(1, len(points)):
        if lengths[index] >= distance or index == len(points) - 1:
            segment = max(lengths[index] - lengths[index - 1], 1e-9)
            t = (distance - lengths[index - 1]) / segment
            (x0, y0), (x1, y1) = points[index - 1], points[index]
            return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t), math.atan2(y1 - y0, x1 - x0)
    return points[-1], 0.0


def _engraving_colors(fill: Sequence[int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Цвет букв и блика для эффекта «гравировки» по цвету области птицы."""
    if rgb_lightness(fill) > 0.5:
        return mix(fill, (0, 0, 0), 0.5), mix(fill, (255, 255, 255), 0.45)
    return mix(fill, (255, 255, 255), 0.62), mix(fill, (0, 0, 0), 0.45)


def draw_text_on_path(image: Image.Image, text: str, points: Sequence[Point], font_path: str,
                      max_size: float, min_size: int, fill: Sequence[int], tracking: float = 0.0) -> bool:
    """Написать слово по буквам вдоль ломаной, поворачивая каждую букву по касательной.

    Размер шрифта подбирается так, чтобы слово занимало ~85% длины линии
    и не было толще области птицы. Возвращает False, если линия слишком мала.
    """
    if len(points) < 2:
        return False
    if points[-1][0] < points[0][0]:
        points = list(points)[::-1]  # читаем слева направо, чтобы буквы не были перевёрнуты
    lengths = _path_lengths(points)
    total = lengths[-1]
    if total < min_size * 2:
        return False
    probe = load_font(font_path, 100)
    width_at_100 = probe.getlength(text) * (1 + tracking)
    size = int(min(max_size, 100 * 0.85 * total / max(width_at_100, 1e-6)))
    if size < min_size:
        size = min_size
    font = load_font(font_path, size)
    # Шаг каждой буквы берётся из длины префикса слова — так учитывается кернинг,
    # и соединения рукописных букв не расходятся.
    prefix_lengths = [font.getlength(text[:i]) for i in range(len(text) + 1)]
    advances = [(prefix_lengths[i + 1] - prefix_lengths[i]) * (1 + tracking) for i in range(len(text))]
    text_length = sum(advances)
    if text_length > total * 1.05:
        return False
    ink, highlight = _engraving_colors(fill)
    cell = int(size * 3.0)  # с запасом для росчерков каллиграфических букв
    distance = (total - text_length) / 2
    for char, advance in zip(text, advances):
        (x, y), angle = _point_on_path(points, lengths, distance + advance / 2)
        distance += advance
        if char.isspace():
            continue
        glyph = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
        glyph_draw = ImageDraw.Draw(glyph)
        baseline = cell / 2 + size * 0.35
        offset = max(1, size // 22)
        glyph_draw.text((cell / 2 + offset, baseline + offset), char, font=font, fill=rgba(highlight, 0.35), anchor="ms")
        glyph_draw.text((cell / 2, baseline), char, font=font, fill=rgba(ink, 0.88), anchor="ms")
        glyph = glyph.rotate(-math.degrees(angle), resample=Image.Resampling.BICUBIC)
        left, top = int(round(x - cell / 2)), int(round(y - cell / 2))
        if left < 0 or top < 0 or left + cell > image.width or top + cell > image.height:
            continue
        image.paste(glyph, (left, top), glyph)
    return True


def draw_keywords(image: Image.Image, spec: BirdSpec, paths: dict[str, tuple[list[Point], tuple[int, int, int], float]],
                  warnings: list[str], font_candidates: Sequence[str] | None = None) -> str | None:
    """Вписать до трёх ключевых слов внутрь птицы вдоль её линий.

    Первое слово идёт по средней линии ближнего крыла, второе — по изгибу
    брюшка внутри корпуса, третье — вдоль оси центрального пера хвоста.
    Цвет букв выводится из цвета соответствующей части птицы.
    """
    min_size = max(12, int(min(image.size) * 0.016))
    candidates = font_candidates if font_candidates is not None else keyword_font_candidates()
    found = find_cyrillic_font(min_size, candidates)
    if found is None:
        warnings.append("Шрифт с поддержкой кириллицы не найден: ключевые слова не нарисованы")
        return None
    _, font_path = found
    max_size_limit = min(image.size) * 0.055
    placed = 0
    for word, region in zip(spec.keywords[:3], ("wing", "body", "tail")):
        if region not in paths:
            continue
        points, color, thickness = paths[region]
        max_size = min(max_size_limit, thickness * 0.55)
        text = word[:20]
        if draw_text_on_path(image, text, points, font_path, max_size, min_size, color):
            placed += 1
        else:
            warnings.append(f"Ключевое слово №{spec.keywords.index(word) + 1} не поместилось вдоль линии птицы")
    return font_path if placed else None


# ============================================================================
# 21a. Послойный художественный рендер (архетипы, материалы, композиция)
# ============================================================================


@dataclass
class ArtFeather:
    """Перо: основание, направление, длина, ширина, положение (0 — у корня, 1 — периферия), часть тела."""

    base: Point
    angle: float
    length: float
    width: float
    t: float
    part: str


@dataclass
class ArtGeometry:
    """Геометрия птицы в локальных координатах (ось y вверх), отдельно от материала."""

    body: list[Point]
    head_center: Point
    head_radius: float
    beak: list[Point]
    eyes: list[tuple[Point, float]]
    wings: dict[str, list[Point]]
    feathers: list[ArtFeather]
    core: tuple[Point, float, float]
    particles: list[tuple[Point, float, float, float]]
    ghosts: list[tuple[float, float, float]]
    ring: list[Point]
    anchors: dict[str, Point]
    text_paths: dict[str, tuple[list[Point], tuple[int, int, int], float]]
    frontal: bool


WING_SHAPE_PARAMS: dict[str, tuple[float, float, float, float]] = {
    # bulge передней кромки, bulge задней кромки, «стягивание» к кончику, ширина корня
    "pointed": (0.12, 0.07, 0.25, 0.22),
    "rounded": (0.30, 0.14, 0.60, 0.24),
    "broad": (0.22, 0.20, 0.45, 0.32),
    "sickle": (-0.05, 0.22, 0.20, 0.22),
    "wave": (0.20, 0.10, 0.40, 0.24),
}


def _art_wing(spec: BirdSpec, root: Point, angle: float, length: float, count: int, part: str,
              rng: random.Random) -> tuple[list[Point], list[ArtFeather], list[tuple[Point, float]]]:
    """Крыло заданной формы: контур из кривых Безье, перья вдоль задней кромки и средняя линия."""
    bulge_k, trailing_k, tip_pull, root_width = WING_SHAPE_PARAMS.get(spec.wing_shape, WING_SHAPE_PARAMS["pointed"])
    roundness = spec.wing_roundness
    axis = (math.cos(angle), math.sin(angle))
    normal = (axis[1], -axis[0])
    tip = (root[0] + axis[0] * length, root[1] + axis[1] * length)
    root_front = (root[0] + root_width * 0.4, root[1] + 0.01)
    root_back = (root[0] - root_width * 0.6, root[1] - 0.02)
    bulge = length * (bulge_k + 0.1 * roundness)
    trailing = length * (trailing_k + 0.06 * roundness)
    leading = cubic_bezier(
        root_front,
        (root_front[0] + axis[0] * length * 0.35 + normal[0] * bulge, root_front[1] + axis[1] * length * 0.35 + normal[1] * bulge),
        (tip[0] - axis[0] * length * tip_pull * 0.4 + normal[0] * bulge * 0.6, tip[1] - axis[1] * length * tip_pull * 0.4 + normal[1] * bulge * 0.6),
        tip, 28,
    )
    back = cubic_bezier(
        tip,
        (tip[0] - axis[0] * length * 0.3 - normal[0] * trailing, tip[1] - axis[1] * length * 0.3 - normal[1] * trailing),
        (root_back[0] + axis[0] * length * 0.3 - normal[0] * trailing, root_back[1] + axis[1] * length * 0.3 - normal[1] * trailing),
        root_back, 28,
    )
    outline = leading + back[1:]
    back_reversed = back[::-1]
    midline = []
    for index in range(3, len(leading) - 5):
        lead, trail = leading[index], back_reversed[index]
        midline.append(((lead[0] * 0.58 + trail[0] * 0.42, lead[1] * 0.58 + trail[1] * 0.42),
                        math.hypot(lead[0] - trail[0], lead[1] - trail[1])))
    feathers: list[ArtFeather] = []
    count = max(3, count)
    gap_scale = max(0.0, spec.abstraction_level - 0.3) * 0.6 + 0.25 * spec.silhouette_openness
    for index in range(count):
        t = 0.08 + 0.9 * index / max(1, count - 1)
        point = back[min(len(back) - 1, int((1 - t) * (len(back) - 1)))]
        direction = _normalize((-normal[0] * (1.0 - 0.55 * t) + axis[0] * (0.25 + 0.75 * t),
                                -normal[1] * (1.0 - 0.55 * t) + axis[1] * (0.25 + 0.75 * t)))
        feather_angle = math.atan2(direction[1], direction[0]) + (spec.fragmentation * 0.4 + spec.wing_asymmetry * 0.2) * rng.uniform(-1, 1)
        feather_length = length * (0.16 + 0.26 * t) * (0.85 + 0.3 * rng.random()) * (0.8 + 0.4 * spec.feather_density)
        if spec.wing_shape == "wave":
            feather_length *= 1 + 0.35 * math.sin(t * math.pi * 4)
        gap = gap_scale * feather_length * rng.random() * t if t > 0.5 else 0.0
        base = (point[0] - direction[0] * feather_length * 0.25 + direction[0] * gap,
                point[1] - direction[1] * feather_length * 0.25 + direction[1] * gap)
        feathers.append(ArtFeather(base, feather_angle, feather_length, feather_length * (0.22 + 0.2 * roundness), t, part))
    return outline, feathers, midline


def _mirror_x(points: Sequence[Point]) -> list[Point]:
    return [(-x, y) for x, y in points]


def _art_tail(spec: BirdSpec, root: Point, direction: float, length: float, rng: random.Random) -> list[ArtFeather]:
    """Хвост выбранного типа: веер, ленты, вилка, короткий, спираль или раздвоенный."""
    feathers: list[ArtFeather] = []
    spread = math.radians(14 + 50 * spec.tail_spread)
    if spec.tail_type == "streamer":
        for index, offset in enumerate((-0.12, 0.0, 0.12)):
            feathers.append(ArtFeather(root, direction + offset, length * (1.5 - 0.2 * abs(index - 1)), length * 0.1, 1.0, "tail"))
        return feathers
    if spec.tail_type == "short":
        count, length = 4, length * 0.6
    elif spec.tail_type == "spiral":
        for index in range(7):
            angle = direction + 0.35 * index - 0.6
            feathers.append(ArtFeather(root, angle, length * (0.9 - 0.08 * index), length * 0.14, index / 6, "tail"))
        return feathers
    else:
        count = 3 + int(round(4 * spec.tail_spread))
    for index in range(count):
        offset = (index / max(1, count - 1) - 0.5) * spread
        if spec.tail_type in ("forked", "split"):
            offset += (0.18 if offset >= 0 else -0.18) * (1.6 if spec.tail_type == "split" else 1.0)
        feather_len = length * (0.8 + 0.35 * (1 - abs(offset) / max(spread, 1e-6))) * (0.9 + 0.2 * rng.random())
        if spec.tail_type == "forked":
            feather_len *= 1.0 + 0.4 * abs(offset) / max(spread, 1e-6)
        feathers.append(ArtFeather(root, direction + offset, feather_len, feather_len * (0.16 + 0.14 * spec.body_roundness),
                                   abs(offset) / max(spread, 1e-6), "tail"))
    return feathers


def _art_particles(spec: BirdSpec, zone: str, anchors: dict[str, Point], feathers: Sequence[ArtFeather], body: Sequence[Point],
                   radius: float, rng: random.Random) -> list[tuple[Point, float, float, float]]:
    """Частицы по зоне композиции: (позиция, размер, угол, глубина 0 — далеко, 1 — близко)."""
    result: list[tuple[Point, float, float, float]] = []
    count = spec.particle_count
    outer = [f for f in feathers if f.t > 0.55] or list(feathers)
    for index in range(count):
        depth = rng.random()
        size = (0.003 + 0.01 * rng.random()) * (0.6 + 0.8 * depth)
        if zone == "ring":
            angle = rng.uniform(0, 2 * math.pi)
            dist = radius * (1.05 + 0.15 * rng.gauss(0, 1))
            pos = (math.cos(angle) * dist, math.sin(angle) * dist * 0.85)
        elif zone == "spiral":
            t = index / max(1, count - 1)
            angle = t * 4 * math.pi + rng.uniform(-0.2, 0.2)
            dist = radius * (0.35 + 0.8 * t)
            pos = (math.cos(angle) * dist, math.sin(angle) * dist)
        elif zone == "trail":
            pos = (-radius * rng.uniform(0.6, 1.8), anchors["tail"][1] * 0.6 + radius * 0.25 * rng.gauss(0, 0.5))
        elif zone == "tight":
            angle = rng.uniform(0, 2 * math.pi)
            dist = radius * rng.uniform(0.55, 0.85)
            pos = (math.cos(angle) * dist, math.sin(angle) * dist)
        elif zone == "behind_low":
            angle = math.radians(rng.uniform(160, 250))
            dist = radius * rng.uniform(0.6, 1.35)
            pos = (math.cos(angle) * dist, math.sin(angle) * dist)
        elif zone == "tail":
            tip = anchors["tail"]
            pos = (tip[0] - radius * rng.uniform(0.0, 0.6), tip[1] + radius * rng.gauss(0, 0.2))
        else:  # escape / outer: от внешних перьев наружу
            feather = outer[index % len(outer)]
            tip = (feather.base[0] + math.cos(feather.angle) * feather.length, feather.base[1] + math.sin(feather.angle) * feather.length)
            reach = radius * (1.2 if zone == "escape" else 0.6)
            angle = feather.angle + rng.uniform(-0.5, 0.5)
            if zone == "escape":
                angle = angle * 0.5 + math.radians(150) * 0.5
            dist = reach * rng.random() ** 0.8
            pos = (tip[0] + math.cos(angle) * dist, tip[1] + math.sin(angle) * dist)
        result.append((pos, size, rng.uniform(0, math.pi), depth))
    if spec.rare_feature == "dissolving_edge":
        for index in range(0, len(body), 2):
            x, y = body[index]
            push = 1.0 + rng.uniform(0.05, 0.35)
            result.append(((x * push, y * push), 0.004 + 0.006 * rng.random(), rng.uniform(0, math.pi), 0.8))
    return result


def build_art_geometry(spec: BirdSpec, rng: random.Random) -> ArtGeometry:
    """Геометрия птицы: композиция и силуэт влияют на крылья, голову, хвост, ядро и частицы."""
    configs = active_configs()
    comp = configs.compositions["compositions"].get(spec.composition_type, {})
    frontal = bool(comp.get("frontal", False))
    zone = str(comp.get("particle_zone", "outer"))
    spread_factor = float(comp.get("wing_spread", 0.6))
    fragile = spec.silhouette_type == "fragile"
    elong = spec.body_elongation
    a = 0.25 * (0.85 + 0.35 * elong) * (0.88 if fragile else 1.0)
    b = a * (0.62 - 0.25 * elong)
    taper = 0.55 if spec.silhouette_type == "droplet" else 0.38
    wobble = 0.03 * spec.fragmentation + 0.015 * spec.wing_asymmetry + (0.02 if spec.material_type == "leaves" else 0.0)
    contour = _body_contour(a, b, spec.body_roundness, wobble, rng)
    if taper != 0.38:
        contour = [(x, y * (1 - (taper - 0.38) * max(0.0, -x / a))) for x, y in contour]
    head_r = 0.085 * (0.75 + 0.6 * spec.head_scale) * (0.92 if spec.abstraction_level > 0.6 else 1.0)
    wing_length = 0.36 * (0.62 + 0.8 * spec.wing_span)
    near_count = max(4, int(spec.feather_count * 0.38))
    far_count = max(3, int(spec.feather_count * 0.25))
    feathers: list[ArtFeather] = []
    wings: dict[str, list[Point]] = {}
    text_paths: dict[str, tuple[list[Point], tuple[int, int, int], float]] = {}
    anchors: dict[str, Point] = {}
    eyes: list[tuple[Point, float]] = []

    if frontal:
        body = [(-y * 1.25, x) for x, y in contour]  # поворот на 90°: птица смотрит на зрителя
        top = max(y for _, y in body)
        bottom = min(y for _, y in body)
        width = max(x for x, _ in body)
        head_c = (0.0, top + head_r * (0.45 + 0.5 * spec.head_lift))
        beak_len = head_r * (0.35 + 0.5 * spec.beak_length)
        beak_half = head_r * (0.28 - 0.12 * spec.beak_sharpness)
        beak = [(-beak_half, head_c[1] - head_r * 0.15), (beak_half, head_c[1] - head_r * 0.15), (0.0, head_c[1] - head_r * 0.15 - beak_len)]
        eye_r = head_r * (0.12 + 0.2 * spec.eye_scale)
        eyes = [((-head_r * 0.42, head_c[1] + head_r * 0.18), eye_r), ((head_r * 0.42, head_c[1] + head_r * 0.18), eye_r)]
        angle_right = math.radians(8 + 45 * spec.wing_sweep)
        root_right = (width * 0.6, top * 0.2)
        right, right_feathers, right_mid = _art_wing(spec, root_right, angle_right, wing_length, near_count, "near_wing", rng)
        asym = 1.0 - 0.3 * spec.wing_asymmetry
        left, left_feathers, left_mid = _art_wing(spec, root_right, angle_right + math.radians(8 * spec.wing_asymmetry),
                                                   wing_length * asym, near_count, "far_wing", rng)
        wings["far"] = _mirror_x(left)
        wings["near"] = right
        for feather in left_feathers:
            feathers.append(ArtFeather((-feather.base[0], feather.base[1]), math.pi - feather.angle, feather.length, feather.width, feather.t, "far_wing"))
        feathers.extend(right_feathers)
        tail = _art_tail(spec, (0.0, bottom * 0.9), -math.pi / 2, 0.14 + 0.25 * spec.tail_length, rng)
        feathers.extend(tail)
        core_center = (0.0, (top + bottom) * 0.35)
        core = (core_center, width * 0.55, (top - bottom) * 0.28)
        anchors.update({"eye": eyes[1][0], "wing_tip": right[len(right) // 2 - 1], "tail": (0.0, bottom - 0.2 * spec.tail_length)})
        body_path = [(x, bottom * 0.35 + 0.02 * math.sin(x * 12)) for x in np.linspace(-width * 0.75, width * 0.75, 20)]
        text_paths["wing"] = ([p for p, _ in right_mid], mix(spec.secondary_color, spec.primary_color, 0.45),
                              sorted(w for _, w in right_mid)[len(right_mid) // 2] if right_mid else 0.05)
        text_paths["body"] = (body_path, spec.primary_color, (top - bottom) * 0.35)
        text_paths["tail"] = ([(0.0, bottom * 0.95 - t * 0.2 * spec.tail_length) for t in np.linspace(0.1, 1.0, 14)],
                              mix(spec.secondary_color, spec.accent_color, 0.25), 0.05)
    else:
        body = contour
        head_c = (a * 0.8 + head_r * 0.3, b * 0.35 + head_r * (0.25 + 1.05 * spec.head_lift))
        beak_base = (head_c[0] + head_r * 0.9, head_c[1] - head_r * 0.12)
        beak_len = head_r * (0.6 + 1.6 * spec.beak_length)
        beak_half = head_r * (0.5 - 0.3 * spec.beak_sharpness)
        tip = (beak_base[0] + beak_len, beak_base[1] - beak_len * (0.08 + 0.2 * (1 - spec.head_lift)))
        beak = [(beak_base[0] - head_r * 0.1, beak_base[1] + beak_half), tip, (beak_base[0] - head_r * 0.1, beak_base[1] - beak_half)]
        eye_r = head_r * (0.14 + 0.3 * spec.eye_scale)
        eyes = [((head_c[0] + head_r * 0.3, head_c[1] + head_r * 0.18), eye_r)]
        near_angle = math.radians(185 - 95 * spec.wing_sweep)
        spread = math.radians(12 + 38 * spec.wing_span) * (0.5 + spread_factor)
        far_angle = near_angle - spread + math.radians(rng.uniform(-1, 1) * 25 * spec.wing_asymmetry)
        far, far_feathers, _ = _art_wing(spec, (0.03, b * 0.72), far_angle, wing_length * (0.92 - 0.2 * spec.wing_asymmetry * rng.random()), far_count, "far_wing", rng)
        near, near_feathers, near_mid = _art_wing(spec, (-0.04, b * 0.62), near_angle, wing_length, near_count, "near_wing", rng)
        wings["far"], wings["near"] = far, near
        feathers.extend(far_feathers)
        feathers.extend(near_feathers)
        tail_root = (-a * 0.9, -b * 0.05)
        tail_dir = math.radians(168 + 48 * spec.tail_drop)
        tail = _art_tail(spec, tail_root, tail_dir, 0.16 + 0.3 * spec.tail_length, rng)
        feathers.extend(tail)
        core = ((a * 0.2, -b * 0.05), a * 0.45, b * 0.55)
        tail_mid = tail[len(tail) // 2]
        tail_tip = (tail_root[0] + math.cos(tail_mid.angle) * tail_mid.length, tail_root[1] + math.sin(tail_mid.angle) * tail_mid.length)
        anchors.update({"eye": eyes[0][0], "wing_tip": near[27], "tail": tail_tip})
        text_paths["wing"] = ([p for p, _ in near_mid], mix(spec.secondary_color, spec.primary_color, 0.45),
                              sorted(w for _, w in near_mid)[len(near_mid) // 2] if near_mid else 0.05)
        text_paths["body"] = ([(x * 0.66, y * 0.66 + b * 0.08) for x, y in contour[40:70]], spec.primary_color, b * 0.7)
        text_paths["tail"] = ([(tail_root[0] + math.cos(tail_mid.angle) * tail_mid.length * t,
                                tail_root[1] + math.sin(tail_mid.angle) * tail_mid.length * t) for t in np.linspace(0.92, 0.12, 16)],
                              mix(spec.secondary_color, spec.accent_color, 0.25), tail_mid.width * 1.6)

    if spec.rare_feature == "shifted_core":
        core = ((core[0][0] - 0.08, core[0][1] + 0.04), core[1] * 0.8, core[2] * 0.8)
    anchors["core"] = core[0]
    anchors["head"] = head_c
    xs = [p[0] for p in body] + [p[0] for w in wings.values() for p in w]
    ys = [p[1] for p in body] + [p[1] for w in wings.values() for p in w]
    radius = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys))
    particles = _art_particles(spec, zone, anchors, feathers, body, radius, rng)
    ghosts: list[tuple[float, float, float]] = []
    if spec.rare_feature == "secondary_ghost" or spec.archetype_weights.get("memory", 0) >= 0.2 \
            or spec.archetype_weights.get("typographic", 0) >= 0.2:
        for k in range(1, 1 + min(3, spec.layer_count - 1)):
            ghosts.append((-0.045 * k, 0.02 * k, 0.22 / k))
    ring: list[Point] = []
    if spec.rare_feature == "ring_trace":
        ring = [(math.cos(t) * radius * 1.15, math.sin(t) * radius * 0.8) for t in np.linspace(0, 2 * math.pi, 120)]
    if xs:
        escape_x = min(xs)
        anchors["break_zone"] = (escape_x + radius * 0.2, max(ys) * 0.6)
    return ArtGeometry(body, head_c, head_r, beak, eyes, wings, feathers, core, particles, ghosts, ring, anchors, text_paths, frontal)


# --- материалы ----------------------------------------------------------------

@dataclass
class PaintContext:
    """Всё, что нужно материалам для рисования части птицы."""

    spec: BirdSpec
    transform: "_Transform"
    rng: random.Random
    unit: float
    line: float
    light_dir: Point
    font_path: str | None
    glyphs: str
    smoke: Image.Image | None = None


def _centroid(points: Sequence[Point]) -> Point:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def _scaled(points: Sequence[Point], factor: float, shift: Point = (0.0, 0.0)) -> list[Point]:
    cx, cy = _centroid(points)
    return [(cx + (x - cx) * factor + shift[0], cy + (y - cy) * factor + shift[1]) for x, y in points]


def _bbox_int(points: Sequence[Point], pad: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
            min(size[0], int(max(xs)) + pad + 1), min(size[1], int(max(ys)) + pad + 1))


def _clip_pattern(layer: Image.Image, points: Sequence[Point], painter: Callable[[ImageDraw.ImageDraw, int, int, int, int], None]) -> None:
    """Нарисовать узор (штриховку, сетку) только внутри многоугольника."""
    x0, y0, x1, y1 = _bbox_int(points, 2, layer.size)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    tile = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    painter(ImageDraw.Draw(tile, "RGBA"), x0, y0, x1, y1)
    mask = Image.new("L", tile.size, 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in points], fill=255)
    alpha = tile.getchannel("A")
    tile.putalpha(Image.fromarray((np.asarray(alpha, dtype=np.uint16) * np.asarray(mask, dtype=np.uint16) // 255).astype(np.uint8)))
    region = layer.crop((x0, y0, x1, y1))
    layer.paste(Image.alpha_composite(region, tile), (x0, y0))


def _sample_in_polygon(points: Sequence[Point], count: int, rng: random.Random, center_bias: float = 1.0) -> list[Point]:
    """Точки внутри (звёздного) многоугольника; center_bias > 1 сгущает их к центру."""
    center = _centroid(points)
    triangles = []
    for p, q in zip(points, list(points[1:]) + [points[0]]):
        area = abs((p[0] - center[0]) * (q[1] - center[1]) - (q[0] - center[0]) * (p[1] - center[1])) / 2
        triangles.append((area, p, q))
    total = sum(t[0] for t in triangles)
    if total <= 0:
        return []
    result = []
    for _ in range(count):
        pick = rng.random() * total
        running = 0.0
        for area, p, q in triangles:
            running += area
            if running >= pick:
                break
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        scale = rng.random() ** (center_bias - 1) if center_bias > 1 else 1.0
        x = center[0] + (u * (p[0] - center[0]) + v * (q[0] - center[0])) * scale
        y = center[1] + (u * (p[1] - center[1]) + v * (q[1] - center[1])) * scale
        result.append((x, y))
    return result


def _polygon_area(points: Sequence[Point]) -> float:
    return abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(points, list(points[1:]) + [points[0]]))) / 2


def _glyph(layer: Image.Image, char: str, center: Point, angle: float, size: int, color: tuple[int, int, int, int], font_path: str) -> None:
    font = load_font(font_path, max(6, size))
    cell = int(size * 2.4) + 2
    glyph = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
    ImageDraw.Draw(glyph).text((cell / 2, cell / 2), char, font=font, fill=color, anchor="mm")
    glyph = glyph.rotate(-math.degrees(angle), resample=Image.Resampling.BICUBIC)
    left, top = int(center[0] - cell / 2), int(center[1] - cell / 2)
    if 0 <= left and 0 <= top and left + cell <= layer.width and top + cell <= layer.height:
        region = layer.crop((left, top, left + cell, top + cell))
        layer.paste(Image.alpha_composite(region, glyph), (left, top))


def paint_region(layer: Image.Image, points: Sequence[Point], color: Sequence[int], material: str, ctx: PaintContext,
                 strength: float = 1.0, axis_angle: float = 0.0) -> None:
    """Нарисовать область (корпус, крыло, перо) выбранным материалом."""
    if len(points) < 3:
        return
    draw = ImageDraw.Draw(layer, "RGBA")
    spec = ctx.spec
    base_alpha = float(active_configs().materials["materials"].get(material, {}).get("alpha", 0.8)) * strength
    light = mix(color, (255, 255, 255), 0.55)
    dark = mix(color, (0, 0, 0), 0.45)
    size = math.sqrt(max(_polygon_area(points), 1.0))
    if material == "light":
        draw.polygon(points, fill=rgba(color, base_alpha * 0.8))
        draw.polygon(_scaled(points, 0.72), fill=rgba(mix(color, light, 0.5), base_alpha * 0.55))
        draw.polygon(_scaled(points, 0.42), fill=rgba(light, base_alpha * 0.45))
    elif material == "glass":
        draw.polygon(points, fill=rgba(color, base_alpha))
        shift = (ctx.light_dir[0] * size * 0.08, ctx.light_dir[1] * size * 0.08)
        draw.polygon(_scaled(points, 0.7, shift), fill=rgba((255, 255, 255), 0.13 * strength))
        if spec.accent_effect == "caustic_dots":
            for x, y in _sample_in_polygon(points, max(2, int(size / 18)), ctx.rng, 1.5):
                r = ctx.unit * 0.0025
                draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba((255, 255, 255), 0.55))
    elif material == "ink":
        draw.polygon(points, fill=rgba(mix(color, spec.outline_color, 0.2), base_alpha * 0.85))
        draw.polygon(_scaled(points, 0.82, (size * 0.03, -size * 0.03)), fill=rgba(dark, 0.18 * strength))
        if spec.accent_effect == "hatching" and size > ctx.unit * 0.04:
            gap = max(3, int(ctx.unit * 0.009))

            def hatch(tile_draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
                for offset in range(-(y1 - y0), x1 - x0, gap):
                    tile_draw.line((offset, 0, offset + (y1 - y0), y1 - y0), fill=rgba(dark, 0.35), width=max(1, gap // 4))

            _clip_pattern(layer, points, hatch)
    elif material == "particles":
        draw.polygon(points, fill=rgba(color, 0.08 * strength))
        radius = ctx.unit * 0.0035
        count = int(min(900, _polygon_area(points) / (radius * radius * 9)))
        bias = 1.6 if spec.accent_effect == "density_gradient" else 1.2
        for x, y in _sample_in_polygon(points, count, ctx.rng, bias):
            r = radius * (0.5 + ctx.rng.random())
            tone = mix(color, light if ctx.rng.random() < 0.5 else dark, ctx.rng.random() * 0.35)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba(tone, base_alpha * (0.6 + 0.4 * ctx.rng.random())))
    elif material == "ribbons":
        draw.polygon(points, fill=rgba(color, base_alpha * 0.75))
        for k, factor in enumerate((0.8, 0.6, 0.4)):
            inner = _scaled(points, factor)
            draw.line(inner + [inner[0]], fill=rgba(light, 0.25 - 0.05 * k), width=max(1, int(ctx.line * 0.6)), joint="curve")
    elif material == "letters":
        draw.polygon(points, fill=rgba(color, 0.32 * strength))
        if ctx.font_path:
            glyph_size = max(9, int(ctx.unit * 0.028))
            count = int(min(260, _polygon_area(points) / (glyph_size * glyph_size * 1.1)))
            for index, (x, y) in enumerate(_sample_in_polygon(points, count, ctx.rng, 1.3)):
                char = ctx.glyphs[index % len(ctx.glyphs)]
                _glyph(layer, char, (x, y), axis_angle + ctx.rng.uniform(-0.25, 0.25), glyph_size,
                       rgba(mix(color, dark, 0.55), 0.7 + 0.3 * ctx.rng.random()), ctx.font_path)
        else:
            draw.polygon(points, fill=rgba(color, base_alpha * 0.8))
    elif material == "leaves":
        draw.polygon(points, fill=rgba(color, base_alpha))
        draw.polygon(_scaled(points, 0.6), fill=rgba(mix(color, light, 0.2), 0.25 * strength))
    elif material == "digital_grid":
        draw.polygon(points, fill=rgba(color, base_alpha * 0.6))
        gap = max(4, int(ctx.unit * 0.012))

        def grid(tile_draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
            for x in range((-x0) % gap, x1 - x0, gap):
                tile_draw.line((x, 0, x, y1 - y0), fill=rgba(light, 0.35), width=1)
            for y in range((-y0) % gap, y1 - y0, gap):
                tile_draw.line((0, y, x1 - x0, y), fill=rgba(light, 0.35), width=1)

        _clip_pattern(layer, points, grid)
        node = max(2, int(ctx.unit * 0.003))
        for x, y in points[:: max(1, len(points) // 10)]:
            draw.rectangle((x - node, y - node, x + node, y + node), fill=rgba(light, 0.85))
    elif material == "crystal":
        step = max(1, len(points) // 8)
        vertices = list(points[::step])
        center = _centroid(points)
        apex = (center[0] - ctx.light_dir[0] * size * 0.18, center[1] - ctx.light_dir[1] * size * 0.18)
        light_angle = math.atan2(ctx.light_dir[1], ctx.light_dir[0])
        for p, q in zip(vertices, vertices[1:] + vertices[:1]):
            mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            facing = math.cos(math.atan2(mid[1] - apex[1], mid[0] - apex[0]) - light_angle)
            shade = mix(dark, light, 0.5 + 0.45 * facing)
            draw.polygon([apex, p, q], fill=rgba(shade, base_alpha))
            draw.line((apex, p), fill=rgba(light, 0.45), width=max(1, int(ctx.line * 0.4)))
    elif material == "smoke":
        target = ctx.smoke if ctx.smoke is not None else layer
        ImageDraw.Draw(target, "RGBA").polygon(points, fill=rgba(color, base_alpha))
        draw.polygon(_scaled(points, 0.78), fill=rgba(mix(color, light, 0.15), 0.35 * strength))
    else:
        draw.polygon(points, fill=rgba(color, base_alpha))


def _feather_polygon(feather: ArtFeather, style: str) -> list[Point]:
    kinds = {"rays": "lance", "layered_curves": "leaf", "brush_strokes": "drop", "dots": "ellipse",
             "glyphs": "leaf", "leaves": "leaf", "pixels": "block", "facets": "triangle", "wisps": "drop"}
    width = feather.width * (0.55 if style == "rays" else 1.0)
    if style == "ribbons":
        points_top, points_bottom = [], []
        for i in range(12):
            s = i / 11
            curl = 0.35 * math.sin(s * math.pi) * feather.width * 2
            x = feather.length * s
            half = feather.width * 0.5 * (1 - 0.8 * s)
            points_top.append((x, curl + half))
            points_bottom.append((x, curl - half))
        local = points_top + points_bottom[::-1]
        return [rotate_point((feather.base[0] + x, feather.base[1] + y), feather.angle, feather.base) for x, y in local]
    return feather_points(feather.base, feather.angle, feather.length, width, kinds.get(style, "leaf"))


def _edge(draw: ImageDraw.ImageDraw, points: Sequence[Point], style: str, color: tuple[int, int, int], ctx: PaintContext,
          light_side_only: bool = False) -> None:
    """Контурный акцент выбранного стиля края."""
    if style in ("none", "") or len(points) < 2:
        return
    closed = list(points) + [points[0]]
    width = max(1, int(ctx.line))
    if style == "sharp":
        draw.line(closed, fill=rgba(color, 0.85), width=max(1, int(width * 0.7)), joint="curve")
    elif style == "ink_line":
        for index, (p, q) in enumerate(zip(closed, closed[1:])):
            w = max(1, int(width * (1.2 + 0.9 * math.sin(index * 0.35))))
            draw.line((p, q), fill=rgba(color, 0.9), width=w)
    elif style == "edge_highlight":
        center = _centroid(points)
        for p, q in zip(closed, closed[1:]):
            normal = _normalize(((p[0] + q[0]) / 2 - center[0], (p[1] + q[1]) / 2 - center[1]))
            facing = normal[0] * ctx.light_dir[0] + normal[1] * ctx.light_dir[1]
            if facing > 0.1:
                draw.line((p, q), fill=rgba((255, 255, 255), 0.75 * facing), width=width)
            else:
                draw.line((p, q), fill=rgba(color, 0.35), width=max(1, width // 2))
    elif style == "double":
        draw.line(closed, fill=rgba(color, 0.5), width=max(1, width // 2), joint="curve")
        outer = _scaled(points, 1.035)
        draw.line(outer + [outer[0]], fill=rgba(color, 0.3), width=max(1, width // 2), joint="curve")
    elif style == "dissolve":
        for index, (p, q) in enumerate(zip(closed, closed[1:])):
            if index % 3 != 2:
                draw.line((p, q), fill=rgba(color, 0.55), width=max(1, width // 2))
    elif style == "living_line":
        wobbly = [(x + ctx.rng.uniform(-1, 1) * width * 0.6, y + ctx.rng.uniform(-1, 1) * width * 0.6) for x, y in closed]
        draw.line(wobbly, fill=rgba(color, 0.75), width=width, joint="curve")
    elif style in ("soft", "soft_glow"):
        draw.line(closed, fill=rgba(color, 0.45), width=max(1, int(width * 0.6)), joint="curve")


def _art_background(width: int, height: int, spec: BirdSpec, light_center: Point, mode: str = "gradient") -> Image.Image:
    """Чистый фон: мягкий градиент к источнику света и едва заметная виньетка, без буквальных мотивов."""
    if mode == "solid":
        return Image.new("RGB", (width, height), spec.background_color)
    small_w, small_h = max(64, width // 4), max(64, height // 4)
    ys, xs = np.mgrid[0:small_h, 0:small_w].astype(np.float32)
    cx, cy = light_center[0] / 4, light_center[1] / 4
    distance = np.sqrt(((xs - cx) / small_w) ** 2 + ((ys - cy) / small_h) ** 2)
    t = np.clip(distance / 0.95, 0, 1) ** 1.3
    dark = rgb_lightness(spec.background_color) < 0.5
    center = mix(spec.background_color, spec.outline_color if dark else (255, 255, 255), 0.12 if dark else 0.45)
    center = mix(center, spec.primary_color, 0.06)
    edge = mix(spec.background_color, (0, 0, 0), 0.45 if dark else 0.08)
    pixels = np.array(center, np.float32) * (1 - t[..., None]) + np.array(edge, np.float32) * t[..., None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB").resize((width, height), Image.Resampling.BICUBIC)


def _blurred(layer: Image.Image, radius: float) -> Image.Image:
    """Размытие через уменьшенную копию (быстро и без лишней памяти)."""
    if radius <= 0.5:
        return layer
    factor = 4 if radius > 6 else 2
    small = layer.resize((max(1, layer.width // factor), max(1, layer.height // factor)), Image.Resampling.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(radius / factor))
    return small.resize(layer.size, Image.Resampling.BICUBIC)


def _screen(base: Image.Image, glow: Image.Image, strength: float) -> Image.Image:
    """Аддитивно выглядящее «световое» наложение (screen) с учётом альфы свечения."""
    from PIL import ImageChops
    alpha = glow.getchannel("A").point(lambda v: int(v * clamp(strength)))
    light = Image.composite(glow.convert("RGB"), Image.new("RGB", glow.size, (0, 0, 0)), alpha)
    return ImageChops.screen(base.convert("RGB"), light).convert("RGBA")


def _texture(image: Image.Image, spec: BirdSpec) -> Image.Image:
    """Детерминированная тонкая текстура (сила 0.02–0.12) и лёгкая цветокоррекция."""
    width, height = image.size
    strength = clamp(spec.texture_strength, 0.0, 0.12)
    amplitude = strength * 32
    style = spec.texture_style
    if style == "soft_film":
        noise = _noise(width, height, spec.seed, 6)
    elif style == "paper":
        noise = 0.6 * _noise(width, height, spec.seed, 1) + 0.6 * _noise(width, height, spec.seed + 1, 3)
    elif style == "digital_noise":
        generator = np.random.default_rng(spec.seed & 0xFFFFFFFF)
        block = max(2, min(width, height) // 300)
        small = generator.standard_normal((height // block + 1, width // block + 1)).astype(np.float32)
        noise = np.repeat(np.repeat(small, block, axis=0), block, axis=1)[:height, :width]
    elif style == "marble":
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        noise = np.sin(xs / width * 7 + _noise(width, height, spec.seed, 90) * 1.2) * 0.6 + 0.4 * _noise(width, height, spec.seed + 3, 1)
    else:
        noise = _noise(width, height, spec.seed, 1)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) + noise[..., None] * amplitude
    result = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    from PIL import ImageEnhance
    result = ImageEnhance.Contrast(result).enhance(1.0 + 0.08 * spec.visual_tension)
    return ImageEnhance.Color(result).enhance(1.04)


def _local_glitch(image: Image.Image, spec: BirdSpec, region: tuple[int, int, int, int]) -> Image.Image:
    """Локальный контролируемый глитч: несколько горизонтальных полос в одной области."""
    rng = random.Random(spec.seed ^ 0x611C)
    array = np.asarray(image, dtype=np.uint8).copy()
    x0, y0, x1, y1 = region
    band_h = max(2, (y1 - y0) // 14)
    for _ in range(4):
        y = rng.randint(y0, max(y0, y1 - band_h))
        shift = rng.randint(-max(2, (x1 - x0) // 12), max(2, (x1 - x0) // 12))
        array[y:y + band_h, x0:x1] = np.roll(array[y:y + band_h, x0:x1], shift, axis=1)
    return Image.fromarray(array, "RGB")


def render_art_bird(
    spec: BirdSpec,
    width: int,
    height: int,
    supersampling: int,
    background: str,
    show_keywords: bool,
    mapping_config: dict[str, Any] | None,
    font_candidates: Sequence[str] | None,
    progress: Callable[[Image.Image, str], None] | None,
) -> tuple[Image.Image, list[str], dict[str, Any]]:
    """Послойный рендер: 1 фон → 2 дальние частицы → 3 вторичный силуэт → 4 заднее крыло и хвост →
    5 корпус → 6 переднее крыло → 7 перья → 8 ядро → 9 контурные акценты → 10 ближние частицы →
    11 текстура и цветокоррекция. Затем ключевые слова."""
    warnings: list[str] = []
    safe = (mapping_config or {}).get("safe_ranges", {})
    margin = float(safe.get("canvas_margin", 0.07))
    big_w, big_h = width * supersampling, height * supersampling
    unit = min(big_w, big_h)
    rng = random.Random(spec.seed ^ 0x3C0F)
    geometry = build_art_geometry(spec, random.Random(spec.seed ^ 0x6E0))
    configs = active_configs()
    comp = configs.compositions["compositions"].get(spec.composition_type, {})
    tilt = math.radians(-10 + 40 * spec.body_tilt)
    feather_shapes = [Shape(_feather_polygon(f, spec.feather_style)) for f in geometry.feathers]
    probe_shapes = [Shape(geometry.body), Shape(geometry.beak), Shape(ellipse_points(geometry.head_center, geometry.head_radius, geometry.head_radius))]
    probe_shapes += [Shape(w) for w in geometry.wings.values()] + feather_shapes
    if geometry.ring:
        probe_shapes.append(Shape(geometry.ring))
    desired = unit * (0.56 + 0.2 * spec.body_scale) * (1.0 - 0.18 * spec.negative_space)
    transform = fit_transform(probe_shapes, tilt, big_w, big_h, desired, margin)
    center = comp.get("center", [0.5, 0.5])
    bbox_pts = [transform(p) for s in probe_shapes for p in s.points]
    min_x, max_x = min(p[0] for p in bbox_pts), max(p[0] for p in bbox_pts)
    min_y, max_y = min(p[1] for p in bbox_pts), max(p[1] for p in bbox_pts)
    dx = clamp(big_w * float(center[0]) - (min_x + max_x) / 2, big_w * margin - min_x, big_w * (1 - margin) - max_x)
    dy = clamp(big_h * float(center[1]) - (min_y + max_y) / 2, big_h * margin - min_y, big_h * (1 - margin) - max_y)
    transform.offset = (transform.offset[0] + dx, transform.offset[1] + dy)
    T = lambda pts: [transform(p) for p in pts]  # noqa: E731

    light_positions = {"core": geometry.anchors["core"], "chest": geometry.anchors["core"], "upper": None,
                       "break": geometry.anchors.get("break_zone", geometry.anchors["core"])}
    light_local = light_positions.get(str(comp.get("light_position", "core")))
    light_center = transform(light_local) if light_local else (big_w * 0.5, big_h * 0.22)
    light_dir = _normalize((-0.5, -0.8))
    font_found = find_cyrillic_font(20, font_candidates if font_candidates is not None else keyword_font_candidates())
    font_path = font_found[1] if font_found else None
    glyphs = "".join(spec.keywords).replace(" ", "") or "птица"
    if spec.material_type == "letters" and not font_path:
        warnings.append("Для материала «буквы» не найден шрифт с кириллицей: буквы заменены чернилами")
    ctx = PaintContext(spec, transform, rng, unit, max(1.0, unit * (0.002 + 0.004 * spec.line_width)), light_dir, font_path, glyphs)

    def frame(label: str, pending: Image.Image | None = None) -> None:
        if not progress:
            return
        size = (max(1, big_w // 4), max(1, big_h // 4))
        preview = canvas.resize(size, Image.Resampling.BILINEAR)
        if pending is not None:
            preview = Image.alpha_composite(preview, pending.resize(size, Image.Resampling.BILINEAR))
        progress(preview, label)

    def material_for(part: str, t: float = 0.0) -> str:
        if spec.rare_feature == "asymmetric_materials" and spec.secondary_material_type:
            return spec.secondary_material_type if part in ("near_wing",) else spec.material_type
        if spec.secondary_material_type and (part == "tail" or (part.endswith("wing") and t > 0.6)):
            return spec.secondary_material_type
        return spec.material_type

    def finish_layer(layer: Image.Image, blur: float = 0.0) -> None:
        nonlocal canvas
        if ctx.smoke is not None:
            smoke = _blurred(ctx.smoke, unit * 0.007)
            layer = Image.alpha_composite(smoke, layer)
            ctx.smoke = None
        if blur > 0:
            layer = _blurred(layer, blur)
        canvas = Image.alpha_composite(canvas, layer)

    def new_layer() -> Image.Image:
        if spec.material_type == "smoke" or spec.secondary_material_type == "smoke":
            ctx.smoke = Image.new("RGBA", (big_w, big_h), (0, 0, 0, 0))
        return Image.new("RGBA", (big_w, big_h), (0, 0, 0, 0))

    openness = spec.silhouette_openness
    periphery = mix(spec.secondary_color, spec.outline_color if rgb_lightness(spec.background_color) < 0.5 else (255, 255, 255), 0.45)

    def feather_color(feather: ArtFeather) -> tuple[int, int, int]:
        base = spec.secondary_color if feather.part != "tail" else mix(spec.secondary_color, spec.accent_color, 0.25)
        return mix(base, periphery, openness * feather.t * 0.8)

    try:
        # 1. Фон
        background_mode = "solid" if background == "solid" else "gradient"
        canvas = _art_background(big_w, big_h, spec, light_center, background_mode).convert("RGBA")
        frame("1. Фон")

        # 2. Дальние частицы (размыты — ощущение глубины)
        layer = new_layer()
        draw = ImageDraw.Draw(layer, "RGBA")
        particle_color = mix(spec.accent_color, periphery, 0.4)
        for pos, size, angle, depth in geometry.particles:
            if depth >= 0.5:
                continue
            x, y = transform(pos)
            if not (big_w * margin * 0.5 < x < big_w * (1 - margin * 0.5) and big_h * margin * 0.5 < y < big_h * (1 - margin * 0.5)):
                continue
            r = size * transform.scale * 0.8
            draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba(particle_color, 0.25 + 0.35 * depth))
        finish_layer(layer, blur=unit * 0.004)
        frame("2. Дальние частицы")

        # 3. Вторичный полупрозрачный силуэт (эхо, память) и следы движения
        layer = new_layer()
        draw = ImageDraw.Draw(layer, "RGBA")
        trails = [(dx_, dy_, a_) for dx_, dy_, a_ in geometry.ghosts]
        if spec.main_effect == "motion_trails":
            trails += [(-0.03 * k, -0.01 * k, 0.12 / k) for k in (1, 2, 3)]
        for gx, gy, alpha in trails:
            for outline in [geometry.body, *geometry.wings.values()]:
                shifted = T([(x + gx, y + gy) for x, y in outline])
                draw.polygon(shifted, fill=rgba(spec.primary_color, alpha * 0.5))
                draw.line(shifted + [shifted[0]], fill=rgba(spec.outline_color, alpha), width=max(1, int(ctx.line * 0.6)), joint="curve")
        finish_layer(layer, blur=unit * 0.003)
        frame("3. Вторичный силуэт")

        # 4. Заднее крыло и хвост
        layer = new_layer()
        far_color = mix(spec.secondary_color, spec.background_color, 0.3 if not geometry.frontal else 0.1)
        paint_region(layer, T(geometry.wings["far"]), far_color, material_for("far_wing"), ctx, 0.8)
        if geometry.frontal:  # во фронтальной композиции оба крыла раскрыты позади корпуса
            paint_region(layer, T(geometry.wings["near"]), mix(spec.secondary_color, spec.primary_color, 0.4), material_for("near_wing"), ctx, 0.9)
        behind_parts = ("far_wing", "tail", "near_wing") if geometry.frontal else ("far_wing", "tail")
        for feather in geometry.feathers:
            if feather.part in behind_parts:
                paint_region(layer, T(_feather_polygon(feather, spec.feather_style)), feather_color(feather),
                             material_for(feather.part, feather.t), ctx, 0.8 if feather.part == "far_wing" else 0.95,
                             axis_angle=-feather.angle - tilt)
        finish_layer(layer, blur=unit * 0.0012 * spec.abstraction_level)
        frame("4. Заднее крыло и хвост")

        # 5. Корпус и голова
        layer = new_layer()
        body_px = T(geometry.body)
        paint_region(layer, body_px, spec.primary_color, material_for("body"), ctx)
        head_px = T(ellipse_points(geometry.head_center, geometry.head_radius, geometry.head_radius, 0, 36))
        paint_region(layer, head_px, mix(spec.primary_color, spec.secondary_color, 0.12), material_for("body"), ctx)
        if not geometry.frontal:
            neck = [(geometry.head_center[0] - geometry.head_radius * 0.5, geometry.head_center[1] - geometry.head_radius * 0.9),
                    (geometry.head_center[0] - geometry.head_radius * 0.3, geometry.head_center[1] + geometry.head_radius * 0.7),
                    (geometry.head_center[0] + geometry.head_radius * 0.6, geometry.head_center[1] - geometry.head_radius * 0.6),
                    (max(p[0] for p in geometry.body) * 0.9, 0.0)]
            ImageDraw.Draw(layer, "RGBA").polygon(T(neck), fill=rgba(mix(spec.primary_color, spec.secondary_color, 0.06), 0.55))
        beak_color = mix(spec.accent_color, spec.outline_color, 0.25 + 0.3 * spec.beak_sharpness)
        ImageDraw.Draw(layer, "RGBA").polygon(T(geometry.beak), fill=rgba(beak_color, 0.95))
        if spec.inner_void > 0.2:
            void = _scaled(body_px, 0.18 + 0.5 * spec.inner_void)
            ImageDraw.Draw(layer).polygon(void, fill=(0, 0, 0, 0))  # настоящая пустота внутри корпуса
        finish_layer(layer)
        frame("5. Корпус")

        # 6. Переднее крыло
        layer = new_layer()
        near_material = material_for("near_wing")
        if not geometry.frontal:
            paint_region(layer, T(geometry.wings["near"]), mix(spec.secondary_color, spec.primary_color, 0.4), near_material, ctx)
        finish_layer(layer)
        frame("6. Переднее крыло")

        # 7. Перья переднего крыла
        layer = new_layer()
        near_feathers = [] if geometry.frontal else [f for f in geometry.feathers if f.part == "near_wing"]
        for index, feather in enumerate(near_feathers):
            polygon = T(_feather_polygon(feather, spec.feather_style))
            material = material_for("near_wing", feather.t)
            if spec.rare_feature == "letter_feathers" and feather.t > 0.55 and font_path:
                cx, cy = _centroid(polygon)
                _glyph(layer, glyphs[index % len(glyphs)], (cx, cy), -feather.angle - tilt,
                       max(8, int(feather.length * transform.scale * 0.5)), rgba(mix(feather_color(feather), spec.outline_color, 0.5), 0.9), font_path)
                continue
            paint_region(layer, polygon, feather_color(feather), material, ctx, 0.95, axis_angle=-feather.angle - tilt)
            if spec.accent_effect == "veins" or spec.feather_style == "leaves":
                base = transform(feather.base)
                tip = transform((feather.base[0] + math.cos(feather.angle) * feather.length, feather.base[1] + math.sin(feather.angle) * feather.length))
                ImageDraw.Draw(layer, "RGBA").line((base, tip), fill=rgba(mix(feather_color(feather), (0, 0, 0), 0.35), 0.5), width=max(1, int(ctx.line * 0.4)))
            if progress and index % max(1, len(near_feathers) // 6) == 0:
                frame("7. Перья", layer)
        finish_layer(layer)
        frame("7. Перья")

        # Основной эффект «свечение» (bloom) — от уже нарисованной птицы
        if spec.main_effect == "bloom" or spec.glow_intensity > 0.45:
            bird = canvas.copy()
            bloom = _blurred(bird, unit * 0.03)
            canvas = _screen(canvas, bloom, 0.25 + 0.35 * spec.glow_intensity)

        # 8. Внутреннее ядро (визуальный фокус, если выбран)
        layer = new_layer()
        (ccx, ccy), crx, cry = geometry.core
        core_strength = 0.35 + 0.45 * spec.focal_strength if spec.focal_point in ("luminous_core", "chest_accent") else 0.25
        core_color = mix(spec.accent_color, (255, 240, 200), 0.3)
        inverted = spec.rare_feature == "inverted_glow"
        if inverted:
            core_color = mix(spec.background_color, (0, 0, 0), 0.6)
        draw = ImageDraw.Draw(layer, "RGBA")
        for k, scale in enumerate((1.0, 0.65, 0.35)):
            points = T(ellipse_points((ccx, ccy), crx * scale, cry * scale, 0, 36))
            draw.polygon(points, fill=rgba(mix(core_color, (255, 255, 255), 0.25 * k if not inverted else 0), core_strength * (0.3 + 0.2 * k)))
        if spec.focal_point == "chest_accent":
            accent_pts = T(feather_points((ccx - crx * 0.2, ccy), math.radians(20), crx * 0.9, cry * 0.6, "leaf"))
            draw.polygon(accent_pts, fill=rgba(spec.accent_color, 0.85))
        finish_layer(layer, blur=unit * (0.012 if spec.focal_point != "chest_accent" else 0.004))
        frame("8. Внутреннее ядро")

        # 9. Контурные акценты, глаз, фокус
        layer = new_layer()
        draw = ImageDraw.Draw(layer, "RGBA")
        edge_color = spec.outline_color
        edge_parts = (body_px, head_px) if geometry.frontal else (body_px, T(geometry.wings["near"]), head_px)
        for outline in edge_parts:
            _edge(draw, outline, spec.edge_style, edge_color, ctx)
        if spec.edge_style == "soft_glow":
            glow_layer = Image.new("RGBA", (big_w, big_h), (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_layer, "RGBA")
            for outline in (body_px, T(geometry.wings["near"])):
                glow_draw.line(outline + [outline[0]], fill=rgba(mix(spec.outline_color, (255, 255, 255), 0.4), 0.6), width=max(2, int(ctx.line * 3)))
            layer = Image.alpha_composite(_blurred(glow_layer, unit * 0.006), layer)
            draw = ImageDraw.Draw(layer, "RGBA")
        eye_white = mix(spec.background_color, (255, 255, 255), 0.75)
        for (ex, ey), er in geometry.eyes:
            cx, cy = transform((ex, ey))
            r = er * transform.scale
            if spec.rare_feature == "luminous_eye" or spec.focal_point == "eye":
                glow_r = r * (2.2 + spec.focal_strength)
                draw.ellipse((cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r), fill=rgba(spec.accent_color, 0.25 + 0.2 * spec.focal_strength))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=rgba(eye_white, 0.95), outline=rgba(edge_color, 0.8), width=max(1, int(ctx.line * 0.5)))
            pr = r * 0.55
            iris = spec.accent_color if spec.focal_point == "eye" else mix(spec.outline_color, (0, 0, 0), 0.5)
            draw.ellipse((cx - pr + r * 0.1, cy - pr, cx + pr + r * 0.1, cy + pr), fill=rgba(iris, 1.0))
            hr = pr * 0.3
            draw.ellipse((cx + r * 0.25 - hr, cy - r * 0.25 - hr, cx + r * 0.25 + hr, cy - r * 0.25 + hr), fill=(255, 255, 255, 230))
        if spec.focal_point == "wing_tip":
            tip = transform(geometry.anchors["wing_tip"])
            r = unit * 0.02 * (0.6 + spec.focal_strength)
            draw.ellipse((tip[0] - r, tip[1] - r, tip[0] + r, tip[1] + r), fill=rgba(spec.accent_color, 0.75))
        if spec.focal_point == "tail":
            for feather in geometry.feathers:
                if feather.part == "tail":
                    tip = transform((feather.base[0] + math.cos(feather.angle) * feather.length, feather.base[1] + math.sin(feather.angle) * feather.length))
                    r = unit * 0.006 * (0.6 + spec.focal_strength)
                    draw.ellipse((tip[0] - r, tip[1] - r, tip[0] + r, tip[1] + r), fill=rgba(spec.accent_color, 0.9))
        _edge(draw, T(geometry.beak), "sharp", edge_color, ctx)
        finish_layer(layer)
        frame("9. Контурные акценты")

        # 10. Ближние частицы и следы
        layer = new_layer()
        draw = ImageDraw.Draw(layer, "RGBA")
        for pos, size, angle, depth in geometry.particles:
            if depth < 0.5:
                continue
            x, y = transform(pos)
            tone = mix(spec.accent_color, periphery, rng.random() * 0.6)
            if not (big_w * margin * 0.5 < x < big_w * (1 - margin * 0.5) and big_h * margin * 0.5 < y < big_h * (1 - margin * 0.5)):
                continue
            r = size * transform.scale
            if spec.focal_point == "break_zone":
                tone = mix(tone, spec.accent_color, 0.5)
            if spec.material_type in ("crystal", "digital_grid"):
                draw.rectangle((x - r, y - r, x + r, y + r), fill=rgba(tone, 0.8))
            elif spec.feather_style in ("leaves", "layered_curves") and rng.random() < 0.4:
                draw.polygon(T(feather_points(pos, angle, size * 3, size * 1.3, "leaf")), fill=rgba(tone, 0.75))
            else:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba(tone, 0.55 + 0.4 * depth))
        if geometry.ring:
            ring = T(geometry.ring)
            for index, (p, q) in enumerate(zip(ring, ring[1:])):
                if index % 4 != 3:
                    draw.line((p, q), fill=rgba(spec.accent_color, 0.55), width=max(1, int(ctx.line * 0.6)))
        finish_layer(layer)
        frame("10. Ближние частицы")

        # 11. Текстура и цветокоррекция
        image = canvas.convert("RGB")
        image = image.resize((width, height), Image.Resampling.LANCZOS) if supersampling > 1 else image
        if spec.accent_effect == "local_glitch" or spec.main_effect == "node_grid" and spec.visual_tension > 0.5:
            bx = [transform(p)[0] / supersampling for p in geometry.wings["far"]]
            by = [transform(p)[1] / supersampling for p in geometry.wings["far"]]
            region = (int(max(0, min(bx))), int(max(0, min(by))), int(min(width, max(bx))), int(min(height, (min(by) + max(by)) / 2)))
            if region[2] - region[0] > 8 and region[3] - region[1] > 8:
                image = _local_glitch(image, spec, region)
        image = _texture(image, spec)
        if progress:
            progress(image, "11. Текстура и цветокоррекция")

        font_used = None
        if show_keywords and spec.keywords and spec.material_type != "letters":
            paths_px = {}
            for name, (points, color, thickness) in geometry.text_paths.items():
                projected = [transform(point) for point in points]
                paths_px[name] = ([(x / supersampling, y / supersampling) for x, y in projected], color,
                                  thickness * transform.scale / supersampling)
            font_used = draw_keywords(image, spec, paths_px, warnings, font_candidates)
            if progress and font_used:
                progress(image, "Ключевые слова из запроса")
        elif spec.material_type == "letters":
            font_used = font_path
    except RenderingError:
        raise
    except (ValueError, MemoryError, OSError) as exc:
        raise RenderingError(f"Не удалось построить изображение: {exc}") from exc
    info = {"background_mode": background_mode, "font": font_used, "style": "", "bird_scale_px": _round(transform.scale / supersampling, 2)}
    return image, warnings, info


# ============================================================================
# 24. Объяснение маппинга
# ============================================================================


def build_mapping_explanation(
    analysis: TextAnalysis,
    bird_spec: BirdSpec,
    themes_config: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Шаблонные объяснения реально применённых правил. Не содержит исходного текста."""
    eff = effective_scores(analysis)
    theme_info = (themes_config or {}).get("themes", {})
    palette: list[str] = []
    shape: list[str] = []
    intent: list[str] = []
    motifs: list[str] = []

    if eff["valence"] > 0.58:
        palette.append("Положительная эмоциональная окраска повысила светлоту и чистоту палитры")
    elif eff["valence"] < 0.42:
        palette.append("Негативная окраска сделала палитру приглушённой, а фон — тёмным и контрастным")
    if analysis.themes:
        name = theme_info.get(analysis.themes[0], {}).get("name", analysis.themes[0])
        palette.append(f"Тема «{name}» повлияла на базовый оттенок")
    strongest = max(EMOTIONS, key=lambda e: (eff[e], -EMOTIONS.index(e)))
    if eff[strongest] >= 0.12:
        palette.append(f"Доминирующая эмоция «{EMOTION_NAMES_RU[strongest]}» определила акцентный цвет")
    if eff["tenderness"] > 0.3:
        palette.append("Нежность смягчила насыщенность до пастельных тонов")
    if eff["sadness"] > 0.3:
        palette.append("Грусть сдвинула оттенки в холодный диапазон")

    if eff["hope"] > 0.3 and bird_spec.wing_sweep > 0.6:
        shape.append("Высокая надежда направила крылья вверх")
    if eff["anxiety"] > 0.25 and bird_spec.wing_asymmetry > 0.15:
        shape.append("Тревога увеличила асимметрию и дробность элементов")
    if eff["anger"] > 0.25 and bird_spec.beak_sharpness > 0.5:
        shape.append("Гнев заострил клюв и сделал перья угловатыми")
    if eff["sadness"] > 0.25 and bird_spec.tail_drop > 0.35:
        shape.append("Грусть опустила хвост и вытянула корпус")
    if eff["tenderness"] > 0.3 and bird_spec.body_roundness > 0.55:
        shape.append("Нежность сделала формы мягкими и округлыми")
    if eff["arousal"] > 0.6:
        shape.append("Высокое возбуждение добавило наклон, диагональ и мелкие детали")
    elif eff["arousal"] < 0.42:
        shape.append("Спокойствие сделало линии плавными, а композицию — горизонтальной")
    if eff["dominance"] > 0.58:
        shape.append("Ощущение контроля приподняло голову и раскрыло крылья")
    elif eff["dominance"] < 0.42:
        shape.append("Уязвимость сделала корпус хрупким и полупрозрачным")
    if eff["surprise"] > 0.25 and bird_spec.eye_scale > 0.5:
        shape.append("Удивление увеличило глаз")
    if analysis.confidence < 0.3:
        shape.append("Низкая уверенность анализа приблизила форму к нейтральной")

    if analysis.intent_type == "release":
        intent.append("Намерение отпустить добавило отделяющиеся частицы, свободное пространство и светлую периферию")
    elif analysis.intent_type == "acquire":
        intent.append("Намерение обрести собрало силуэт вокруг плотного тёплого ядра и направило линии к центру")
    elif analysis.intent_type == "mixed":
        intent.append("Смешанное намерение соединило устойчивое ядро, раскрытые крылья и отделяющиеся частицы")
    else:
        intent.append("Направление намерения не определено, поэтому композиция нейтральна")
    if analysis.intent_type != "unclear":
        intent.append(f"Сила влияния намерения умножена на его уверенность ({analysis.intent_confidence:.2f})")

    for theme_id in analysis.themes:
        info = theme_info.get(theme_id, {})
        applied = [m for m in info.get("motifs", []) if m in bird_spec.decorative_motifs]
        if applied:
            motifs.append(f"Тема «{info.get('name', theme_id)}» добавила мотивы: {', '.join(applied)}")
    for motif in ("separated_particles", "open_space", "warm_core"):
        if motif in bird_spec.decorative_motifs and not any(motif in line for line in motifs):
            motifs.append(f"Намерение добавило мотив {motif}")
    for emotion, motif in (("anger", "fire"), ("sadness", "faded_layers"), ("tenderness", "warm_glow"), ("hope", "open_wings")):
        if motif in bird_spec.decorative_motifs and not any(motif in line for line in motifs):
            motifs.append(f"Эмоция «{EMOTION_NAMES_RU[emotion]}» добавила мотив {motif}")
    style_lines: list[str] = []
    style = STYLES.get(bird_spec.style)
    if style:
        style_lines.append(f"Стиль «{style.title}»: {style.description}")
        style_lines.append("Стиль меняет палитру, форму перьев, контур, фон и фактуру; эмоциональную геометрию "
                           "(размах, наклон, хвост, асимметрию, частицы) по-прежнему задаёт текст")
    art_lines: list[str] = []
    if bird_spec.primary_archetype:
        configs = active_configs()
        archetypes = configs.archetypes["archetypes"]
        weights = bird_spec.archetype_weights
        parts = [f"основной «{archetypes[bird_spec.primary_archetype]['name']}» ({weights.get(bird_spec.primary_archetype, 0):.0%})",
                 f"вторичный «{archetypes[bird_spec.secondary_archetype]['name']}» ({weights.get(bird_spec.secondary_archetype, 0):.0%})"]
        if bird_spec.accent_archetype:
            parts.append(f"акцентный «{archetypes[bird_spec.accent_archetype]['name']}» ({weights.get(bird_spec.accent_archetype, 0):.0%})")
        signals = art_signals(analysis)
        contributions = {k: float(c) * signals.get(k, 0.0) for k, c in archetypes[bird_spec.primary_archetype]["affinity"].items()}
        top = [k for k, v in sorted(contributions.items(), key=lambda kv: -kv[1]) if v > 0.03][:2]
        names = [SIGNAL_NAMES_RU.get(k) or ("тема «" + theme_info.get(k[6:], {}).get("name", k[6:]) + "»") for k in top]
        art_lines.append("Архетипы: " + ", ".join(parts) + (f" — основной усилен сигналами: {', '.join(names)}" if names else ""))
        composition = configs.compositions["compositions"][bird_spec.composition_type]
        art_lines.append(f"Композиция «{composition['name']}» задаёт наклон, раскрытие крыльев, хвост и зону частиц")
        materials = configs.materials["materials"]
        material_line = f"Материал «{materials[bird_spec.material_type]['name']}»"
        if bird_spec.secondary_material_type:
            material_line += f" + «{materials[bird_spec.secondary_material_type]['name']}» на периферии"
        art_lines.append(material_line + f"; силуэт «{bird_spec.silhouette_type}», крыло «{bird_spec.wing_shape}», хвост «{bird_spec.tail_type}»")
        art_lines.append(f"Палитра: стратегия «{configs.palettes['strategies'][bird_spec.palette_strategy]['name']}»; "
                         f"эффекты: {bird_spec.main_effect} + {bird_spec.accent_effect}")
        art_lines.append(f"Фокус: {FOCAL_NAMES_RU.get(bird_spec.focal_point, bird_spec.focal_point)}; абстрактность {bird_spec.abstraction_level:.2f}; "
                         f"отрицательное пространство {bird_spec.negative_space:.2f}")
        if bird_spec.rare_feature:
            feature = configs.archetypes["rare_features"]["features"][bird_spec.rare_feature]["name"]
            art_lines.append(f"Редкая особенность: {feature}")
    return {"palette": palette[:5], "shape": shape[:6], "intent": intent[:3], "motifs": motifs[:6],
            "style": style_lines, "art": art_lines}


# ============================================================================
# 25. Сохранение PNG
# ============================================================================


def _atomic_write(target: Path, write: Callable[[Path], None], verify: Callable[[Path], None]) -> None:
    """Записать во временный файл той же директории, проверить и атомарно заменить."""
    target = Path(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputWriteError(f"Не удалось создать директорию {target.parent}: {exc.strerror}") from exc
    if target.exists() and target.is_dir():
        raise OutputWriteError(f"Путь вывода является директорией: {target}")
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    except OSError as exc:
        raise OutputWriteError(f"Не удалось создать временный файл в {target.parent}: {exc.strerror}") from exc
    os.close(fd)
    temporary = Path(temp_name)
    try:
        write(temporary)
        verify(temporary)
        os.replace(temporary, target)
    except OutputWriteError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise OutputWriteError(f"Не удалось сохранить {target}: {exc}") from exc


def save_png(image: Image.Image, path: Path) -> None:
    """Сохранить PNG атомарно, без метаданных, и проверить его."""
    expected = image.size

    def write(temporary: Path) -> None:
        image.save(temporary, format="PNG")

    def verify(temporary: Path) -> None:
        with Image.open(temporary) as check:
            check.verify()
        with Image.open(temporary) as check:
            if check.size != expected:
                raise OutputWriteError(f"Размер сохранённого PNG {check.size} не совпадает с {expected}")
            if check.mode not in ("RGB", "RGBA"):
                raise OutputWriteError(f"Неожиданный режим PNG: {check.mode}")

    _atomic_write(Path(path), write, verify)


# ============================================================================
# 26. Сохранение JSON
# ============================================================================


def save_json(document: dict[str, Any], path: Path) -> None:
    """Сохранить JSON атомарно (ensure_ascii=False) и проверить повторным чтением."""

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def verify(temporary: Path) -> None:
        with temporary.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if loaded.get("schema_version") != SCHEMA_VERSION:
            raise OutputWriteError("Проверка JSON не пройдена: неверная schema_version")

    _atomic_write(Path(path), write, verify)


# ============================================================================
# 27. Логирование
# ============================================================================


def configure_logging(debug: bool) -> None:
    """Настроить логирование в stderr. Исходные тексты никогда не логируются."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s: %(message)s" if debug else "%(message)s",
        stream=sys.stderr,
        force=True,
    )


def log_report(result: GenerationResult) -> None:
    """Краткий технический отчёт без исходных комментариев."""
    analysis = result.analysis
    prepared = result.prepared
    LOGGER.info("Комментариев: исходных %d, уникальных %d, дублей %d, проанализировано %d",
                prepared.source_comment_count, prepared.unique_comment_count,
                prepared.duplicate_comment_count, prepared.analyzed_comment_count)
    LOGGER.info("Язык: %s; намерение: %s (уверенность %.2f, release_strength %.2f)",
                analysis.language, analysis.intent_type, analysis.intent_confidence, analysis.release_strength)
    top = sorted(analysis.emotions.items(), key=lambda kv: -kv[1])[:3]
    LOGGER.info("Основные эмоции: %s", ", ".join(f"{k}={v:.2f}" for k, v in top))
    LOGGER.info("valence=%.2f arousal=%.2f intensity=%.2f dominance=%.2f; уверенность %.2f",
                analysis.valence, analysis.arousal, analysis.intensity, analysis.dominance, analysis.confidence)
    LOGGER.info("Ключевые слова: %s", ", ".join(analysis.keywords) or "—")
    LOGGER.info("Темы: %s", ", ".join(analysis.themes) or "—")
    LOGGER.info("Seed: %d (%s)", result.seed, result.seed_source)
    render = result.document.get("render", {})
    spec = result.bird_spec
    if spec.primary_archetype:
        weights = ", ".join(f"{k} {v:.0%}" for k, v in spec.archetype_weights.items())
        LOGGER.info("Архетипы: %s; композиция: %s; материал: %s%s", weights, spec.composition_type, spec.material_type,
                    f" + {spec.secondary_material_type}" if spec.secondary_material_type else "")
        LOGGER.info("Силуэт: %s; палитра: %s; фокус: %s; абстрактность %.2f%s", spec.silhouette_type, spec.palette_strategy,
                    spec.focal_point, spec.abstraction_level, f"; редкая особенность: {spec.rare_feature}" if spec.rare_feature else "")
        selection = result.document.get("art_selection", {})
        if selection:
            LOGGER.info("Отбор: кандидат %d из %d, оценка %.3f", selection["winner_index"] + 1, selection["candidates"], selection["winner_score"])
    if render.get("style"):
        how = {"random": "выбран случайно", "seed": "выбран по seed", "cli": "задан явно"}.get(render.get("style_source"), "")
        LOGGER.info("Стиль: %s [%s] (%s)", render.get("style_title"), render.get("style"), how)
    LOGGER.info("PNG: %s", result.image_path)
    LOGGER.info("JSON: %s", result.analysis_path)
    for warning in result.warnings:
        LOGGER.warning("Предупреждение: %s", warning)
    diagnostics = result.diagnostics
    LOGGER.debug("Эмоциональных слов: %d; фраз: %d; усилителей: %d; ослабителей: %d; отрицаний: %d",
                 diagnostics.matched_word_count, diagnostics.matched_phrase_count, diagnostics.intensifier_count,
                 diagnostics.diminisher_count, diagnostics.negation_count)
    LOGGER.debug("Пунктуационных сигналов: %d; заглавных: %d; эмодзи: %d; доля распознанных слов: %.2f",
                  diagnostics.punctuation_signal_count, diagnostics.uppercase_signal_count,
                  diagnostics.emoji_signal_count, diagnostics.recognized_token_ratio)
    for param, sources in diagnostics.parameter_contributions.items():
        LOGGER.debug("Вклад в %s: %s", param, ", ".join(f"{s}={v:+.2f}" for s, v in sources.items()))
    LOGGER.debug("Уверенность по параметрам: %s", analysis.score_confidence)
    LOGGER.debug("Выбранные мотивы: %s", ", ".join(result.bird_spec.decorative_motifs) or "—")


# ============================================================================
# 28. Генерация (публичный API) и CLI
# ============================================================================


def _check_art_overrides(overrides: dict[str, str], configs: ConfigBundle) -> None:
    tables = {
        "archetype": configs.archetypes["archetypes"], "composition": configs.compositions["compositions"],
        "material": configs.materials["materials"], "palette": configs.palettes["strategies"],
    }
    for key, value in overrides.items():
        if value not in tables[key]:
            raise InputDataError(f"Неизвестное значение --{key} '{value}'. Доступны: {', '.join(tables[key])}")


def load_recent_specs(path: Path) -> list[dict[str, Any]]:
    """Прочитать сводки последних птиц (без текстов) для флага --avoid-recent-similarity."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [item for item in data.get("recent", []) if isinstance(item, dict)][-20:]
    except (OSError, ValueError, AttributeError):
        return []


def save_recent_specs(path: Path, spec: BirdSpec) -> None:
    """Дописать сводку выбранной птицы (архетип, композиция, материал…, без текста)."""
    recent = load_recent_specs(path) + [art_summary(spec)]
    save_json({"schema_version": SCHEMA_VERSION, "recent": recent[-20:]}, Path(path))


def resolve_output_paths(
    image_path: Path | None,
    analysis_path: Path | None,
    seed: int,
    overwrite: bool = False,
    moment: datetime | None = None,
    style: str = "",
) -> tuple[Path, Path]:
    """Подобрать пути PNG и JSON так, чтобы не перезаписывать прежних птиц.

    Без явных путей имя строится из локальных даты, времени и seed:
    ``output/images/bird_20260929-153012_a1b2c3.png`` и JSON с тем же именем.
    Если файл уже существует (и не задан ``overwrite``), к обоим именам
    добавляется суффикс ``_2``, ``_3``… Время влияет только на имя файла,
    но не на изображение.
    """
    moment = moment or datetime.now(timezone.utc)
    if image_path is None:
        stamp = moment.astimezone().strftime("%Y%m%d-%H%M%S")
        suffix = f"_{style}" if style else ""
        image_path = DEFAULT_IMAGE_DIR / f"bird_{stamp}_{seed % 0x1000000:06x}{suffix}.png"
    image_path = Path(image_path)
    analysis_path = Path(analysis_path) if analysis_path is not None else DEFAULT_ANALYSIS_DIR / f"{image_path.stem}.json"
    if image_path.resolve() == analysis_path.resolve():
        raise InputDataError("Пути --output и --analysis-output должны различаться")
    if overwrite or (not image_path.exists() and not analysis_path.exists()):
        return image_path, analysis_path
    for number in range(2, 100000):
        candidate_image = image_path.with_name(f"{image_path.stem}_{number}{image_path.suffix}")
        candidate_analysis = analysis_path.with_name(f"{analysis_path.stem}_{number}{analysis_path.suffix}")
        if not candidate_image.exists() and not candidate_analysis.exists():
            return candidate_image, candidate_analysis
    raise OutputWriteError(f"Не удалось подобрать свободное имя файла для {image_path}")


def append_history(history_path: Path, result: "GenerationResult") -> int:
    """Дописать нормализованные комментарии запроса в CSV-историю (накопительно).

    Файл содержит исходные тексты, поэтому лежит в data/history/ (исключено
    из Git). Колонка ``comment`` позволяет подать историю снова на вход:
    ``--input data/history/requests.csv``. Возвращает число добавленных строк.
    """
    history_path = Path(history_path)
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not history_path.exists() or history_path.stat().st_size == 0
        with history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(HISTORY_FIELDS))
            if is_new:
                writer.writeheader()
            for comment in result.prepared.comments:
                writer.writerow({
                    "comment": comment,
                    "generated_at": result.document["generated_at"],
                    "image": str(result.image_path),
                    "analysis": str(result.analysis_path),
                    "intent_type": result.analysis.intent_type,
                    "keywords": ", ".join(result.analysis.keywords[:5]),
                })
    except OSError as exc:
        raise OutputWriteError(f"Не удалось дописать историю запросов {history_path}: {exc.strerror}") from exc
    return len(result.prepared.comments)


def open_with_system_viewer(path: Path) -> bool:
    """Открыть файл в стандартной программе просмотра ОС (фиксированная команда, без shell)."""
    import subprocess

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        return True
    except OSError:
        return False


class DrawingRecorder:
    """Собирает уменьшенные кадры во время рисования для окна предпросмотра."""

    def __init__(self, max_side: int = 720) -> None:
        self.max_side = max_side
        self.frames: list[tuple[Image.Image, str]] = []

    def __call__(self, image: Image.Image, label: str) -> None:
        scale = self.max_side / max(image.size)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        frame = image.resize(size, Image.Resampling.BILINEAR, reducing_gap=2.0).convert("RGB")
        self.frames.append((frame, label))


def show_drawing_preview(frames: Sequence[tuple[Image.Image, str]], final_path: Path, frame_ms: int = 55) -> bool:
    """Показать окно, в котором птица появляется шаг за шагом, затем итоговый PNG.

    Кадры записаны во время настоящего рендера. Сам рендер длится меньше
    секунды, поэтому воспроизведение замедлено, чтобы процесс был виден.
    Окно закрывается пользователем. Возвращает False, если окно открыть нельзя.
    """
    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError:
        return False
    try:
        root = tk.Tk()
    except tk.TclError:
        return False
    root.title(f"Птица рисуется — {Path(final_path).name}")
    root.configure(background="#1e1e1e")
    with Image.open(final_path) as saved:
        final = saved.convert("RGB")
    side = max(frames[0][0].size) if frames else 720
    final.thumbnail((side, side), Image.Resampling.LANCZOS)
    sequence = list(frames) + [(final, f"Готово: {Path(final_path).name}. Закройте окно, чтобы продолжить.")]
    picture = tk.Label(root, background="#1e1e1e")
    picture.pack(padx=12, pady=(12, 6))
    caption = tk.Label(root, font=("Helvetica", 14), foreground="#f0f0f0", background="#1e1e1e")
    caption.pack(padx=12, pady=(0, 12))
    photos: list[Any] = []

    def show(index: int) -> None:
        frame, label = sequence[index]
        photo = ImageTk.PhotoImage(frame, master=root)
        photos.append(photo)
        picture.configure(image=photo)
        caption.configure(text=label if index == len(sequence) - 1 else f"{label}…  ({index + 1}/{len(sequence) - 1})")
        if index + 1 < len(sequence):
            stage_changed = sequence[index + 1][1] != label
            root.after(frame_ms * (6 if stage_changed else 1), show, index + 1)

    root.after(200, show, 0)
    root.lift()
    root.attributes("-topmost", True)
    root.after(600, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return True


def run_generation(raw_comments: Sequence[str], options: GenerationOptions, configs: ConfigBundle | None = None) -> GenerationResult:
    """Полный цикл: подготовка → анализ → BirdSpec → PNG → JSON."""
    configs = configs or load_configs()
    prepared = prepare_comments(raw_comments, options.max_comments, options.max_characters)
    warnings = list(prepared.warnings)
    analysis, diagnostics, analysis_warnings = analyze_comments(prepared, configs)
    warnings.extend(analysis_warnings)

    if options.seed is not None:
        if options.seed < 0:
            raise InputDataError("seed должен быть неотрицательным целым числом")
        seed, seed_source = int(options.seed), "cli"
    else:
        seed, seed_source = compute_seed(prepared.comments), "sha256"

    theme_hues = {tid: list(t["hue_range"]) for tid, t in configs.themes["themes"].items()}
    spec = analysis_to_bird_spec(analysis, seed, configs.mapping, theme_hues)
    style_name, style_source = choose_style(options.style, seed)
    art_report: dict[str, Any] | None = None
    if style_name:
        spec = apply_style_to_spec(spec, style_name, configs.mapping["limits"])
        name_suffix = style_name
    else:
        overrides = {k: v for k, v in options.art_overrides.items() if v}
        _check_art_overrides(overrides, configs)
        recent = load_recent_specs(options.recent_path) if options.avoid_recent_similarity else []
        spec, art_report = direct_art(analysis, spec, seed, configs, options.candidates, overrides, recent)
        warnings.extend(f"Художественная проверка: {p}" for p in validate_bird_spec(spec))
        name_suffix = f"{spec.primary_archetype}-{spec.composition_type}"
    generated_at = datetime.now(timezone.utc)
    image_path, analysis_path = resolve_output_paths(
        options.output_path, options.analysis_path, seed, options.overwrite, generated_at, name_suffix,
    )
    image, render_warnings, render_info = render_bird(
        spec, options.width, options.height, options.supersampling, options.background,
        options.show_keywords, configs.mapping,
        font_candidates=keyword_font_candidates(options.keyword_font) if options.keyword_font else None,
        progress=options.progress,
    )
    warnings.extend(render_warnings)
    if options.keyword_font and options.show_keywords and render_info["font"] != str(options.keyword_font):
        warnings.append("Указанный --keyword-font не найден или не поддерживает кириллицу: использован запасной шрифт")
    save_png(image, image_path)

    explanation = build_mapping_explanation(analysis, spec, configs.themes)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_parameters": {
            "seed": seed,
            "seed_source": seed_source,
            "width": options.width,
            "height": options.height,
            "supersampling": options.supersampling,
            "background": options.background,
            "style_requested": options.style,
            "show_keywords": options.show_keywords,
            "include_source_text": options.include_source_text,
            "max_comments": options.max_comments,
            "max_characters": options.max_characters,
            "input": options.input_description,
        },
        "input_statistics": {
            "source_comment_count": prepared.source_comment_count,
            "unique_comment_count": prepared.unique_comment_count,
            "duplicate_comment_count": prepared.duplicate_comment_count,
            "analyzed_comment_count": prepared.analyzed_comment_count,
            "word_count": analysis.word_count,
            "truncated": prepared.truncated,
        },
        "truncated": prepared.truncated,
        "analysis": asdict(analysis),
        "bird_spec": asdict(spec),
        "mapping_explanation": explanation,
        "render": {
            "width": options.width,
            "height": options.height,
            "supersampling": options.supersampling,
            "background_mode": render_info["background_mode"],
            "style": style_name,
            "style_title": STYLES[style_name].title if style_name else "",
            "style_source": style_source,
            "keywords_font": Path(render_info["font"]).name if render_info["font"] else None,
            "output": str(image_path),
        },
        "paths": {"image": str(image_path), "analysis": str(analysis_path)},
        "config_versions": dict(configs.versions),
        "config_hashes": dict(configs.hashes),
        "warnings": warnings,
    }
    if art_report is not None:
        document["art_selection"] = art_report
        if options.avoid_recent_similarity:
            save_recent_specs(options.recent_path, spec)
    if options.include_source_text:
        document["source_comments"] = list(prepared.comments)
    if options.debug:
        document["diagnostics"] = asdict(diagnostics)
    save_json(document, analysis_path)
    return GenerationResult(
        prepared=prepared, analysis=analysis, diagnostics=diagnostics, bird_spec=spec, seed=seed,
        seed_source=seed_source, image_path=image_path, analysis_path=analysis_path,
        warnings=warnings, document=document,
    )


def generate_all_styles(
    raw_comments: Sequence[str],
    options: GenerationOptions,
    configs: ConfigBundle | None = None,
    collage_path: Path | None = None,
    tile: int = 360,
) -> tuple[list[GenerationResult], Path]:
    """Нарисовать птицу по одному запросу во всех 10 стилях и собрать коллаж.

    Seed у всех птиц общий, поэтому форма птицы одинакова и различается
    только стиль. Каждая птица сохраняется отдельным файлом (с именем стиля),
    коллаж — в ``output/collages/styles_<дата-время>_<seed>.png``.
    """
    import bird_library  # локальный импорт: bird_library сам импортирует этот модуль

    configs = configs or load_configs()
    results: list[GenerationResult] = []
    for style_name in STYLE_NAMES:
        styled = dataclasses.replace(options, style=style_name, progress=None)
        if options.output_path is not None:
            base = Path(options.output_path)
            styled.output_path = base.with_name(f"{base.stem}_{style_name}{base.suffix or '.png'}")
        if options.analysis_path is not None:
            base = Path(options.analysis_path)
            styled.analysis_path = base.with_name(f"{base.stem}_{style_name}{base.suffix or '.json'}")
        results.append(run_generation(raw_comments, styled, configs))
    records = []
    for result in results:
        moment = datetime.fromisoformat(result.document["generated_at"].replace("Z", "+00:00"))
        analysis = asdict(result.analysis)
        records.append(bird_library.BirdRecord(
            image_path=result.image_path, analysis_path=result.analysis_path, generated_at=moment,
            dominant_emotion=bird_library.dominant_emotion(analysis), intent_type=result.analysis.intent_type,
            keywords=list(result.analysis.keywords[:3]), style=result.bird_spec.style,
        ))
    keywords = ", ".join(results[0].analysis.keywords[:3]) or "без ключевых слов"
    collage = bird_library.build_collage(records, tile=tile, columns=5, title=f"Одна птица — 10 стилей ({keywords})")
    if collage_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        collage_path = bird_library.unique_path(
            bird_library.DEFAULT_COLLAGE_DIR / f"styles_{stamp}_{results[0].seed % 0x1000000:06x}.png")
    save_png(collage, collage_path)
    return results, Path(collage_path)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидается целое число, получено '{value}'") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("значение должно быть положительным целым числом")
    return number


def _max_characters(value: str) -> int:
    number = _positive_int(value)
    if number < MIN_TOTAL_CHARACTERS:
        raise argparse.ArgumentTypeError(f"значение должно быть не меньше {MIN_TOTAL_CHARACTERS}")
    return number


def _dimension(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидается целое число, получено '{value}'") from exc
    if not MIN_CANVAS <= number <= MAX_CANVAS:
        raise argparse.ArgumentTypeError(f"размер должен быть от {MIN_CANVAS} до {MAX_CANVAS}")
    return number


def _seed(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"seed должен быть целым числом, получено '{value}'") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("seed должен быть неотрицательным")
    return number


def _supersampling(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидается 1, 2 или 3, получено '{value}'") from exc
    if number not in (1, 2, 3):
        raise argparse.ArgumentTypeError("допустимые значения: 1, 2, 3")
    return number


def _candidates(value: str) -> int:
    number = _positive_int(value)
    if not 3 <= number <= 12:
        raise argparse.ArgumentTypeError("число кандидатов должно быть от 3 до 12")
    return number


def _style(value: str) -> str:
    key = value.strip().lower()
    if key in STYLE_ALIASES or key in ("random", "seed", "none"):
        return STYLE_ALIASES.get(key, key)
    raise argparse.ArgumentTypeError(f"неизвестный стиль '{value}'; доступны: random, seed, {', '.join(STYLE_NAMES)}")


def build_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="bird_generator.py",
        description="Генерирует абстрактную птицу по комментариям о желаниях и о том, что хочется отпустить.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Текст комментария прямо из командной строки")
    source.add_argument("--input", type=Path, help="Путь к TXT, CSV или JSON с комментариями")
    parser.add_argument("--column", default="comment", help="Колонка CSV с текстом (по умолчанию comment)")
    parser.add_argument("--text-field", default="comment", help="Поле JSON с текстом (по умолчанию comment)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Путь к PNG (по умолчанию новый файл output/images/bird_<дата-время>_<seed>.png)")
    parser.add_argument("--analysis-output", type=Path, default=None,
                        help="Путь к JSON (по умолчанию output/analysis/<имя PNG>.json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Разрешить перезапись существующих файлов (по умолчанию к имени добавляется _2, _3…)")
    parser.add_argument("--width", type=_dimension, default=1200, help="Ширина 512–4096 (по умолчанию 1200)")
    parser.add_argument("--height", type=_dimension, default=1200, help="Высота 512–4096 (по умолчанию 1200)")
    parser.add_argument("--seed", type=_seed, default=None, help="Неотрицательный seed; по умолчанию SHA-256 от текста")
    keywords = parser.add_mutually_exclusive_group()
    keywords.add_argument("--show-keywords", dest="show_keywords", action="store_true", default=True,
                          help="Нарисовать на птице до трёх ключевых слов запроса (включено по умолчанию)")
    keywords.add_argument("--hide-keywords", dest="show_keywords", action="store_false",
                          help="Не рисовать ключевые слова на изображении")
    parser.add_argument("--keyword-font", type=Path, default=None,
                        help="Свой файл шрифта (.ttf/.otf/.ttc, для .ttc можно 'путь#индекс') для слов внутри птицы; "
                             "по умолчанию каллиграфический Trattatello или другой найденный")
    parser.add_argument("--include-source-text", action="store_true", help="Сохранить нормализованные комментарии в JSON")
    parser.add_argument("--max-comments", type=_positive_int, default=1000, help="Максимум комментариев (по умолчанию 1000)")
    parser.add_argument("--max-characters", type=_max_characters, default=20000, help="Максимум символов, не меньше 10 (по умолчанию 20000)")
    parser.add_argument("--supersampling", type=_supersampling, default=2, help="Сглаживание: 1, 2 или 3 (по умолчанию 2)")
    parser.add_argument("--background", choices=BACKGROUND_MODES, default="auto",
                        help="Фон: auto (как задумано стилем), solid, gradient")
    parser.add_argument("--style", type=_style, default="none",
                        help="Архитектурный стиль поверх птицы: none (по умолчанию — художественная система архетипов), "
                             "random (случайный), seed (по seed) или одно из: "
                             + ", ".join(STYLE_NAMES) + "; можно по-русски: модерн, ар-деко, баухаус, готика, "
                             "конструктивизм, укиё-э, мозаика, минимализм, барокко, брутализм")
    parser.add_argument("--list-styles", action="store_true", help="Показать список стилей и выйти")
    parser.add_argument("--gui", action="store_true", help="Открыть окно «Отпусти» для ввода текста")
    parser.add_argument("--candidates", type=_candidates, default=None,
                        help="Сколько кандидатов BirdSpec создать и оценить (3–12, по умолчанию из конфигурации — 5)")
    parser.add_argument("--archetype", default=None, help="Задать основной архетип (light, crystal, typographic, flow, particles, organic, digital, memory)")
    parser.add_argument("--composition", default=None, help="Задать композицию (iconic_front, ascending_diagonal, quiet_profile, "
                        "open_spiral, fragmented_escape, floating_horizontal, compressed_core, dual_direction)")
    parser.add_argument("--material", default=None, help="Задать основной материал (light, glass, ink, particles, ribbons, "
                        "letters, leaves, digital_grid, crystal, smoke)")
    parser.add_argument("--palette", default=None, help="Задать палитровую стратегию (analogous, complementary_accent, "
                        "monochrome, dark_field_light_accent, warm_cool_contrast)")
    parser.add_argument("--avoid-recent-similarity", action="store_true",
                        help="Избегать сходства с последними птицами (хранит сводки без текстов в output/recent_specs.json)")
    parser.add_argument("--all-styles", action="store_true",
                        help="Нарисовать птицу во всех 10 стилях и собрать из них коллаж (output/collages/)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Не открывать окно, в котором видно, как рисуется птица")
    parser.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY_PATH,
                        help="CSV, куда накопительно дописываются тексты запросов (по умолчанию data/history/requests.csv)")
    parser.add_argument("--no-history", action="store_true", help="Не записывать текст запроса в историю")
    parser.add_argument("--debug", action="store_true", help="Подробный лог и traceback при ошибках")
    return parser


def load_input(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    """Получить комментарии из --text, --input или data/input/comments.csv."""
    if args.text is not None:
        return [args.text], {"type": "cli_text"}
    if args.input is not None:
        path = Path(args.input)
        comments = read_comments(path, args.column, args.text_field)
        return comments, {"type": path.suffix.lower().lstrip(".") or "unknown", "path": str(path),
                          "column": args.column, "text_field": args.text_field}
    if DEFAULT_INPUT_PATH.is_file():
        comments = read_comments(DEFAULT_INPUT_PATH, args.column, args.text_field)
        return comments, {"type": "csv", "path": str(DEFAULT_INPUT_PATH), "column": args.column, "default_source": True}
    raise InputDataError(NO_SOURCE_MESSAGE)


# ============================================================================
# 29. main()
# ============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код завершения."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_CLI
    configure_logging(args.debug)
    if args.gui:
        import bird_app  # локальный импорт: окно нужно только в этом режиме

        return bird_app.main()
    if args.list_styles:
        for name, profile in STYLES.items():
            print(f"{name:<15} {profile.title:<28} {profile.description}")
        return EXIT_OK
    try:
        raw_comments, input_description = load_input(args)
        configs = load_configs()
        options = GenerationOptions(
            output_path=args.output,
            analysis_path=args.analysis_output,
            width=args.width,
            height=args.height,
            seed=args.seed,
            show_keywords=args.show_keywords,
            include_source_text=args.include_source_text,
            max_comments=args.max_comments,
            max_characters=args.max_characters,
            supersampling=args.supersampling,
            background=args.background,
            debug=args.debug,
            input_description=input_description,
            overwrite=args.overwrite,
            keyword_font=args.keyword_font,
            style=args.style,
            candidates=args.candidates,
            art_overrides={"archetype": args.archetype, "composition": args.composition,
                           "material": args.material, "palette": args.palette},
            avoid_recent_similarity=args.avoid_recent_similarity,
        )
        if args.all_styles:
            LOGGER.info("Рисую птицу во всех %d стилях…", len(STYLE_NAMES))
            results, collage_path = generate_all_styles(raw_comments, options, configs)
            first = results[0]
            LOGGER.info("Комментариев проанализировано: %d; язык: %s; намерение: %s",
                        first.analysis.analyzed_comment_count, first.analysis.language, first.analysis.intent_type)
            LOGGER.info("Ключевые слова: %s; seed: %d (общий для всех стилей)",
                        ", ".join(first.analysis.keywords) or "—", first.seed)
            for result in results:
                LOGGER.info("  %-28s %s", STYLES[result.bird_spec.style].title, result.image_path)
            for warning in dict.fromkeys(w for r in results for w in r.warnings):
                LOGGER.warning("Предупреждение: %s", warning)
            LOGGER.info("Коллаж: %s", collage_path)
            if not args.no_history:
                added = append_history(args.history_file, first)
                LOGGER.info("История запросов: %s (добавлено строк: %d)", args.history_file, added)
            if not args.no_preview:
                open_with_system_viewer(collage_path)
            return EXIT_OK
        recorder = None if args.no_preview else DrawingRecorder()
        options.progress = recorder
        result = run_generation(raw_comments, options, configs)
        log_report(result)
        if not args.no_history:
            added = append_history(args.history_file, result)
            LOGGER.info("История запросов: %s (добавлено строк: %d)", args.history_file, added)
        if recorder is not None:
            LOGGER.info("Открываю окно с процессом рисования (закройте его, чтобы завершить)…")
            if not show_drawing_preview(recorder.frames, result.image_path):
                LOGGER.warning("Окно предпросмотра недоступно, открываю PNG в системном просмотрщике")
                open_with_system_viewer(result.image_path)
        return EXIT_OK
    except BirdGeneratorError as exc:
        LOGGER.error("Ошибка (%s): %s", type(exc).__name__, exc)
        if args.debug:
            LOGGER.error("%s", traceback.format_exc())
        return exc.exit_code
    except KeyboardInterrupt:
        LOGGER.error("Прервано пользователем")
        return EXIT_OTHER
    except Exception as exc:  # noqa: BLE001 — последняя линия защиты CLI
        LOGGER.error("Непредвиденная ошибка (%s). Запустите с --debug для подробностей.", type(exc).__name__)
        if args.debug:
            LOGGER.error("%s", traceback.format_exc())
        return EXIT_OTHER


if __name__ == "__main__":
    raise SystemExit(main())
