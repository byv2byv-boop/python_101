#!/usr/bin/env python3
"""Библиотека сгенерированных птиц: список и коллаж с фильтрами.

Читает JSON-отчёты из ``output/analysis/`` (включая вложенные папки, например
галерею), находит соответствующие PNG и собирает из них одну картинку‑коллаж.
Фильтры: по дате генерации, по доминирующей эмоции и по намерению.

Примеры::

    python bird_library.py                       # все птицы
    python bird_library.py --date 2026-09-29     # птицы за день
    python bird_library.py --emotion надежда     # доминирует надежда (или hope)
    python bird_library.py --intent release      # только «отпускание»
    python bird_library.py --list                # только таблица в консоли

Коллаж сохраняется в ``output/collages/`` (новый файл на каждый запуск)
и открывается в системном просмотрщике. Исходные тексты не используются.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

import bird_generator as bg

DEFAULT_ANALYSIS_DIR = Path("output/analysis")
DEFAULT_COLLAGE_DIR = Path("output/collages")
EMOTION_ALIASES: dict[str, str] = {**{e: e for e in bg.EMOTIONS}, **{v: k for k, v in bg.EMOTION_NAMES_RU.items()}}
INTENT_NAMES_RU = {"acquire": "обрести", "release": "отпустить", "mixed": "смешанное", "unclear": "не определено"}


@dataclass
class BirdRecord:
    """Одна сохранённая птица (без исходного текста)."""

    image_path: Path
    analysis_path: Path
    generated_at: datetime
    dominant_emotion: str
    intent_type: str
    keywords: list[str]
    style: str = ""
    archetype: str = ""
    composition: str = ""


def dominant_emotion(analysis: dict) -> str:
    """Доминирующая эмоция с учётом уверенности: effective = confidence × value."""
    emotions = analysis.get("emotions", {})
    confidence = analysis.get("score_confidence", {})
    scores = {e: float(emotions.get(e, 0.0)) * float(confidence.get(e, 1.0)) for e in bg.EMOTIONS}
    return max(bg.EMOTIONS, key=lambda e: (scores[e], -bg.EMOTIONS.index(e)))


def _resolve_image(raw: str, analysis_path: Path, base_dir: Path) -> Path | None:
    candidate = Path(raw)
    for option in (candidate, base_dir / candidate, analysis_path.parent / candidate.name):
        if option.is_file():
            return option
    return None


def load_records(analysis_dir: Path = DEFAULT_ANALYSIS_DIR, base_dir: Path = Path(".")) -> tuple[list[BirdRecord], int]:
    """Прочитать все JSON-отчёты. Возвращает (записи по времени, число пропущенных файлов)."""
    records: list[BirdRecord] = []
    skipped = 0
    for path in sorted(Path(analysis_dir).rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema_version") != bg.SCHEMA_VERSION:
                raise ValueError("schema")
            raw_image = data.get("paths", {}).get("image") or data.get("render", {}).get("output")
            image = _resolve_image(str(raw_image), path, base_dir) if raw_image else None
            if image is None:
                raise FileNotFoundError(raw_image)
            moment = datetime.fromisoformat(str(data["generated_at"]).replace("Z", "+00:00"))
            analysis = data["analysis"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            skipped += 1
            continue
        records.append(BirdRecord(
            image_path=image,
            analysis_path=path,
            generated_at=moment,
            dominant_emotion=dominant_emotion(analysis),
            intent_type=str(analysis.get("intent_type", "unclear")),
            keywords=list(analysis.get("keywords", []))[:3],
            style=str(data.get("bird_spec", {}).get("style", "") or ""),
            archetype=str(data.get("bird_spec", {}).get("primary_archetype", "") or ""),
            composition=str(data.get("bird_spec", {}).get("composition_type", "") or ""),
        ))
    records.sort(key=lambda r: (r.generated_at, str(r.image_path)))
    return records, skipped


def filter_records(records: Sequence[BirdRecord], day: date | None = None, emotion: str | None = None,
                   intent: str | None = None, style: str | None = None, archetype: str | None = None) -> list[BirdRecord]:
    """Отфильтровать по локальной дате, доминирующей эмоции, намерению и стилю."""
    result = []
    for record in records:
        if day is not None and record.generated_at.astimezone().date() != day:
            continue
        if emotion is not None and record.dominant_emotion != emotion:
            continue
        if intent is not None and record.intent_type != intent:
            continue
        if style is not None and record.style != style:
            continue
        if archetype is not None and record.archetype != archetype:
            continue
        result.append(record)
    return result


def build_collage(records: Sequence[BirdRecord], tile: int = 360, columns: int | None = None,
                  title: str = "") -> Image.Image:
    """Собрать коллаж: миниатюры птиц с подписями (дата, эмоция, ключевые слова)."""
    if not records:
        raise ValueError("Нет птиц для коллажа")
    columns = columns or max(1, min(6, math.ceil(math.sqrt(len(records)))))
    rows = math.ceil(len(records) / columns)
    found = bg.find_cyrillic_font(max(12, tile // 22))
    title_font = bg.find_cyrillic_font(max(14, tile // 14))
    caption_h = tile // 5 if found else 0
    title_h = tile // 6 if title and title_font else 0
    gap = max(6, tile // 30)
    width = columns * tile + (columns + 1) * gap
    height = title_h + rows * (tile + caption_h) + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), (30, 30, 34))
    draw = ImageDraw.Draw(sheet)
    if title_h:
        draw.text((gap, gap // 2 + title_h // 6), title, font=title_font[0], fill=(235, 235, 240))
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x = gap + column * (tile + gap)
        y = title_h + gap + row * (tile + caption_h + gap)
        with Image.open(record.image_path) as image:
            thumb = image.convert("RGB")
        thumb.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x + (tile - thumb.width) // 2, y + (tile - thumb.height) // 2))
        if found:
            font = found[0]
            local = record.generated_at.astimezone()
            line1 = f"{local:%d.%m.%Y %H:%M} · {bg.EMOTION_NAMES_RU[record.dominant_emotion]} · {INTENT_NAMES_RU.get(record.intent_type, record.intent_type)}"
            if record.style in bg.STYLES:
                style_title = bg.STYLES[record.style].title
            elif record.archetype:
                configs = bg.active_configs()
                arch = configs.archetypes["archetypes"].get(record.archetype, {}).get("name", record.archetype)
                comp = configs.compositions["compositions"].get(record.composition, {}).get("name", record.composition)
                style_title = f"{arch}, {comp}"
            else:
                style_title = ""
            line2 = " · ".join(part for part in (style_title, ", ".join(record.keywords)) if part)
            max_chars = max(12, int(tile / (font.size * 0.52)))
            line1 = line1 if len(line1) <= max_chars else line1[: max_chars - 1] + "…"
            line2 = line2 if len(line2) <= max_chars else line2[: max_chars - 1] + "…"
            draw.text((x + 4, y + tile + 4), line1, font=font, fill=(225, 225, 230))
            draw.text((x + 4, y + tile + 4 + caption_h // 2), line2, font=font, fill=(170, 175, 190))
    return sheet


def unique_path(path: Path) -> Path:
    """Вернуть путь, не занятый существующим файлом (добавляя _2, _3…)."""
    path = Path(path)
    number = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        number += 1
    return candidate


def _parse_date(value: str) -> date:
    if value.lower() in ("today", "сегодня"):
        return datetime.now().date()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("дата в формате ГГГГ-ММ-ДД, например 2026-09-29, или 'сегодня'") from exc


def _parse_emotion(value: str) -> str:
    key = value.strip().lower()
    if key not in EMOTION_ALIASES:
        names = ", ".join(f"{k} ({v})" for k, v in bg.EMOTION_NAMES_RU.items())
        raise argparse.ArgumentTypeError(f"неизвестная эмоция; доступны: {names}")
    return EMOTION_ALIASES[key]


def _parse_style(value: str) -> str:
    key = value.strip().lower()
    if key not in bg.STYLE_ALIASES:
        raise argparse.ArgumentTypeError(f"неизвестный стиль; доступны: {', '.join(bg.STYLE_NAMES)}")
    return bg.STYLE_ALIASES[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bird_library.py", description="Коллаж из сохранённых птиц с фильтрами.")
    parser.add_argument("--date", type=_parse_date, help="Дата генерации ГГГГ-ММ-ДД или 'сегодня'")
    parser.add_argument("--emotion", type=_parse_emotion, help="Доминирующая эмоция: hope/надежда, joy/радость, ...")
    parser.add_argument("--intent", choices=bg.INTENT_TYPES, help="Намерение: acquire, release, mixed, unclear")
    parser.add_argument("--archetype", choices=("light", "crystal", "typographic", "flow", "particles", "organic", "digital", "memory"),
                        help="Основной архетип птицы")
    parser.add_argument("--style", type=_parse_style, help="Стиль: " + ", ".join(bg.STYLE_NAMES) + " (или по-русски: баухаус, готика…)")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR, help="Папка с JSON-отчётами")
    parser.add_argument("--output", type=Path, default=None, help="Путь к коллажу (по умолчанию новый файл в output/collages/)")
    parser.add_argument("--tile", type=int, default=360, help="Размер миниатюры в пикселях (по умолчанию 360)")
    parser.add_argument("--columns", type=int, default=None, help="Число колонок (по умолчанию автоматически)")
    parser.add_argument("--list", action="store_true", help="Только вывести список, без коллажа")
    parser.add_argument("--no-open", action="store_true", help="Не открывать коллаж после создания")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    if not 64 <= args.tile <= 1024 or (args.columns is not None and not 1 <= args.columns <= 20):
        print("Ошибка: --tile должен быть 64–1024, --columns 1–20", file=sys.stderr)
        return 2
    records, skipped = load_records(args.analysis_dir)
    selected = filter_records(records, args.date, args.emotion, args.intent, args.style, args.archetype)
    filters = []
    if args.date:
        filters.append(f"дата {args.date:%d.%m.%Y}")
    if args.emotion:
        filters.append(f"эмоция «{bg.EMOTION_NAMES_RU[args.emotion]}»")
    if args.intent:
        filters.append(f"намерение «{INTENT_NAMES_RU[args.intent]}»")
    if args.style:
        filters.append(f"стиль «{bg.STYLES[args.style].title}»")
    if args.archetype:
        filters.append(f"архетип «{args.archetype}»")
    description = ", ".join(filters) or "все птицы"
    print(f"Найдено птиц: {len(records)}; подходит под фильтр ({description}): {len(selected)}")
    if skipped:
        print(f"Пропущено отчётов без PNG или с неверным форматом: {skipped}")
    for record in selected:
        print(f"  {record.generated_at.astimezone():%Y-%m-%d %H:%M}  {bg.EMOTION_NAMES_RU[record.dominant_emotion]:<9} "
              f"{record.intent_type:<8} {record.style or record.archetype or '—':<15} {record.image_path}")
    if args.list:
        return 0
    if not selected:
        print("Нет птиц для коллажа — измените фильтр или сгенерируйте птиц.", file=sys.stderr)
        return 3
    collage = build_collage(selected, args.tile, args.columns, title=f"Библиотека птиц: {description} ({len(selected)})")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = unique_path(args.output or DEFAULT_COLLAGE_DIR / f"collage_{stamp}.png")
    try:
        bg.save_png(collage, output)
    except bg.BirdGeneratorError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return exc.exit_code
    print(f"Коллаж: {output}")
    if not args.no_open:
        bg.open_with_system_viewer(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
