#!/usr/bin/env python3
"""Контрольная галерея: семь эталонных текстов с фиксированными seed.

Использует только публичные функции ``bird_generator`` и не дублирует код
анализа и рендеринга. Результаты сохраняются в ``output/images/gallery/``
и ``output/analysis/gallery/`` (эти директории исключены из Git).
"""

from __future__ import annotations

import sys
from pathlib import Path

import bird_generator as bg

GALLERY: tuple[tuple[str, int, str], ...] = (
    ("01_positive_acquire", 101, "Я очень хочу увидеть море, путешествовать и почувствовать свободу."),
    ("02_calm_desire", 102, "Я хочу жить спокойно рядом с домом, чувствовать тепло и заботиться о семье."),
    ("03_anxious_desire", 103, "Я очень боюсь, что ничего не получится, но всё равно хочу изменить свою жизнь."),
    ("04_release", 104, "Я хочу отпустить прошлое, освободиться от тревоги и перестать бояться перемен."),
    ("05_mixed", 105, "Я хочу оставить страх в прошлом и начать путешествовать по миру."),
    ("06_anger", 106, "Меня ужасно злит эта несправедливость, я больше не хочу с ней мириться!"),
    ("07_sad_memory", 107, "Я очень скучаю по дому и хочу снова почувствовать тепло тех дней."),
)

IMAGE_DIR = Path("output/images/gallery")
ANALYSIS_DIR = Path("output/analysis/gallery")


def main() -> int:
    """Сгенерировать галерею; вернуть 0 при успехе и код ошибки иначе."""
    bg.configure_logging(debug=False)
    try:
        configs = bg.load_configs()
    except bg.BirdGeneratorError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return exc.exit_code
    exit_code = 0
    for name, seed, text in GALLERY:
        options = bg.GenerationOptions(
            output_path=IMAGE_DIR / f"{name}.png",
            analysis_path=ANALYSIS_DIR / f"{name}.json",
            seed=seed,
            input_description={"type": "gallery", "name": name},
            overwrite=True,  # эталонная галерея обновляется на месте
        )
        try:
            result = bg.run_generation([text], options, configs)
        except bg.BirdGeneratorError as exc:
            print(f"{name}: ошибка ({type(exc).__name__}): {exc}", file=sys.stderr)
            exit_code = exc.exit_code
            continue
        analysis = result.analysis
        print(
            f"{name}: intent={analysis.intent_type} valence={analysis.valence:.2f} "
            f"arousal={analysis.arousal:.2f} motifs={','.join(result.bird_spec.decorative_motifs)}"
        )
        print(f"  PNG:  {result.image_path}")
        print(f"  JSON: {result.analysis_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
