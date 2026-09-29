"""Тесты генератора абстрактной птицы.

Тесты не используют интернет, внешние API, пользовательскую папку data/input
и абсолютные пути конкретного компьютера. Изображения генерируются в
минимальном размере 512×512 без supersampling, чтобы тесты шли быстро.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import bird_generator as bg  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

GALLERY_TEXTS = [
    "Я очень хочу увидеть море, путешествовать и почувствовать свободу.",
    "Я хочу жить спокойно рядом с домом, чувствовать тепло и заботиться о семье.",
    "Я очень боюсь, что ничего не получится, но всё равно хочу изменить свою жизнь.",
    "Я хочу отпустить прошлое, освободиться от тревоги и перестать бояться перемен.",
    "Я хочу оставить страх в прошлом и начать путешествовать по миру.",
    "Меня ужасно злит эта несправедливость, я больше не хочу с ней мириться!",
    "Я очень скучаю по дому и хочу снова почувствовать тепло тех дней.",
]

SECRET_TEXT = "Я очень хочу увидеть северное сияние над тихим озером"


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def configs() -> bg.ConfigBundle:
    return bg.load_configs()


def analyze(texts: list[str] | str, configs: bg.ConfigBundle) -> bg.TextAnalysis:
    if isinstance(texts, str):
        texts = [texts]
    analysis, _, _ = bg.analyze_comments(bg.prepare_comments(texts), configs)
    return analysis


def spec_for(texts: list[str] | str, configs: bg.ConfigBundle, seed: int = 7) -> bg.BirdSpec:
    theme_hues = {k: v["hue_range"] for k, v in configs.themes["themes"].items()}
    return bg.analysis_to_bird_spec(analyze(texts, configs), seed, configs.mapping, theme_hues)


def small_options(tmp_path: Path, name: str = "bird", **kwargs) -> bg.GenerationOptions:
    params = dict(
        output_path=tmp_path / f"{name}.png",
        analysis_path=tmp_path / f"{name}.json",
        width=512,
        height=512,
        supersampling=1,
        seed=11,
        show_keywords=False,
    )
    params.update(kwargs)
    return bg.GenerationOptions(**params)


def run_cli(tmp_path: Path, *args: str) -> int:
    return bg.main([
        *args,
        "--output", str(tmp_path / "cli.png"),
        "--analysis-output", str(tmp_path / "cli.json"),
        "--width", "512", "--height", "512", "--supersampling", "1",
        "--no-preview", "--no-history",
    ])


NUMERIC_SPEC_FIELDS = [
    f.name for f in dataclasses.fields(bg.BirdSpec)
    if f.type in ("float", float) and f.name != "seed"
]


# ---------------------------------------------------------------------------
# 1–10. Чтение входных данных
# ---------------------------------------------------------------------------


def test_read_txt():
    comments = bg.read_comments(FIXTURES / "comments.txt")
    non_empty = [c for c in comments if c.strip()]
    assert len(non_empty) == 4
    assert non_empty[0] == "Я хочу путешествовать по миру."


def test_read_json_array_of_strings():
    assert bg.read_comments(FIXTURES / "comments_array.json") == ["Я хочу увидеть море", "Я мечтаю о путешествии"]


def test_read_json_array_of_objects():
    assert bg.read_comments(FIXTURES / "comments_objects.json") == ["Я хочу увидеть море", "Я мечтаю о путешествии"]


def test_read_json_container_with_comments():
    assert bg.read_comments(FIXTURES / "comments_container.json") == ["Я хочу увидеть море", "Я мечтаю о путешествии"]


def test_read_json_text_field_fallback():
    comments = bg.read_comments(FIXTURES / "comments_text_field.json", text_field="comment")
    assert comments == ["Я хочу увидеть море", "Я хочу отпустить тревогу"]
    assert bg.read_comments(FIXTURES / "comments_text_field.json", text_field="text") == comments


def test_read_csv_utf8():
    comments = bg.read_comments(FIXTURES / "comments.csv")
    assert comments[0] == "Я хочу увидеть море, горы и леса"
    assert len(comments) == 5


def test_read_csv_utf8_sig():
    comments = bg.read_comments(FIXTURES / "comments_utf8_sig.csv")
    assert comments == ["Я хочу научиться играть на гитаре", "Я мечтаю о спокойной жизни"]
    assert not comments[0].startswith("﻿")


def test_read_csv_extra_columns_and_custom_column(tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text('id,text,created_at,source,language\n1,"Я хочу увидеть море","2026","survey","ru"\n', encoding="utf-8")
    assert bg.read_comments(path, column="text") == ["Я хочу увидеть море"]
    no_column = _write(tmp_path / "nocol.csv", "a,b\n1,2\n")
    with pytest.raises(bg.InputDataError, match="колонки 'missing'"):
        bg.read_comments(no_column, column="missing")


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_read_csv_commas_and_quotes():
    comments = bg.read_comments(FIXTURES / "comments.csv")
    assert "Я хочу увидеть море, горы и леса" in comments
    assert 'Я мечтаю о "тихом" доме у реки' in comments


def test_read_csv_newline_inside_field():
    comments = bg.read_comments(FIXTURES / "comments.csv")
    assert "Я хочу отпустить тревогу.\nИ перестать бояться перемен" in comments


# ---------------------------------------------------------------------------
# 11–18. Подготовка, статистика, лимиты
# ---------------------------------------------------------------------------


def test_empty_records_removed():
    prepared = bg.prepare_comments(["", "   ", "\n\t", "Я хочу увидеть море"])
    assert prepared.comments == ["Я хочу увидеть море"]
    assert prepared.source_comment_count == 1


def test_whitespace_normalization_keeps_punctuation_and_case():
    text = "  Я   ОЧЕНЬ\tхочу\n\nморе!!!  Правда?.. 🙂 "
    assert bg.normalize_text(text) == "Я ОЧЕНЬ хочу море!!! Правда?.. 🙂"


def test_duplicates_removed_after_normalization():
    prepared = bg.prepare_comments(["Я хочу море", "Я  хочу   море ", "Я хочу горы и лес"])
    assert prepared.comments == ["Я хочу море", "Я хочу горы и лес"]
    assert prepared.duplicate_comment_count == 1


def test_comment_statistics_from_csv_fixture():
    prepared = bg.prepare_comments(bg.read_comments(FIXTURES / "comments.csv"))
    assert prepared.source_comment_count == 4
    assert prepared.unique_comment_count == 3
    assert prepared.duplicate_comment_count == 1
    assert prepared.analyzed_comment_count == 3
    assert prepared.truncated is False


def test_max_comments_limit():
    comments = [f"Я хочу увидеть море номер {i}" for i in range(10)]
    prepared = bg.prepare_comments(comments, max_comments=3)
    assert prepared.analyzed_comment_count == 3
    assert prepared.unique_comment_count == 10
    assert prepared.truncated is True
    assert prepared.warnings


def test_max_characters_excludes_whole_comments_and_truncates_first():
    prepared = bg.prepare_comments(["Я хочу увидеть море", "Очень длинный комментарий " * 5, "Хочу покоя"], max_characters=35)
    assert prepared.comments == ["Я хочу увидеть море", "Хочу покоя"]
    assert prepared.truncated is True
    single = bg.prepare_comments(["Я хочу увидеть море и почувствовать свободу прямо сейчас ✨"], max_characters=20)
    assert len(single.comments[0]) <= 20
    assert single.truncated is True
    single.comments[0].encode("utf-8")  # Unicode не повреждён


def test_missing_file_raises_and_cli_returns_input_error(tmp_path):
    with pytest.raises(bg.InputDataError, match="не найден"):
        bg.read_comments(tmp_path / "absent.csv")
    assert run_cli(tmp_path, "--input", str(tmp_path / "absent.csv")) == 3


def test_minimum_length():
    with pytest.raises(bg.InputDataError, match="короткий"):
        bg.prepare_comments(["Хочу"])
    with pytest.raises(bg.InputDataError):
        bg.prepare_comments(["", "  "])


# ---------------------------------------------------------------------------
# 19–29. Анализ, seed, стабильность
# ---------------------------------------------------------------------------


def test_seed_stability():
    first = bg.compute_seed(["Я хочу море", "Я хочу горы"])
    assert first == bg.compute_seed(["  Я хочу   море ", "Я хочу горы", "Я хочу море"])
    assert first != bg.compute_seed(["Я хочу горы", "Я хочу море"])
    expected = int.from_bytes(hashlib.sha256("Я хочу море\n␞\nЯ хочу горы".encode()).digest()[:8], "big")
    assert first == expected


@pytest.mark.parametrize("text", GALLERY_TEXTS + [
    "Я не счастлив!!! ПРАВДА?! 😢😢",
    "I want to see the sea and feel free",
    "Хочу море and freedom, очень-очень!!!",
    "🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂",
    "Сегодня обычный день, я пишу этот текст.",
])
def test_emotion_scores_in_range(text, configs):
    analysis = analyze(text, configs)
    for name in bg.BIPOLAR_PARAMS:
        assert 0.0 <= getattr(analysis, name) <= 1.0
    for value in list(analysis.emotions.values()) + list(analysis.score_confidence.values()):
        assert 0.0 <= value <= 1.0
    assert 0.0 <= analysis.confidence <= 1.0
    assert 0.0 <= analysis.intent_confidence <= 1.0
    assert 0.0 <= analysis.release_strength <= 1.0
    assert analysis.intent_type in bg.INTENT_TYPES
    assert len(analysis.keywords) <= 8
    assert len(analysis.intent_targets) <= 5


@pytest.mark.parametrize("text", GALLERY_TEXTS)
def test_bird_spec_ranges(text, configs):
    spec = spec_for(text, configs)
    for name in NUMERIC_SPEC_FIELDS:
        assert 0.0 <= getattr(spec, name) <= 1.0, name
    limits = configs.mapping["limits"]
    assert limits["minimum_feather_count"] <= spec.feather_count <= limits["maximum_feather_count"]
    assert 0 <= spec.particle_count <= limits["maximum_particle_count"]
    for color in (spec.background_color, spec.primary_color, spec.secondary_color, spec.accent_color, spec.outline_color):
        assert len(color) == 3 and all(0 <= c <= 255 for c in color)
    assert len(spec.decorative_motifs) <= limits["maximum_motif_count"]
    assert len(spec.keywords) <= 3


def test_intent_acquire(configs):
    assert analyze("Хочу любви и тепла", configs).intent_type == "acquire"
    analysis = analyze("Я очень хочу увидеть море и почувствовать свободу", configs)
    assert analysis.intent_type == "acquire"
    assert "море" in analysis.intent_targets


def test_intent_release(configs):
    analysis = analyze("Хочу отпустить прошлую любовь", configs)
    assert analysis.intent_type == "release"
    assert analysis.release_strength > 0.5
    assert analyze("Я хочу перестать тревожиться.", configs).intent_type == "release"
    assert analyze("Хочу не чувствовать тревогу.", configs).intent_type == "release"


def test_intent_mixed_and_unclear(configs):
    assert analyze("Хочу перестать бояться и начать путешествовать", configs).intent_type == "mixed"
    assert analyze("Я хочу оставить страх в прошлом и начать путешествовать по миру.", configs).intent_type == "mixed"
    assert analyze("Сегодня обычный день, я пишу этот текст.", configs).intent_type == "unclear"
    multi = analyze(["Я очень хочу увидеть море", "Я мечтаю путешествовать", "Я хочу отпустить тревогу"], configs)
    assert multi.intent_type == "mixed"


def test_fear_vs_no_longer_afraid(configs):
    afraid = analyze("Я боюсь темноты и одиночества", configs)
    not_afraid = analyze("Я больше не боюсь темноты и одиночества", configs)
    assert not_afraid.emotions["anxiety"] < afraid.emotions["anxiety"]
    assert not_afraid.dominance > afraid.dominance
    assert not_afraid.valence > afraid.valence
    happy = analyze("Я счастлив сегодня вечером", configs)
    unhappy = analyze("Я не счастлив сегодня вечером", configs)
    assert unhappy.valence < 0.5 < happy.valence
    assert unhappy.emotions["joy"] < happy.emotions["joy"]


def test_anxiety_vs_wish_to_stop_anxiety(configs):
    direct = analyze("Я сильно тревожусь и боюсь", configs)
    release = analyze("Я хочу перестать тревожиться.", configs)
    assert release.intent_type == "release"
    assert release.release_strength > direct.release_strength
    assert release.emotions["hope"] > direct.emotions["hope"]
    assert 0.0 < release.emotions["anxiety"] < direct.emotions["anxiety"]
    assert "release" in release.themes
    assert "тревога" in release.keywords
    no_feel = analyze("Хочу не чувствовать тревогу.", configs)
    assert 0.0 < no_feel.emotions["anxiety"] < analyze("Я чувствую тревогу.", configs).emotions["anxiety"]


def test_same_text_and_seed_give_same_spec(configs):
    assert spec_for(GALLERY_TEXTS[0], configs, seed=5) == spec_for(GALLERY_TEXTS[0], configs, seed=5)
    assert spec_for(GALLERY_TEXTS[0], configs, seed=5) != spec_for(GALLERY_TEXTS[0], configs, seed=6)


def test_same_text_and_seed_give_same_png(tmp_path, configs):
    first = bg.run_generation([GALLERY_TEXTS[3]], small_options(tmp_path, "a"), configs)
    second = bg.run_generation([GALLERY_TEXTS[3]], small_options(tmp_path, "b"), configs)
    assert first.image_path.read_bytes() == second.image_path.read_bytes()


def test_different_texts_give_different_specs(configs):
    specs = [spec_for(t, configs, seed=1) for t in GALLERY_TEXTS]
    assert len({json.dumps(dataclasses.asdict(s), sort_keys=True) for s in specs}) == len(specs)


def test_control_texts_differ_in_enough_significant_parameters(configs):
    """Раздел 40: контрольные тексты различаются по достаточному числу значимых параметров."""
    significant = [
        "body_elongation", "body_roundness", "wing_span", "wing_sweep", "wing_roundness",
        "wing_asymmetry", "tail_drop", "tail_length", "beak_sharpness", "fragmentation", "open_space",
    ]
    specs = [spec_for(t, configs, seed=1) for t in GALLERY_TEXTS]
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = specs[i], specs[j]
            differing = sum(1 for name in significant if abs(getattr(a, name) - getattr(b, name)) >= 0.05)
            palette_differs = a.primary_color != b.primary_color
            motifs_differ = a.decorative_motifs != b.decorative_motifs
            assert differing >= 3, (i, j, differing)
            assert palette_differs or motifs_differ


# ---------------------------------------------------------------------------
# 30–36. PNG и JSON
# ---------------------------------------------------------------------------


def test_png_created_and_valid(tmp_path, configs):
    result = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path), configs)
    assert result.image_path.is_file()
    with Image.open(result.image_path) as image:
        image.verify()
    with Image.open(result.image_path) as image:
        assert image.mode in ("RGB", "RGBA")
        extrema = image.convert("L").getextrema()
        assert extrema[1] - extrema[0] > 50  # изображение не однотонное
        assert not image.info.get("comment") and "Software" not in image.info


def test_png_size_is_correct(tmp_path, configs):
    options = small_options(tmp_path, width=640, height=512, supersampling=2)
    result = bg.run_generation([GALLERY_TEXTS[1]], options, configs)
    with Image.open(result.image_path) as image:
        assert image.size == (640, 512)


def test_output_directories_created_automatically(tmp_path, configs):
    options = small_options(tmp_path, output_path=tmp_path / "a" / "b" / "bird.png",
                            analysis_path=tmp_path / "c" / "d" / "bird.json")
    result = bg.run_generation([GALLERY_TEXTS[2]], options, configs)
    assert result.image_path.is_file() and result.analysis_path.is_file()
    assert not list((tmp_path / "a" / "b").glob("*.tmp"))


def test_json_excludes_source_text_by_default(tmp_path, configs):
    result = bg.run_generation([SECRET_TEXT], small_options(tmp_path), configs)
    raw = result.analysis_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert SECRET_TEXT not in raw
    assert "северное сияние над тихим" not in raw
    assert "source_comments" not in data
    assert data["schema_version"] == "1.0"
    for key in ("generated_at", "input_statistics", "analysis", "bird_spec", "mapping_explanation",
                "render", "config_versions", "config_hashes", "warnings", "truncated", "paths", "run_parameters"):
        assert key in data


def test_json_includes_source_text_only_with_flag(tmp_path, configs):
    options = small_options(tmp_path, include_source_text=True)
    result = bg.run_generation(["  " + SECRET_TEXT + "  "], options, configs)
    data = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert data["source_comments"] == [SECRET_TEXT]


def test_json_keeps_cyrillic_unescaped(tmp_path, configs):
    result = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path), configs)
    raw = result.analysis_path.read_text(encoding="utf-8")
    assert "море" in raw
    assert "\\u04" not in raw


def test_missing_font_does_not_crash(tmp_path, configs, monkeypatch):
    monkeypatch.setattr(bg, "FONT_CANDIDATES", ())
    monkeypatch.setattr(bg, "CALLIGRAPHY_FONT_CANDIDATES", ())
    result = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path, show_keywords=True), configs)
    assert result.image_path.is_file()
    assert any("Шрифт" in w for w in result.warnings)


def test_show_keywords_with_available_font_or_warning(tmp_path, configs):
    result = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path, show_keywords=True), configs)
    assert result.image_path.is_file()
    font = result.document["render"]["keywords_font"]
    assert font is not None or any("Шрифт" in w for w in result.warnings)


def test_keywords_do_not_change_image_without_flag(tmp_path, configs, monkeypatch):
    first = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path, "a"), configs)
    monkeypatch.setattr(bg, "FONT_CANDIDATES", ())
    monkeypatch.setattr(bg, "CALLIGRAPHY_FONT_CANDIDATES", ())
    second = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path, "b"), configs)
    assert first.image_path.read_bytes() == second.image_path.read_bytes()


# ---------------------------------------------------------------------------
# 37–45. Нейтральность, ошибки, конфиденциальность, воспроизводимость
# ---------------------------------------------------------------------------


def test_neutral_text_gives_moderate_bird(configs):
    analysis = analyze("Сегодня обычный день, я пишу этот текст.", configs)
    assert analysis.intent_type == "unclear"
    assert analysis.confidence < 0.3
    spec = spec_for("Сегодня обычный день, я пишу этот текст.", configs)
    for name in ("body_scale", "body_roundness", "wing_span", "wing_sweep", "head_lift", "body_tilt"):
        assert 0.2 <= getattr(spec, name) <= 0.8, name
    assert spec.fragmentation < 0.3
    assert spec.wing_asymmetry < 0.3
    assert spec.particle_count < 40


def test_corrupted_config_raises_clear_error(tmp_path, monkeypatch):
    config_copy = tmp_path / "config"
    shutil.copytree(PROJECT_DIR / "config", config_copy)
    (config_copy / "phrase_rules_ru.json").write_text("{ broken json", encoding="utf-8")
    with pytest.raises(bg.ConfigurationError, match="phrase_rules_ru.json"):
        bg.load_configs(config_copy)

    shutil.copy(PROJECT_DIR / "config" / "phrase_rules_ru.json", config_copy / "phrase_rules_ru.json")
    data = json.loads((config_copy / "themes_ru.json").read_text(encoding="utf-8"))
    data["schema_version"] = "9.9"
    (config_copy / "themes_ru.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(bg.ConfigurationError, match="версия схемы"):
        bg.load_configs(config_copy)

    data["schema_version"] = "1.0"
    del data["themes"]
    (config_copy / "themes_ru.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(bg.ConfigurationError, match="themes"):
        bg.load_configs(config_copy)

    monkeypatch.setattr(bg, "CONFIG_DIR", config_copy)
    assert run_cli(tmp_path, "--text", GALLERY_TEXTS[0]) == 4


def test_invalid_input_gives_nonzero_exit_code(tmp_path, monkeypatch, capsys):
    assert run_cli(tmp_path, "--text", GALLERY_TEXTS[0], "--width", "100") == 2
    assert bg.main(["--text", "a", "--input", "b.csv"]) == 2
    assert bg.main(["--text", GALLERY_TEXTS[0], "--supersampling", "4"]) == 2
    assert bg.main(["--text", GALLERY_TEXTS[0], "--seed", "-1"]) == 2
    assert bg.main(["--text", GALLERY_TEXTS[0], "--max-characters", "5"]) == 2
    assert run_cli(tmp_path, "--text", "Хочу") == 3
    bad_json = _write(tmp_path / "bad.json", "{not json")
    assert run_cli(tmp_path, "--input", str(bad_json)) == 3
    unsupported = _write(tmp_path / "data.xml", "<a/>")
    assert run_cli(tmp_path, "--input", str(unsupported)) == 3
    monkeypatch.chdir(tmp_path)
    assert bg.main([]) == 3
    assert bg.NO_SOURCE_MESSAGE in capsys.readouterr().err


def test_default_input_csv_is_used(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "input").mkdir(parents=True)
    shutil.copy(FIXTURES / "comments.csv", tmp_path / "data" / "input" / "comments.csv")
    assert bg.main(["--width", "512", "--height", "512", "--supersampling", "1", "--no-preview", "--no-history"]) == 0
    images = list((tmp_path / "output" / "images").glob("bird_*.png"))
    assert len(images) == 1
    data = json.loads((tmp_path / "output" / "analysis" / f"{images[0].stem}.json").read_text(encoding="utf-8"))
    assert data["input_statistics"]["analyzed_comment_count"] == 3


def test_cli_logs_do_not_contain_source_text(tmp_path, capsys):
    assert run_cli(tmp_path, "--text", SECRET_TEXT, "--debug") == 0
    captured = capsys.readouterr()
    assert SECRET_TEXT not in captured.err + captured.out
    assert "северное сияние" not in captured.err + captured.out
    data = json.loads((tmp_path / "cli.json").read_text(encoding="utf-8"))
    assert "diagnostics" in data
    assert SECRET_TEXT not in json.dumps(data, ensure_ascii=False)


def test_mapping_explanation_contains_no_source_text(configs):
    for text in GALLERY_TEXTS + [SECRET_TEXT]:
        analysis = analyze(text, configs)
        spec = spec_for(text, configs)
        explanation = bg.build_mapping_explanation(analysis, spec, configs.themes)
        joined = " ".join(line for lines in explanation.values() for line in lines)
        assert set(explanation) == {"palette", "shape", "intent", "motifs", "style", "art"}
        assert text not in joined
        assert text.rstrip(".!") not in joined
        assert all(len(lines) <= 6 for lines in explanation.values())


def test_low_confidence_weakens_artistic_influence(configs):
    analysis = analyze(GALLERY_TEXTS[5], configs)
    low = dataclasses.replace(
        analysis,
        score_confidence={k: 0.1 for k in analysis.score_confidence},
        intent_confidence=0.1,
    )
    neutral = dataclasses.replace(
        analysis,
        valence=0.5, arousal=0.5, intensity=0.5, dominance=0.5,
        emotions={k: 0.0 for k in analysis.emotions}, intent_type="unclear", intent_confidence=0.0,
        release_strength=0.0, themes=[], visual_motifs=[],
    )
    high_spec = bg.analysis_to_bird_spec(analysis, 3, configs.mapping)
    low_spec = bg.analysis_to_bird_spec(low, 3, configs.mapping)
    neutral_spec = bg.analysis_to_bird_spec(neutral, 3, configs.mapping)

    def distance(spec):
        return sum(abs(getattr(spec, n) - getattr(neutral_spec, n)) for n in NUMERIC_SPEC_FIELDS if n != "intent_strength")

    assert distance(low_spec) < 0.5 * distance(high_spec)
    effective_low = bg.effective_scores(low)
    assert effective_low["anger"] < 0.15
    assert abs(effective_low["arousal"] - 0.5) < abs(analysis.arousal - 0.5)


def test_config_versions_and_hashes_saved(tmp_path, configs):
    result = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path), configs)
    data = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert set(data["config_versions"]) == set(bg.CONFIG_FILES)
    assert all(v == "1.0" for v in data["config_versions"].values())
    for key, file_name in bg.CONFIG_FILES.items():
        expected = hashlib.sha256((PROJECT_DIR / "config" / file_name).read_bytes()).hexdigest()[:12]
        assert data["config_hashes"][key] == expected


def test_release_differs_visually_from_direct_anxiety(tmp_path, configs):
    direct_text = "Я очень тревожусь, мне страшно и я боюсь перемен."
    release_text = "Я хочу отпустить тревогу, освободиться от страха и перестать бояться перемен."
    direct = spec_for(direct_text, configs, seed=4)
    release = spec_for(release_text, configs, seed=4)
    assert release.intent_type == "release" and direct.intent_type == "unclear"
    assert release.particle_count > direct.particle_count
    assert release.open_space > direct.open_space
    assert "separated_particles" in release.decorative_motifs
    assert release.fragmentation < direct.fragmentation
    assert release.wing_sweep > direct.wing_sweep
    assert rgb_lightness(release.background_color) != rgb_lightness(direct.background_color)
    a = bg.run_generation([direct_text], small_options(tmp_path, "direct", seed=4), configs)
    b = bg.run_generation([release_text], small_options(tmp_path, "release", seed=4), configs)
    assert a.image_path.read_bytes() != b.image_path.read_bytes()


def rgb_lightness(color):
    return bg.rgb_lightness(color)


def test_result_does_not_depend_on_global_random(tmp_path, configs):
    random.seed(1)
    first = bg.run_generation([GALLERY_TEXTS[4]], small_options(tmp_path, "g1", seed=None), configs)
    random.seed(987654)
    random.random()
    second = bg.run_generation([GALLERY_TEXTS[4]], small_options(tmp_path, "g2", seed=None), configs)
    assert first.bird_spec == second.bird_spec
    assert first.image_path.read_bytes() == second.image_path.read_bytes()
    assert first.seed_source == "sha256"


def test_builtin_hash_not_used_for_seed():
    source = inspect.getsource(bg.compute_seed)
    code_lines = [line for line in source.splitlines() if not line.strip().startswith(("#", '"', "``", "от ", "Встроенный"))]
    assert not any(" hash(" in line or "=hash(" in line for line in code_lines)
    script = "import bird_generator as bg; print(bg.compute_seed(['Я хочу увидеть море']))"
    outputs = set()
    for hash_seed in ("1", "2", "random"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        completed = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_DIR, env=env,
                                   capture_output=True, text=True, check=True)
        outputs.add(completed.stdout.strip())
    assert len(outputs) == 1


# ---------------------------------------------------------------------------
# Дополнительные проверки надёжности
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("background", ["auto", "solid", "gradient"])
def test_all_background_modes(tmp_path, configs, background):
    result = bg.run_generation([GALLERY_TEXTS[3]], small_options(tmp_path, background=background), configs)
    assert result.document["render"]["background_mode"] in ("solid", "gradient")
    if background != "auto":
        assert result.document["render"]["background_mode"] == background


@pytest.mark.parametrize("text", [
    "I want to see the sea and feel free",
    "Хочу море and freedom!!! 😊😊",
    "Абракадабра зюзюка фыщь кряквач",
    "Я счастлив, но мне страшно и грустно, и я злюсь!!!",
])
def test_robust_inputs_do_not_crash(tmp_path, configs, text):
    result = bg.run_generation([text], small_options(tmp_path), configs)
    assert result.image_path.is_file()
    if result.analysis.language != "ru":
        assert any("Язык" in w for w in result.warnings)


def test_many_comments_aggregate_into_one_bird(tmp_path, configs):
    comments = bg.read_comments(PROJECT_DIR / "data" / "examples" / "example_comments.json", text_field="comment")
    result = bg.run_generation(comments, small_options(tmp_path), configs)
    assert result.analysis.analyzed_comment_count == 3
    assert result.analysis.intent_type == "mixed"
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_long_comment_does_not_suppress_short_ones(configs):
    long_neutral = "Сегодня я шёл по улице и смотрел на дома, машины, людей и витрины магазинов. " * 6
    short_angry = "Меня ужасно злит несправедливость!"
    analysis = analyze([long_neutral, short_angry], configs)
    assert analysis.emotions["anger"] > 0.4


def test_existing_output_kept_when_write_fails(tmp_path, configs, monkeypatch):
    options = small_options(tmp_path, overwrite=True)
    bg.run_generation([GALLERY_TEXTS[0]], options, configs)
    before = options.output_path.read_bytes()

    def broken_save(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", broken_save)
    with pytest.raises(bg.OutputWriteError):
        bg.run_generation([GALLERY_TEXTS[1]], options, configs)
    assert options.output_path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


# ---------------------------------------------------------------------------
# Библиотека птиц, история, ключевые слова, предпросмотр
# ---------------------------------------------------------------------------


def test_new_birds_do_not_overwrite_previous(tmp_path, configs):
    options = small_options(tmp_path)
    first = bg.run_generation([GALLERY_TEXTS[0]], options, configs)
    before = first.image_path.read_bytes()
    second = bg.run_generation([GALLERY_TEXTS[1]], options, configs)
    third = bg.run_generation([GALLERY_TEXTS[2]], options, configs)
    assert first.image_path.name == "bird.png"
    assert second.image_path.name == "bird_2.png" and second.analysis_path.name == "bird_2.json"
    assert third.image_path.name == "bird_3.png"
    assert first.image_path.read_bytes() == before
    data = json.loads(second.analysis_path.read_text(encoding="utf-8"))
    assert data["paths"]["image"].endswith("bird_2.png")


def test_default_names_are_unique_per_run(tmp_path, monkeypatch, configs):
    monkeypatch.chdir(tmp_path)
    options = bg.GenerationOptions(width=512, height=512, supersampling=1, show_keywords=False)
    first = bg.run_generation([GALLERY_TEXTS[0]], options, configs)
    second = bg.run_generation([GALLERY_TEXTS[0]], options, configs)
    assert first.image_path.parent == Path("output/images")
    assert first.image_path.name.startswith("bird_") and first.image_path != second.image_path
    assert first.analysis_path == Path("output/analysis") / f"{first.image_path.stem}.json"
    assert first.image_path.read_bytes() == second.image_path.read_bytes()


def test_history_is_appended_across_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = ["--width", "512", "--height", "512", "--supersampling", "1", "--no-preview"]
    assert bg.main(["--text", GALLERY_TEXTS[0], *base]) == 0
    assert bg.main(["--text", GALLERY_TEXTS[3], *base]) == 0
    assert bg.main(["--text", GALLERY_TEXTS[5], *base, "--no-history"]) == 0
    history = tmp_path / "data" / "history" / "requests.csv"
    rows = list(csv_rows(history))
    assert [r["comment"] for r in rows] == [GALLERY_TEXTS[0], GALLERY_TEXTS[3]]
    assert all(Path(r["image"]).name.startswith("bird_") for r in rows)
    assert bg.read_comments(history) == [GALLERY_TEXTS[0], GALLERY_TEXTS[3]]


def csv_rows(path):
    import csv
    with open(path, encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def test_keywords_are_drawn_by_default_and_can_be_hidden(tmp_path):
    assert bg.GenerationOptions().show_keywords is True
    assert bg.build_parser().parse_args([]).show_keywords is True
    assert bg.build_parser().parse_args(["--hide-keywords"]).show_keywords is False
    with_words = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path, "on", show_keywords=True))
    without = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path, "off", show_keywords=False))
    if with_words.document["render"]["keywords_font"]:
        assert with_words.image_path.read_bytes() != without.image_path.read_bytes()


def test_progress_frames_recorded_without_changing_result(tmp_path, configs):
    recorder = bg.DrawingRecorder(max_side=200)
    with_progress = bg.run_generation([GALLERY_TEXTS[3]], small_options(tmp_path, "p", progress=recorder), configs)
    plain = bg.run_generation([GALLERY_TEXTS[3]], small_options(tmp_path, "q"), configs)
    assert with_progress.image_path.read_bytes() == plain.image_path.read_bytes()
    labels = [label for _, label in recorder.frames]
    assert len(recorder.frames) >= 12
    layer_numbers = [int(label.split(".")[0]) for label in labels if label[0].isdigit()]
    assert layer_numbers[0] == 1 and layer_numbers[-1] == 11
    assert layer_numbers == sorted(layer_numbers)  # слои строго в порядке 1 → 11
    assert set(range(1, 12)) <= set(layer_numbers)
    assert max(recorder.frames[0][0].size) == 200


def test_money_wishes_are_recognized(configs):
    analysis = analyze("Я хочу зарабатывать на том что люблю и жить безбедно", configs)
    assert analysis.themes[0] == "money"
    assert {"заработок", "безбедность"} <= set(analysis.keywords)
    fear_of_job_loss = analyze("Я боюсь что останусь без работы хочу перестать бояться", configs)
    assert "career" in fear_of_job_loss.themes
    assert "останусь" not in fear_of_job_loss.keywords
    assert "стану" not in analyze("Я хочу что стану смелее и буду путешествовать", configs).keywords


def test_library_collage_filters_by_emotion_and_date(tmp_path, configs):
    import bird_library as lib
    for index, text in enumerate([GALLERY_TEXTS[0], GALLERY_TEXTS[5], GALLERY_TEXTS[6]]):
        bg.run_generation([text], small_options(tmp_path / "birds", f"b{index}"), configs)
    records, skipped = lib.load_records(tmp_path / "birds")
    assert len(records) == 3 and skipped == 0
    angry = lib.filter_records(records, emotion="anger")
    assert len(angry) == 1 and angry[0].image_path.name == "b1.png"
    today = records[0].generated_at.astimezone().date()
    assert len(lib.filter_records(records, day=today)) == 3
    assert lib.filter_records(records, day=today.replace(year=today.year - 1)) == []
    collage = lib.build_collage(records, tile=128)
    assert collage.width >= 2 * 128 and collage.height >= 2 * 128 and collage.mode == "RGB"
    output = tmp_path / "collage.png"
    assert lib.main(["--analysis-dir", str(tmp_path / "birds"), "--emotion", "гнев", "--tile", "128",
                     "--output", str(output), "--no-open"]) == 0
    assert output.is_file()
    assert lib.main(["--analysis-dir", str(tmp_path / "birds"), "--emotion", "нечто"]) == 2
    assert lib.main(["--analysis-dir", str(tmp_path / "empty"), "--no-open"]) == 3


def test_keywords_follow_inner_lines_of_the_bird(configs):
    spec = spec_for(GALLERY_TEXTS[0], configs)
    geometry = bg.build_bird_geometry(spec, random.Random(1))
    assert set(geometry.text_paths) == {"wing", "body", "tail"}
    for points, color, thickness in geometry.text_paths.values():
        assert len(points) >= 10 and thickness > 0 and len(color) == 3
    found = bg.find_cyrillic_font(20)
    if found is None:
        pytest.skip("В системе нет шрифта с кириллицей")
    image = Image.new("RGB", (400, 200), (200, 180, 120))
    arc = [(40 + 3.2 * i, 120 - 40 * math.sin(math.pi * i / 100)) for i in range(101)]
    assert bg.draw_text_on_path(image, "заработок", arc, found[1], 30, 10, (200, 180, 120))
    changed = [(x, y) for x in range(0, 400, 2) for y in range(0, 200, 2) if image.getpixel((x, y)) != (200, 180, 120)]
    assert changed, "буквы не нарисованы"
    assert min(y for _, y in changed) > 50 and max(y for _, y in changed) < 150  # буквы вдоль дуги, а не где-то ещё
    reversed_image = Image.new("RGB", (400, 200), (200, 180, 120))
    assert bg.draw_text_on_path(reversed_image, "заработок", arc[::-1], found[1], 30, 10, (200, 180, 120))
    assert reversed_image.tobytes() == image.tobytes()  # направление линии не переворачивает текст
    assert not bg.draw_text_on_path(image, "очень длинное слово", arc[:5], found[1], 30, 10, (0, 0, 0))


def test_calligraphy_font_preferred_and_custom_font_option(tmp_path):
    candidates = bg.keyword_font_candidates("my_font.ttf")
    assert candidates[0] == "my_font.ttf"
    assert candidates[1:1 + len(bg.CALLIGRAPHY_FONT_CANDIDATES)] == list(bg.CALLIGRAPHY_FONT_CANDIDATES)
    assert bg.build_parser().parse_args(["--keyword-font", "x.ttf"]).keyword_font == Path("x.ttf")
    found = bg.find_cyrillic_font(20, bg.keyword_font_candidates())
    if found is None:
        pytest.skip("В системе нет шрифта с кириллицей")
    result = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path, show_keywords=True,
                                                                 keyword_font=tmp_path / "missing.ttf"))
    assert result.document["render"]["keywords_font"] is not None
    assert any("--keyword-font" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Художественно-архитектурные стили
# ---------------------------------------------------------------------------


def test_ten_styles_are_defined():
    assert len(bg.STYLES) == 10
    assert bg.choose_style("баухаус", 1) == ("bauhaus", "cli")
    assert bg.choose_style("seed", 5) == bg.choose_style("seed", 5)
    assert bg.choose_style("random", 5)[0] in bg.STYLES
    assert {bg.choose_style("seed", s)[0] for s in range(200)} == set(bg.STYLES)
    with pytest.raises(bg.InputDataError):
        bg.choose_style("рококо", 1)
    assert bg.build_parser().parse_args([]).style == "none"
    assert bg.choose_style("none", 1) == ("", "none")
    assert bg.build_parser().parse_args(["--style", "Готика"]).style == "gothic"


@pytest.mark.parametrize("style", sorted(bg.STYLES))
def test_every_style_renders_and_is_recorded(tmp_path, configs, style):
    result = bg.run_generation([GALLERY_TEXTS[3]], small_options(tmp_path, style=style), configs)
    with Image.open(result.image_path) as image:
        assert image.size == (512, 512) and image.mode == "RGB"
    data = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert data["bird_spec"]["style"] == style
    assert data["render"]["style_title"] == bg.STYLES[style].title
    assert data["render"]["style_source"] == "cli"
    assert data["mapping_explanation"]["style"][0].startswith(f"Стиль «{bg.STYLES[style].title}»")
    for name in NUMERIC_SPEC_FIELDS:
        assert 0.0 <= getattr(result.bird_spec, name) <= 1.0


def test_styles_change_image_but_keep_emotional_geometry(tmp_path, configs):
    a = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path, "a", style="bauhaus"), configs)
    b = bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path, "b", style="ukiyo_e"), configs)
    assert a.image_path.read_bytes() != b.image_path.read_bytes()
    assert a.bird_spec.primary_color != b.bird_spec.primary_color
    for name in ("wing_span", "wing_sweep", "tail_drop", "wing_asymmetry", "body_tilt", "open_space"):
        assert getattr(a.bird_spec, name) == getattr(b.bird_spec, name)


def test_same_style_and_seed_reproducible(tmp_path, configs):
    first = bg.run_generation([GALLERY_TEXTS[6]], small_options(tmp_path, "x", style="byzantine"), configs)
    second = bg.run_generation([GALLERY_TEXTS[6]], small_options(tmp_path, "y", style="byzantine"), configs)
    assert first.image_path.read_bytes() == second.image_path.read_bytes()


def test_cli_random_style_is_reported_and_in_file_name(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert bg.main(["--text", GALLERY_TEXTS[0], "--width", "512", "--height", "512", "--supersampling", "1",
                    "--no-preview", "--no-history", "--style", "random"]) == 0
    image = next((tmp_path / "output" / "images").glob("bird_*.png"))
    data = json.loads((tmp_path / "output" / "analysis" / f"{image.stem}.json").read_text(encoding="utf-8"))
    assert data["render"]["style_source"] == "random"
    assert image.stem.endswith(data["bird_spec"]["style"])
    assert "Стиль:" in capsys.readouterr().err
    assert bg.main(["--list-styles"]) == 0


def test_all_styles_collage_from_one_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert bg.main(["--text", GALLERY_TEXTS[0], "--all-styles", "--width", "512", "--height", "512",
                    "--supersampling", "1", "--no-preview", "--no-history"]) == 0
    images = sorted((tmp_path / "output" / "images").glob("bird_*.png"))
    assert len(images) == 10
    assert {image.stem.split("_", 3)[-1] for image in images} == set(bg.STYLE_NAMES)
    collages = list((tmp_path / "output" / "collages").glob("styles_*.png"))
    assert len(collages) == 1
    with Image.open(collages[0]) as collage:
        assert collage.width > collage.height  # 5 колонок × 2 ряда
    seeds = {json.loads(p.read_text(encoding="utf-8"))["run_parameters"]["seed"]
             for p in (tmp_path / "output" / "analysis").glob("*.json")}
    assert len(seeds) == 1


# ---------------------------------------------------------------------------
# Художественная система: архетипы, композиции, материалы, отбор кандидатов
# ---------------------------------------------------------------------------

ARCHETYPES = {"light", "crystal", "typographic", "flow", "particles", "organic", "digital", "memory"}
COMPOSITIONS = {"iconic_front", "ascending_diagonal", "quiet_profile", "open_spiral", "fragmented_escape",
                "floating_horizontal", "compressed_core", "dual_direction"}
MATERIALS = {"light", "glass", "ink", "particles", "ribbons", "letters", "leaves", "digital_grid", "crystal", "smoke"}


def art_spec(text, configs, seed=7, **overrides):
    analysis = analyze(text, configs)
    theme_hues = {k: v["hue_range"] for k, v in configs.themes["themes"].items()}
    base = bg.analysis_to_bird_spec(analysis, seed, configs.mapping, theme_hues)
    return bg.direct_art(analysis, base, seed, configs, overrides=overrides or None)


def test_art_configs_define_required_vocabulary(configs):
    assert set(configs.archetypes["archetypes"]) == ARCHETYPES
    assert set(configs.compositions["compositions"]) == COMPOSITIONS
    assert set(configs.materials["materials"]) == MATERIALS
    assert set(configs.palettes["strategies"]) == {"analogous", "complementary_accent", "monochrome",
                                                   "dark_field_light_accent", "warm_cool_contrast"}
    assert set(configs.archetypes["rare_features"]["features"]) == {
        "split_tail", "luminous_eye", "hollow_core", "secondary_ghost", "letter_feathers", "ring_trace",
        "asymmetric_materials", "shifted_core", "dissolving_edge", "inverted_glow"}


def test_broken_art_config_gives_clear_error(tmp_path):
    config_copy = tmp_path / "config"
    shutil.copytree(PROJECT_DIR / "config", config_copy)
    data = json.loads((config_copy / "visual_archetypes.json").read_text(encoding="utf-8"))
    data["archetypes"]["light"]["materials"] = ["plasma"]
    (config_copy / "visual_archetypes.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(bg.ConfigurationError, match="неизвестный материал"):
        bg.load_configs(config_copy)
    bg.load_configs()  # вернуть рабочие конфигурации в кэш


@pytest.mark.parametrize("text", GALLERY_TEXTS)
def test_archetype_mixing_limits(configs, text):
    signals = bg.art_signals(analyze(text, configs))
    for seed in range(60):
        primary, secondary, accent, weights = bg.mix_archetypes(signals, configs, random.Random(seed))
        assert len(weights) <= 3 and len({primary, secondary, accent} - {""}) == len(weights)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert 0.55 <= weights[primary] <= 0.75
        assert 0.20 <= weights[secondary] <= 0.35
        if accent:
            assert 0.0 < weights[accent] <= 0.15


def test_emotions_shift_archetype_weights(configs):
    def raw(text):
        return bg.archetype_raw_weights(bg.art_signals(analyze(text, configs)), configs)
    hopeful = raw("Я очень хочу увидеть море и почувствовать свободу, я так надеюсь!")
    releasing = raw("Я хочу отпустить прошлое, освободиться от тревоги и перестать бояться перемен.")
    angry = raw("Меня ужасно злит эта несправедливость, я больше не хочу с ней мириться!")
    tender = raw("Я хочу жить спокойно рядом с домом, чувствовать тепло и заботиться о семье.")
    assert max(hopeful, key=hopeful.get) in {"light", "flow"}
    assert releasing["particles"] > hopeful["particles"] and releasing["memory"] > hopeful["memory"]
    assert angry["crystal"] > tender["crystal"]
    assert tender["organic"] > angry["organic"]


def test_rare_features_are_rare_and_deterministic(configs):
    signals = bg.art_signals(analyze(GALLERY_TEXTS[3], configs))
    features = [bg.choose_rare_feature(signals, seed, configs) for seed in range(1000)]
    share = sum(bool(f) for f in features) / len(features)
    assert 0.08 <= share <= 0.15
    assert features == [bg.choose_rare_feature(signals, seed, configs) for seed in range(1000)]
    assert set(filter(None, features)) <= set(configs.archetypes["rare_features"]["features"])


@pytest.mark.parametrize("text", GALLERY_TEXTS)
def test_selected_spec_is_valid_and_bounded(configs, text):
    spec, report = art_spec(text, configs, seed=17)
    assert bg.validate_bird_spec(spec) == []
    assert 0.25 <= spec.abstraction_level <= 0.70
    assert 0.02 <= spec.texture_strength <= 0.12
    assert 2 <= spec.layer_count <= 7
    assert spec.composition_type in COMPOSITIONS and spec.material_type in MATERIALS
    assert spec.secondary_material_type != spec.material_type
    assert spec.focal_point in bg.FOCAL_POINTS
    assert spec.main_effect and spec.accent_effect and spec.main_effect != spec.accent_effect
    assert len(spec.decorative_motifs) <= 3
    colors = {spec.primary_color, spec.secondary_color, spec.accent_color, spec.outline_color}
    assert 2 <= len(colors) <= 4
    assert abs(bg.rgb_lightness(spec.primary_color) - bg.rgb_lightness(spec.background_color)) >= 0.25
    assert report["candidates"] == 5 and len(report["candidate_scores"]) == 5
    assert set(report["criteria"]) == set(bg.CRITERIA_NAMES_RU)
    assert report["winner_score"] == max(c["score"] for c in report["candidate_scores"])
    assert report["reasons"]


def test_candidate_selection_is_deterministic(configs):
    first = art_spec(GALLERY_TEXTS[0], configs, seed=3)
    second = art_spec(GALLERY_TEXTS[0], configs, seed=3)
    assert first == second
    assert art_spec(GALLERY_TEXTS[0], configs, seed=4)[0] != first[0]


def test_validate_and_repair(configs):
    spec, _ = art_spec(GALLERY_TEXTS[2], configs)
    broken = dataclasses.replace(
        spec, body_scale=0.1, eye_scale=0.95, fragmentation=0.9, wing_asymmetry=0.9, abstraction_level=0.95,
        secondary_material_type=spec.material_type, primary_color=spec.background_color,
        decorative_motifs=["waves", "stars", "leaves", "fire", "home"], focal_point="", feather_count=84, layer_count=7,
    )
    issues = bg.validate_bird_spec(broken)
    assert len(issues) >= 8
    repaired = bg.repair_bird_spec(broken, configs)
    assert bg.validate_bird_spec(repaired) == []
    assert bg.repair_bird_spec(broken, configs) == repaired  # без случайности


@pytest.mark.parametrize("composition", sorted(COMPOSITIONS))
def test_every_composition_renders_and_changes_geometry(tmp_path, configs, composition):
    options = small_options(tmp_path, art_overrides={"composition": composition})
    result = bg.run_generation([GALLERY_TEXTS[0]], options, configs)
    assert result.bird_spec.composition_type == composition
    comp = configs.compositions["compositions"][composition]
    low, high = comp["wing_sweep"]
    assert low - 1e-6 <= result.bird_spec.wing_sweep <= high + 1e-6
    geometry = bg.build_art_geometry(result.bird_spec, random.Random(1))
    assert geometry.frontal == comp["frontal"]
    with Image.open(result.image_path) as image:
        assert image.size == (512, 512)


@pytest.mark.parametrize("material", sorted(MATERIALS))
def test_every_material_renders(tmp_path, configs, material):
    result = bg.run_generation([GALLERY_TEXTS[6]], small_options(tmp_path, art_overrides={"material": material}), configs)
    assert result.bird_spec.material_type == material
    with Image.open(result.image_path) as image:
        extrema = image.convert("L").getextrema()
        assert extrema[1] - extrema[0] > 40


def test_frontal_geometry_is_symmetric(configs):
    spec, _ = art_spec(GALLERY_TEXTS[0], configs, composition="iconic_front")
    spec = dataclasses.replace(spec, wing_asymmetry=0.0)
    geometry = bg.build_art_geometry(spec, random.Random(5))
    near_x = sorted(round(x, 3) for x, _ in geometry.wings["near"])
    far_x = sorted(round(-x, 3) for x, _ in geometry.wings["far"])
    assert near_x == far_x
    assert len(geometry.eyes) == 2


def test_art_selection_saved_to_json(tmp_path, configs):
    result = bg.run_generation([GALLERY_TEXTS[4]], small_options(tmp_path, candidates=4), configs)
    data = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    selection = data["art_selection"]
    assert selection["candidates"] == 4 and selection["winner_seed"] == data["bird_spec"]["seed"]
    for key in ("primary_archetype", "archetype_weights", "composition_type", "abstraction_level", "material_type",
                "silhouette_type", "feather_style", "edge_style", "texture_style", "texture_strength", "layer_count",
                "negative_space", "inner_void", "silhouette_openness", "visual_tension", "focal_point",
                "focal_strength", "rare_feature", "palette_strategy"):
        assert key in data["bird_spec"]
    assert data["mapping_explanation"]["art"]
    assert GALLERY_TEXTS[4] not in json.dumps(data, ensure_ascii=False)


def test_art_cli_options(tmp_path):
    assert run_cli(tmp_path, "--text", GALLERY_TEXTS[0], "--candidates", "2") == 2
    assert run_cli(tmp_path, "--text", GALLERY_TEXTS[0], "--composition", "zigzag") == 3
    assert run_cli(tmp_path, "--text", GALLERY_TEXTS[0], "--archetype", "digital", "--material", "ink",
                   "--palette", "monochrome", "--candidates", "3") == 0
    data = json.loads((tmp_path / "cli.json").read_text(encoding="utf-8"))
    assert data["bird_spec"]["primary_archetype"] == "digital"
    assert data["bird_spec"]["material_type"] == "ink"
    assert data["bird_spec"]["palette_strategy"] == "monochrome"


def test_avoid_recent_similarity_stores_only_summaries(tmp_path, configs):
    recent_path = tmp_path / "recent.json"
    for name in ("a", "b"):
        bg.run_generation([GALLERY_TEXTS[0]], small_options(tmp_path, name, avoid_recent_similarity=True,
                                                            recent_path=recent_path), configs)
    data = json.loads(recent_path.read_text(encoding="utf-8"))
    assert len(data["recent"]) == 2
    raw = recent_path.read_text(encoding="utf-8")
    assert "море" not in raw and GALLERY_TEXTS[0] not in raw


# ---------------------------------------------------------------------------
# Оконный режим «Отпусти»
# ---------------------------------------------------------------------------


def test_gui_generation_logic_without_window(tmp_path, configs):
    import bird_app
    recorder = bg.DrawingRecorder(160)
    history = tmp_path / "history.csv"
    result = bird_app.generate_bird(GALLERY_TEXTS[3], recorder, history,
                                    small_options(tmp_path, show_keywords=False))
    assert result.image_path.is_file() and recorder.frames
    assert bg.read_comments(history) == [GALLERY_TEXTS[3]]
    description = bird_app.describe(result)
    assert description.startswith("Эмоция:") and GALLERY_TEXTS[3] not in description


def test_gui_window_flow(tmp_path, monkeypatch):
    import time
    import tkinter as tk
    import bird_app
    monkeypatch.chdir(tmp_path)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Нет графического окружения")
    try:
        app = bird_app.BirdApp(root)
        assert app.button.cget("text") == "Отпусти"
        app.release()
        assert "10 символов" in app.status.cget("text")
        app.text.insert("1.0", GALLERY_TEXTS[0])
        app.release()
        deadline = time.time() + 60
        while (app.busy or app.result is None) and time.time() < deadline and app.error is None:
            root.update()
            time.sleep(0.01)
        assert app.error is None and app.result is not None
        assert app.photo is not None and app.open_link.cget("text").startswith("Открыть файл")
        assert (tmp_path / "data" / "history" / "requests.csv").is_file()
    finally:
        root.destroy()
