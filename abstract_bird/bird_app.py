#!/usr/bin/env python3
"""Оконный режим «Отпусти»: ввод текста и рождение птицы в том же окне.

Запуск::

    python bird_app.py
    python bird_generator.py --gui
    двойной щелчок по «Отпусти.command» (macOS)

Пользователь пишет о том, чего очень хочет или что хочет отпустить, нажимает
«Отпусти» — и справа в том же окне видно, как слой за слоем рисуется птица.
Генерация идёт в фоновом потоке, окно не зависает. Птица, JSON и история
запроса сохраняются так же, как при запуске из командной строки.
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from typing import Any

from PIL import Image, ImageTk

import bird_generator as bg

TITLE = "Отпусти"
PROMPT = ("Напишите о том, чего вы очень хотите — или что хотите отпустить. "
          "Это может быть мечта, чувство, воспоминание или перемена. "
          "Ваши слова превратятся в уникальную птицу.")
BUTTON = "Отпусти"

BACKGROUND = "#f4f1ea"
PANEL = "#ebe6dc"
INK = "#2b2a28"
MUTED = "#6f6a62"
ACCENT = "#3d4a6b"
ACCENT_ACTIVE = "#566389"
DISABLED = "#a7a39b"
ERROR = "#a3362f"
PREVIEW_SIZE = 640
FRAME_MS = 70


def pick_font_family(root: tk.Misc, preferred: tuple[str, ...], fallback: str) -> str:
    """Первая доступная в системе гарнитура из списка (шрифты не скачиваются)."""
    available = set(tkfont.families(root))
    return next((name for name in preferred if name in available), fallback)


def generate_bird(text: str, recorder: bg.DrawingRecorder | None = None,
                  history_path: Path | None = bg.DEFAULT_HISTORY_PATH,
                  options: bg.GenerationOptions | None = None) -> bg.GenerationResult:
    """Сгенерировать птицу по тексту (без интерфейса) — используется окном и тестами."""
    options = options or bg.GenerationOptions()
    options.progress = recorder
    options.input_description = {"type": "gui"}
    result = bg.run_generation([text], options)
    if history_path is not None:
        bg.append_history(history_path, result)
    return result


def describe(result: bg.GenerationResult) -> str:
    """Короткое описание птицы для подписи в окне (без исходного текста)."""
    spec = result.bird_spec
    analysis = result.analysis
    configs = bg.active_configs()
    top = max(analysis.emotions, key=analysis.emotions.get)
    parts = [f"Эмоция: {bg.EMOTION_NAMES_RU[top]}"]
    if spec.primary_archetype:
        arch = configs.archetypes["archetypes"][spec.primary_archetype]["name"]
        comp = configs.compositions["compositions"][spec.composition_type]["name"]
        material = configs.materials["materials"][spec.material_type]["name"]
        parts += [f"{arch}", f"композиция: {comp}", f"материал: {material}"]
        if spec.rare_feature:
            parts.append("редкая особенность: " + configs.archetypes["rare_features"]["features"][spec.rare_feature]["name"])
    elif spec.style:
        parts.append(f"стиль: {bg.STYLES[spec.style].title}")
    if analysis.keywords:
        parts.append("слова: " + ", ".join(analysis.keywords[:3]))
    return " · ".join(parts)


class BirdApp:
    """Окно: слева заголовок, призыв, поле ввода и кнопка; справа птица."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.frames: "queue.Queue[tuple[Image.Image, str]]" = queue.Queue()
        self.result: bg.GenerationResult | None = None
        self.error: str | None = None
        self.worker: threading.Thread | None = None
        self.photo: Any = None
        self.busy = False

        root.title(TITLE)
        root.configure(background=BACKGROUND)
        root.minsize(980, 700)
        title_family = pick_font_family(root, ("Trattatello", "Snell Roundhand", "Palatino", "Georgia"), "Times")
        text_family = pick_font_family(root, ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"), "TkDefaultFont")

        left = tk.Frame(root, background=BACKGROUND)
        left.pack(side="left", fill="both", expand=False, padx=(40, 24), pady=36)
        right = tk.Frame(root, background=PANEL, width=PREVIEW_SIZE + 32, height=PREVIEW_SIZE + 90)
        right.pack(side="right", fill="both", expand=True, padx=(0, 36), pady=36)
        right.pack_propagate(False)

        tk.Label(left, text=TITLE, font=(title_family, 64), foreground=INK, background=BACKGROUND,
                 anchor="w").pack(fill="x")
        tk.Label(left, text=PROMPT, font=(text_family, 16), foreground=MUTED, background=BACKGROUND,
                 justify="left", anchor="w", wraplength=400).pack(fill="x", pady=(8, 20))

        border = tk.Frame(left, background="#cfc8bb", padx=1, pady=1)
        border.pack(fill="both", expand=True)
        self.text = tk.Text(border, height=9, width=38, wrap="word", font=(text_family, 17), foreground=INK,
                            background="#fffdf8", insertbackground=INK, relief="flat", padx=14, pady=12,
                            highlightthickness=0, undo=True)
        self.text.pack(fill="both", expand=True)
        self.text.focus_set()

        self.button = tk.Label(left, text=BUTTON, font=(title_family, 26), foreground="white", background=ACCENT,
                               padx=24, pady=10, cursor="hand2")
        self.button.pack(fill="x", pady=(18, 10))
        self.button.bind("<Button-1>", lambda _event: self.release())
        self.button.bind("<Enter>", lambda _event: self._hover(True))
        self.button.bind("<Leave>", lambda _event: self._hover(False))
        root.bind("<Command-Return>", lambda _event: self.release())
        root.bind("<Control-Return>", lambda _event: self.release())

        self.status = tk.Label(left, text="Cmd/Ctrl + Enter — тоже отпускает", font=(text_family, 13),
                               foreground=MUTED, background=BACKGROUND, justify="left", anchor="w", wraplength=400)
        self.status.pack(fill="x")

        self.canvas = tk.Label(right, background=PANEL, text="Здесь появится ваша птица",
                               font=(title_family, 22), foreground=MUTED)
        self.canvas.pack(fill="both", expand=True, padx=16, pady=(16, 6))
        self.caption = tk.Label(right, text="", font=(text_family, 13), foreground=INK, background=PANEL,
                                wraplength=PREVIEW_SIZE, justify="center")
        self.caption.pack(fill="x", padx=16)
        self.open_link = tk.Label(right, text="", font=(text_family, 13, "underline"), foreground=ACCENT,
                                  background=PANEL, cursor="hand2")
        self.open_link.pack(pady=(4, 14))
        self.open_link.bind("<Button-1>", lambda _event: self._open_file())

    # --- взаимодействие ------------------------------------------------------

    def _hover(self, inside: bool) -> None:
        if not self.busy:
            self.button.configure(background=ACCENT_ACTIVE if inside else ACCENT)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.button.configure(background=DISABLED if busy else ACCENT, text="Птица рождается…" if busy else BUTTON,
                              cursor="watch" if busy else "hand2")
        self.text.configure(state="disabled" if busy else "normal")

    def release(self) -> None:
        """Нажатие «Отпусти»: проверить текст и запустить генерацию в фоне."""
        if self.busy:
            return
        text = self.text.get("1.0", "end").strip()
        if len(text) < bg.MIN_TOTAL_CHARACTERS:
            self.status.configure(text="Напишите хотя бы несколько слов — не меньше 10 символов.", foreground=ERROR)
            return
        self.status.configure(text="Птица рождается слой за слоем…", foreground=MUTED)
        self.caption.configure(text="")
        self.open_link.configure(text="")
        self.result, self.error = None, None
        self.frames = queue.Queue()
        self._set_busy(True)
        recorder = _QueueRecorder(self.frames, PREVIEW_SIZE)
        self.worker = threading.Thread(target=self._work, args=(text, recorder), daemon=True)
        self.worker.start()
        self.root.after(FRAME_MS, self._animate)

    def _work(self, text: str, recorder: "_QueueRecorder") -> None:
        try:
            self.result = generate_bird(text, recorder)
        except bg.BirdGeneratorError as exc:
            self.error = str(exc)
        except Exception as exc:  # noqa: BLE001 — окно не должно падать
            self.error = f"Непредвиденная ошибка ({type(exc).__name__})"

    def _show(self, image: Image.Image) -> None:
        image = image.copy()
        image.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        self.canvas.configure(image=self.photo, text="")

    def _animate(self) -> None:
        """Показывать кадры по мере рисования, затем итоговый PNG."""
        try:
            frame, label = self.frames.get_nowait()
            self._show(frame)
            self.status.configure(text=f"{label}…", foreground=MUTED)
            self.root.after(FRAME_MS, self._animate)
            return
        except queue.Empty:
            pass
        if self.worker is not None and self.worker.is_alive():
            self.root.after(FRAME_MS, self._animate)
            return
        self._set_busy(False)
        if self.error:
            self.status.configure(text=self.error, foreground=ERROR)
            return
        if self.result is not None:
            with Image.open(self.result.image_path) as final:
                self._show(final.convert("RGB"))
            self.status.configure(text="Готово. Можно написать ещё — каждая птица сохраняется отдельно.", foreground=MUTED)
            self.caption.configure(text=describe(self.result))
            self.open_link.configure(text=f"Открыть файл: {self.result.image_path.name}")

    def _open_file(self) -> None:
        if self.result is not None:
            bg.open_with_system_viewer(self.result.image_path)


class _QueueRecorder:
    """Передаёт уменьшенные кадры рисования из фонового потока в окно."""

    def __init__(self, target: "queue.Queue[tuple[Image.Image, str]]", max_side: int) -> None:
        self.target = target
        self.recorder = bg.DrawingRecorder(max_side)

    def __call__(self, image: Image.Image, label: str) -> None:
        self.recorder(image, label)
        self.target.put(self.recorder.frames[-1])


def main() -> int:
    """Открыть окно «Отпусти». Возвращает код завершения."""
    bg.configure_logging(debug=False)
    try:
        bg.load_configs()
    except bg.BirdGeneratorError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return exc.exit_code
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Не удалось открыть окно (нет графического окружения?): {exc}", file=sys.stderr)
        return 1
    BirdApp(root)
    root.lift()
    root.attributes("-topmost", True)
    root.after(600, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
