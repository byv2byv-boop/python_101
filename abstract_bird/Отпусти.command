#!/bin/zsh
# Двойной щелчок в Finder открывает окно «Отпусти».
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  .venv/bin/python bird_app.py
else
  python3 bird_app.py
fi
