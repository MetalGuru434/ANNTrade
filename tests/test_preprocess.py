import os
import sys
import pytest

# Support execution in interactive environments where ``__file__`` may not be
# available by default. When absent we fall back to the current working
# directory to locate the project root.
base_dir = os.path.dirname(os.path.dirname(
    globals().get('__file__', os.getcwd())
))
sys.path.insert(0, base_dir)
from preprocess import clean_text


def test_clean_text_simple():
    assert clean_text("Текст 😊") == "текст"
