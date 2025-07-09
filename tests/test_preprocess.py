import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from preprocess import clean_text


def test_clean_text_simple():
    assert clean_text("Текст 😊") == "текст"
