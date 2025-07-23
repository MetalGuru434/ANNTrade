import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cos_model import CosModel

def test_trainable_variable_count():
    model = CosModel()
    assert len(model.trainable_variables) == 2
