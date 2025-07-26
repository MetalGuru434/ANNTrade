import os
import sys

# Support running in environments where ``__file__`` may not be defined, such
# as interactive notebooks. Fallback to the current working directory in that
# case so imports from the project still resolve correctly.
base_dir = os.path.dirname(os.path.dirname(
    globals().get('__file__', os.getcwd())
))
sys.path.insert(0, base_dir)
from cos_model import CosModel

def test_trainable_variable_count():
    model = CosModel()
    assert len(model.trainable_variables) == 2
