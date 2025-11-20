"""
iMDBN model components: RBM, iDBN, and iMDBN.
"""

from .gdbn_model_complete import RBM, iDBN, iMDBN

__all__ = ["RBM", "iDBN", "iMDBN"]

# Compatibility aliases for loading old pickled models from Groundeep
# Old models were saved with src.classes.* module paths
import sys
from types import ModuleType

# Create fake src module hierarchy
_src = ModuleType('src')
_src_classes = ModuleType('src.classes')

# Map old module names to new classes
_src_classes.rbm_model = sys.modules[__name__]
_src_classes.dbn_model = sys.modules[__name__]
_src_classes.gdbn_model = sys.modules[__name__]

# Also need the class names accessible directly
_src_classes.rbm_model.RBM = RBM
_src_classes.dbn_model.iDBN = iDBN
_src_classes.gdbn_model.iMDBN = iMDBN

_src.classes = _src_classes

sys.modules['src'] = _src
sys.modules['src.classes'] = _src_classes
sys.modules['src.classes.rbm_model'] = _src_classes.rbm_model
sys.modules['src.classes.dbn_model'] = _src_classes.dbn_model
sys.modules['src.classes.gdbn_model'] = _src_classes.gdbn_model
