"""The one place umap-learn is imported from.

umap-learn unconditionally imports its optional ParametricUMAP submodule,
which unconditionally does `import tensorflow`. On machines where the
installed tensorflow hangs on import (see embedding_pipeline.py for the
same issue with transformers), the stub below makes that import resolve
instantly instead of hanging — umap/__init__.py already falls back to a
dummy ParametricUMAP class when tensorflow isn't usable, we just need the
import itself to not hang. It must run before `import umap`, so every
module goes through here."""

import importlib.util
import sys
import types

if "tensorflow" not in sys.modules:
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.__spec__ = importlib.util.spec_from_loader("tensorflow", loader=None)  # torch's dynamic import needs a spec
    sys.modules["tensorflow"] = tf_stub

import umap  # noqa: E402
from umap.layouts import optimize_layout_euclidean  # noqa: E402
from umap.umap_ import (  # noqa: E402
    compute_membership_strengths,
    find_ab_params,
    make_epochs_per_sample,
    smooth_knn_dist,
)

__all__ = [
    "umap", "optimize_layout_euclidean", "compute_membership_strengths",
    "find_ab_params", "make_epochs_per_sample", "smooth_knn_dist",
]
