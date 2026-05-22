import warnings

warnings.warn(
    "pytorchwildlife has been renamed to sparrow-engine; this alias will be removed in 0.2.0",
    DeprecationWarning,
    stacklevel=2,
)

from sparrow_engine import *  # noqa: F401, F403, E402

# Shim version, kept distinct from the underlying sparrow_engine `__version__`.
# Defined after the wildcard import so a future sparrow_engine `__version__` re-export
# can't shadow the shim's identity. `from foo import *` skips dunder names
# anyway, but the explicit assignment is defensive and surfaces in
# `pytorchwildlife.__version__` checks downstream.
__version__ = "0.1.0"
