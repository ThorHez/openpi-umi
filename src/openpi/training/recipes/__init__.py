"""Recipe registries grouped by training method.

Load shared definitions before a recipe submodule, including direct imports of
``recipes.sft``. The facade imports these lists only after its shared types and
model objects are initialized; their original module identities are preserved.
"""

from openpi.training import config as _config  # noqa: F401
