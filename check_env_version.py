import jax, jaxlib
from jaxlib import version as jv

print("jax =", jax.__version__)
print("jaxlib =", jaxlib.__version__)
print("default_backend =", jax.default_backend())  # 期望是 'gpu' 或 'cuda'
print("devices =", jax.devices())
print("platform_version =", jax.lib.xla_bridge.get_backend().platform_version)  # 这里通常会带 CUDA/Driver/NCCL 版本

# jaxlib 内置的编译期版本信息（字段名随版本可能不同，做容错）
print("jaxlib.cuda =", getattr(jv, "cuda", None) or getattr(jv, "cuda_version", None))
print("jaxlib.cudnn =", getattr(jv, "cudnn", None) or getattr(jv, "cudnn_version", None))
print("jaxlib.nccl =", getattr(jv, "nccl_version", None))