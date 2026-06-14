"""进程级环境修正——必须在导入 torch / onnxruntime / fastembed 之前执行。

Windows 上 torch(Intel OpenMP: libiomp5md) 与 onnxruntime(LLVM OpenMP: libomp)
会因重复加载 OpenMP 运行时报 "OMP: Error #15" 并中止进程。设此变量放行。

用法：入口脚本把 `import _bootstrap  # noqa: F401` 放在所有其它 import 之前。
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
