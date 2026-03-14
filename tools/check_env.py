"""
check_env.py — verify all MCA dependencies are installed and working.
Run with: python check_env.py
"""

import sys

PASS = "OK  "
FAIL = "FAIL"

def check(name, fn):
    try:
        result = fn()
        print(f"  [{PASS}] {name:<20} {result}")
        return True
    except Exception as e:
        print(f"  [{FAIL}] {name:<20} {e}")
        return False

print(f"\nPython {sys.version}")
print("=" * 60)
print("Core")
print("=" * 60)

results = []

results.append(check("torch", lambda: (
    __import__("torch").__version__ +
    f"  CUDA={__import__('torch').cuda.is_available()}" +
    f"  devices={__import__('torch').cuda.device_count()}"
)))
results.append(check("numpy",    lambda: __import__("numpy").__version__))
results.append(check("h5py",     lambda: __import__("h5py").__version__))

print()
print("=" * 60)
print("OpenMMLab")
print("=" * 60)

results.append(check("mmcv",      lambda: __import__("mmcv").__version__))
results.append(check("mmengine",  lambda: __import__("mmengine").__version__))
results.append(check("mmselfsup", lambda: __import__("mmselfsup").__version__))

print()
print("=" * 60)
print("ML / analysis")
print("=" * 60)

results.append(check("sklearn",   lambda: __import__("sklearn").__version__))
results.append(check("timm",      lambda: __import__("timm").__version__))
results.append(check("umap",      lambda: __import__("umap").__version__))
results.append(check("matplotlib",lambda: __import__("matplotlib").__version__))

print()
print("=" * 60)
print("External model deps")
print("=" * 60)

results.append(check("transformers",    lambda: __import__("transformers").__version__))
results.append(check("huggingface_hub", lambda: __import__("huggingface_hub").__version__))

print()
print("=" * 60)
print("GPU details")
print("=" * 60)

import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name}  {mem_gb:.1f} GB  (CUDA {torch.version.cuda})")
else:
    print("  No GPU available")

print()
n_fail = results.count(False)
if n_fail == 0:
    print("All checks passed.")
else:
    print(f"{n_fail} check(s) failed — install missing packages before running experiments.")
print()
