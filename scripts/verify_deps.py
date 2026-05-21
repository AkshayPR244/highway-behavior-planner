"""
Smoke test: verifies all core dependencies import correctly and that
MPS (Apple Silicon GPU) is available for PyTorch.
Run with: python scripts/verify_deps.py
"""
import sys


def check(label: str, fn):
    try:
        result = fn()
        print(f"  [OK]  {label}" + (f" — {result}" if result else ""))
    except Exception as e:
        print(f"  [FAIL] {label} — {e}")


print(f"\nPython {sys.version}\n")
print("Core dependencies:")

check("torch", lambda: __import__("torch").__version__)
check("torch MPS available", lambda: (
    "yes" if __import__("torch").backends.mps.is_available() else "no (CPU only)"
))
check("gymnasium", lambda: __import__("gymnasium").__version__)
check("highway_env", lambda: __import__("highway_env") and "ok")
check("stable_baselines3", lambda: __import__("stable_baselines3").__version__)
check("imitation", lambda: __import__("imitation").__version__)
check("numpy", lambda: __import__("numpy").__version__)
check("matplotlib", lambda: __import__("matplotlib").__version__)

print("\nhighway-env sanity check:")
check("make highway-v0", lambda: (
    __import__("gymnasium").make("highway-v0").reset() and "env created ok"
))

print()
