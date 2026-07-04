"""AutoRAG local setup verification script.

Checks that Ollama and AutoRAG are installed and working.
"""

import sys

import httpx


OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4:e2b"


def check_ollama() -> bool:
    print("=" * 60)
    print("Step 1: Checking Ollama")
    print("=" * 60)
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        model_names = [m["name"] for m in models]
        print(f"  Ollama is running. Models available: {len(models)}")
        for name in model_names:
            print(f"    - {name}")

        has_model = any(MODEL.split(":")[0] in n for n in model_names)
        if not has_model:
            print(f"\n  WARNING: {MODEL} not found!")
            print(f"  Run: ollama pull {MODEL}")
            return False

        print(f"  {MODEL} is available.")
        return True
    except Exception as e:
        print(f"  FAILED: Ollama is not running at {OLLAMA_URL}")
        print(f"  Error: {e}")
        print("  Start Ollama first: ollama serve")
        return False


def check_ollama_inference() -> bool:
    print()
    print("=" * 60)
    print("Step 2: Testing Ollama inference")
    print("=" * 60)
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": "Say hello in one sentence.",
                "stream": False,
                "options": {"num_predict": 50},
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")
        print(f"  Response: {text.strip()}")
        print("  Ollama inference is working!")
        return True
    except Exception as e:
        print(f"  FAILED: Ollama inference test failed")
        print(f"  Error: {e}")
        return False


def check_autorag() -> bool:
    print()
    print("=" * 60)
    print("Step 3: Checking AutoRAG installation")
    print("=" * 60)
    try:
        import autorag

        version = getattr(autorag, "__version__", "unknown")
        print(f"  AutoRAG is installed (version: {version})")
        return True
    except ImportError:
        print("  FAILED: AutoRAG is not installed")
        print("  Run: uv add autorag")
        return False


def check_data_deps() -> bool:
    print()
    print("=" * 60)
    print("Step 4: Checking data dependencies")
    print("=" * 60)
    all_ok = True

    try:
        import pandas as pd

        print(f"  pandas: {pd.__version__}")
    except ImportError:
        print("  FAILED: pandas not installed (uv add pandas)")
        all_ok = False

    try:
        import pyarrow

        print(f"  pyarrow: {pyarrow.__version__}")
    except ImportError:
        print("  WARNING: pyarrow not installed — needed for Parquet files")
        print("  Run: uv add pyarrow")
        all_ok = False

    if all_ok:
        print("  Data dependencies are ready.")

    return all_ok


def main() -> None:
    print()
    print("AutoRAG Local Setup Verification")
    print("=" * 60)
    print()

    results = {}
    results["Ollama"] = check_ollama()

    if results["Ollama"]:
        results["Inference"] = check_ollama_inference()
    else:
        results["Inference"] = False
        print()
        print("  Skipping inference test (Ollama not ready)")

    results["AutoRAG"] = check_autorag()
    results["Data deps"] = check_data_deps()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:12s} : {status}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All checks passed! AutoRAG is ready for tutorial lessons.")
    else:
        print("Some checks failed. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
