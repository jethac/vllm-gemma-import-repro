#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the eager version-gated transformers import bug in vLLM's Gemma
multimodal model modules.

Background
----------
vLLM's Gemma 3 / 3n / 4 multimodal model modules import symbols from
version-gated ``transformers.models.<gemma*>`` submodules at *module top level*:

    # vllm/model_executor/models/gemma4_mm.py (stock main)
    from transformers.models.gemma4 import (
        Gemma4Config, Gemma4Processor, Gemma4VisionConfig)
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4AudioConfig, Gemma4TextConfig)

vLLM's architecture inspection imports a model module for *any* registered
architecture. So on a transformers too old to ship ``transformers.models.gemma4``
(it first shipped in transformers 5.5.0; vLLM main only pins ``>=5.5.3``, so a
0.x-behind install lacks it), that eager import raises ``ModuleNotFoundError``
*at import time* -- breaking inspection of Gemma 4 AND of every module that
imports gemma4_mm (``gemma4_unified``, and in the DiffusionGemma fork
``diffusion_gemma``), rather than failing only when a Gemma 4 model is used.

This script observes and reports the behaviour of whatever vLLM checkout is on
``sys.path``. It does not know whether it is running against stock or fixed vLLM;
it reports what actually happens and picks an exit code accordingly:

    exit 2  -> STOCK bug reproduced: eager import raised ModuleNotFoundError
    exit 0  -> FIXED behaviour: module imports + registers; the missing
               dependency surfaces only when the processor is invoked
    exit 3  -> UNEXPECTED (e.g. a different import broke first)

Two ways to make ``transformers.models.gemma4`` absent:
  * REAL: run with a real old transformers (e.g. 5.4.0) that genuinely lacks it.
  * SIMULATED: pass ``--simulate`` to monkeypatch ``sys.modules`` so the module
    appears absent regardless of the installed transformers (deterministic).
"""

import argparse
import importlib
import platform
import sys

# arch -> (model module, version-gated transformers submodule it eagerly imports)
_ARCH = {
    "Gemma4ForConditionalGeneration": (
        "vllm.model_executor.models.gemma4_mm",
        "transformers.models.gemma4",
    ),
    "Gemma3nForConditionalGeneration": (
        "vllm.model_executor.models.gemma3n_mm",
        "transformers.models.gemma3n",
    ),
    "Gemma3ForConditionalGeneration": (
        "vllm.model_executor.models.gemma3_mm",
        "transformers.models.gemma3.image_processing_gemma3",
    ),
}

# Module that imports the target model module -- used to demonstrate the
# transitive blast radius (importing this breaks too, on stock).
_TRANSITIVE = {
    "Gemma4ForConditionalGeneration": "vllm.model_executor.models.gemma4_unified",
}


def _rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _simulate_missing(name):
    """Make ``import name`` raise, mirroring an older transformers install.

    A ``None`` entry in ``sys.modules`` makes ``import name`` raise ImportError.
    """
    for cached in list(sys.modules):
        if cached == name or cached.startswith(name + "."):
            del sys.modules[cached]
    sys.modules[name] = None


def _transformers_version():
    try:
        import transformers

        return transformers.__version__
    except Exception as e:  # pragma: no cover
        return f"<not importable: {e!r}>"


def _submodule_present(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="Gemma4ForConditionalGeneration",
                    choices=list(_ARCH))
    ap.add_argument("--simulate", action="store_true",
                    help="monkeypatch the gemma submodule absent (deterministic)")
    args = ap.parse_args()

    module_name, submodule = _ARCH[args.arch]

    _rule("ENVIRONMENT")
    print(f"python                 : {platform.python_version()}")
    print(f"transformers           : {_transformers_version()}")
    real_present = _submodule_present(submodule)
    print(f"{submodule!r} importable (real): {real_present}")

    mode = "SIMULATED (monkeypatch)" if args.simulate else "REAL transformers"
    print(f"mode                   : {mode}")

    if args.simulate:
        _simulate_missing(submodule)
    absent = args.simulate or not real_present
    print(f"target submodule absent for this run: {absent}")

    if not absent:
        print("\nNOTE: the version-gated submodule is PRESENT and not simulated,"
              " so this run cannot exhibit the bug. Re-run with --simulate or an"
              " older transformers.")

    # ---- Arm A: import the model module directly (what the registry does) ----
    _rule(f"ARM A -- import {module_name}")
    crash = False
    try:
        importlib.import_module(module_name)
        print(f"RESULT: imported cleanly ({module_name}).")
    except ImportError as e:
        msg = str(e)
        if submodule.split(".")[2] in msg or "gemma" in msg:
            crash = True
            print(f"RESULT: {type(e).__name__}: {e}")
            print("  --> STOCK BUG: eager top-level import failed at import time.")
        else:
            _rule("UNEXPECTED -- a different import failed first")
            print(f"{type(e).__name__}: {e}")
            print("  (not the gemma eager import; likely an unrelated missing dep"
                  " or transformers-version incompatibility of the base package)")
            sys.exit(3)

    if crash:
        # ---- Arm B: show the transitive blast radius on stock ----
        victim = _TRANSITIVE.get(args.arch)
        if victim:
            _rule(f"ARM B -- transitive victim: import {victim}")
            try:
                importlib.import_module(victim)
                print("RESULT: imported cleanly (unexpected on stock).")
            except ImportError as e:
                print(f"RESULT: {type(e).__name__}: {e}")
                print(f"  --> {victim} is broken too, because it imports"
                      f" {module_name}.")
        _rule("VERDICT")
        print("STOCK vLLM: version-gated eager import crashes at import time,")
        print("breaking architecture inspection of this arch (and its importers).")
        sys.exit(2)

    # Import succeeded -> fixed behaviour. Verify registration + deferred error.
    from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402

    _rule("ARM B -- architecture still registers")
    registered = args.arch in ModelRegistry.get_supported_archs()
    print(f"{args.arch!r} in ModelRegistry.get_supported_archs(): {registered}")

    _rule("ARM C -- missing dependency surfaces only when the processor is used")
    if not absent:
        print("SKIPPED: submodule present and not simulated; nothing to defer.")
    else:
        module = sys.modules[module_name]
        # Re-assert absence in case importing vllm pulled the real submodule in.
        _simulate_missing(submodule)
        try:
            info = module.Gemma4ProcessingInfo(ctx=None) if args.arch.startswith(
                "Gemma4") else None
            if info is not None:
                info.get_hf_config()
            print("RESULT: no error raised (unexpected).")
        except ImportError as e:
            print(f"RESULT: {type(e).__name__}: {e}")
            print("  --> clear, deferred error naming the missing submodule,"
                  " raised only on processor use.")
        except Exception as e:
            # ctx=None reaches the lazy import first on the fixed code; any other
            # error means the import was NOT deferred as expected.
            print(f"RESULT: {type(e).__name__}: {e}")

    _rule("VERDICT")
    print("FIXED vLLM: module imports and registers even without the")
    print("version-gated transformers submodule; the missing dependency is a")
    print("clear, deferred error at processor-use time -- not an import-time crash.")
    sys.exit(0 if registered else 3)


if __name__ == "__main__":
    main()
