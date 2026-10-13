"""Load the FlexiBLAS SYCL hook outside FlexiBLAS's own initialization and call its init.

Separates "the hook DLL cannot initialize" from "loading it inside FlexiBLAS's DllMain fails".
Usage: python load_hook.py <hook.dll> [<dll search dir> ...]
"""

import ctypes
import os
import sys


def main() -> int:
    hook = sys.argv[1]
    for d in sys.argv[2:]:
        os.add_dll_directory(d)
    # The hook loads the engine with plain LoadLibrary, which searches PATH.
    os.environ["PATH"] = os.pathsep.join(sys.argv[2:] + [os.environ.get("PATH", "")])
    os.environ.pop("FLEXIBLAS_HOOK", None)
    lib = ctypes.CDLL(hook)
    print("loaded", hook)
    lib.__flexiblas_hook_initialize()
    print("initialize returned")
    lib.__flexiblas_hook_finalize()
    print("finalize returned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
