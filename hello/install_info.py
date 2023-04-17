from pathlib import Path

def get_libpath():
    import os, sys, platform
    join = os.path.join
    exists = os.path.exists
    dirname = os.path.dirname
    cn_path = dirname(dirname(__file__))
    so_ext = {
        "": "",
        "Java": ".jar",
        "Linux": ".so",
        "Darwin": ".dylib",
        "Windows": ".dll"
    }[platform.system()]

    def find_lib(libdir):
        target = f"libhello{so_ext}*"
        search_path = Path(libdir)
        matches = [m for m in search_path.rglob(target)]
        if matches:
          return matches[0].parent
        return None

    return (
        find_lib("/home/nfs/rorym/legate-hello-world/build/legate_hello") or
        find_lib(join(dirname(dirname(dirname(cn_path))), "lib")) or
        find_lib(join(dirname(dirname(sys.executable)), "lib")) or
        ""
    )

libpath: str = get_libpath()

header: str = """
  enum HelloOpCode { _OP_CODE_BASE = 0, HELLO_WORLD = 1, SUM = 2, SQUARE = 3, IOTA = 4, QUANTILE = 5, QUANTILE_REDUCE = 6, QUANTILE_OUTPUT = 7, QUANTISE_DATA = 8 };

  void hello_perform_registration();
"""
