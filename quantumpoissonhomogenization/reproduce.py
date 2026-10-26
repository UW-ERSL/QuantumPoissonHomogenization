"""reproduce -- the command line of the paper scripts (makeTables.py, makeFigures.py) and the check of
a code listing against the output printed in the paper.

    dispatch(sys.argv[1:], {"tables": TABLES, "checks": CHECKS, "listings": LISTINGS})

runs every entry of the first group when no name is given; a group name runs that group, any other
name runs that entry, and --list prints the names. Each entry is (function, description).

    run_listing(name, code, expected)

runs `code`, the listing exactly as it appears in the paper, in a fresh namespace with its output
captured, prints the output, and raises AssertionError if it differs from `expected`, the text of
the paper's `----- output --------------` block.
"""
import contextlib
import io
import time


def dispatch(argv, groups):
    entries = {k: v for g in groups.values() for k, v in g.items()}
    if "--list" in argv:
        for g, items in groups.items():
            print(f"{g}:")
            for k, (_, d) in items.items():
                print(f"  {k:<14} {d}")
        return
    wanted = []
    for a in [a for a in argv if not a.startswith("-")] or [next(iter(groups))]:
        if a in groups:
            wanted += list(groups[a])
        elif a in entries:
            wanted.append(a)
        else:
            raise SystemExit(f"unknown name {a!r}; choose from {list(groups)} or {list(entries)}")
    for name in wanted:
        fn, desc = entries[name]
        print(f"{name}  ({desc})", flush=True)
        fn()
    print("\ndone")


def run_listing(name, code, expected):
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, f"<{name}>", "exec"), {"__name__": f"listing_{name}"})
    out = buf.getvalue().rstrip("\n")
    print(f"----- output --------------   [{time.time() - t0:.0f} s]")
    print(out)
    assert out == expected.strip("\n"), f"{name}: output differs from the paper"
    print("    identical to the output printed in the paper")
