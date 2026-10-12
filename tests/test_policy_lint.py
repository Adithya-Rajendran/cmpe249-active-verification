"""Static lint for verify/policies.py (README section 2, "Guarantee and its boundary").

Runtime sealing stops public and single-underscore access to evidence, but in-process Python
cannot stop deliberate introspection (frames, gc, closures, mangled names) or a policy opening
files. Policy code must therefore not contain any of it; this test enforces that on the source.
"""
import ast
import unittest
from pathlib import Path

POLICIES = Path(__file__).resolve().parent.parent / "verify" / "policies.py"

ALLOWED_IMPORTS = {"__future__", "numbers", "collections.abc", "typing", "numpy",
                   "verify.schema", "verify.replay", "verify.stopping"}
BANNED_NAMES = {"open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr", "vars",
                "globals", "locals", "breakpoint", "input"}
BANNED_ATTRS = {"__closure__", "__globals__", "__code__", "__dict__", "__self__", "__wrapped__",
                "_getframe", "f_locals", "f_back", "f_globals", "gi_frame", "cell_contents",
                "get_objects", "get_referrers", "get_referents"}


def violations(source: str) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out += [f"import {a.name}" for a in node.names if a.name not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORTS:
                out.append(f"from {node.module} import")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            out.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS or (node.attr.startswith("_") and "__" in node.attr[1:]
                                             and not node.attr.endswith("__")):
                out.append(f"attribute .{node.attr}")
            # private (single-underscore) access on anything except self
            elif (node.attr.startswith("_") and not node.attr.startswith("__")
                  and not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"))):
                out.append(f"private attribute .{node.attr}")
    return out


class TestPolicyLint(unittest.TestCase):
    def test_policies_module_is_clean(self):
        self.assertEqual(violations(POLICIES.read_text()), [])

    def test_lint_catches_known_exploits(self):
        # the reviewer exploits (frame walk, gc, closure peek, mangled slot, private store data, file read)
        for src in ("import sys\nsys._getframe(1).f_locals['env']",
                    "import gc\ngc.get_objects()",
                    "env._ReplayEnv__reveal.__closure__[0].cell_contents",
                    "store._data['x']",
                    "open('runs/x/ground_truth.json')",
                    "getattr(env, '_ReplayEnv__reveal')",
                    "from pathlib import Path",
                    "import inspect"):
            with self.subTest(src=src):
                self.assertTrue(violations(src))


if __name__ == "__main__":
    unittest.main()
