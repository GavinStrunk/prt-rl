from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


# # ---------- templates ----------
# CONFIG_TMPL = """\
# from __future__ import annotations

# from dataclasses import dataclass


# @dataclass
# class {ClassName}Config:
#     \"\"\"Configuration for {ClassName}.\"\"\"

#     learning_rate: float = 3e-4
#     gamma: float = 0.99
# """

# AGENT_TMPL = """\
# from __future__ import annotations

# from typing import Optional

# import torch

# from .config import {ClassName}Config
# from prt_rl.agent import BaseAgent


# class {ClassName}(BaseAgent):
#     \"\"\"{ClassName} agent.\"\"\"

#     def __init__(self, config: {ClassName}Config):
#         super().__init__()
#         self.config = config

#     def collect(self) -> None:
#         raise NotImplementedError

#     def update(self) -> None:
#         raise NotImplementedError

#     def evaluate(self) -> None:
#         raise NotImplementedError
# """

# INIT_TMPL = """\
# from .config import {ClassName}Config
# from .agent import {ClassName}

# __all__ = ["{ClassName}", "{ClassName}Config"]
# """

# TEST_TMPL = """\
# def test_import():
#     from prt_rl.algorithms.{pkg_name} import {ClassName}, {ClassName}Config  # noqa: F401
# """


# # ---------- helpers ----------
# _SNAKE_RE = re.compile(r"[^a-z0-9]+")

# def to_snake(name: str) -> str:
#     s = name.strip()
#     s = s.replace(" ", "_")
#     s = s.replace("-", "_")
#     s = s.lower()
#     s = _SNAKE_RE.sub("_", s)
#     s = s.strip("_")
#     if not s:
#         raise ValueError("Name produced an empty package name.")
#     return s

# def to_pascal(name: str) -> str:
#     parts = re.split(r"[\s_\-]+", name.strip())
#     parts = [p for p in parts if p]
#     if not parts:
#         raise ValueError("Name produced an empty class name.")
#     return "".join(p[:1].upper() + p[1:] for p in parts)

# def write_file(path: Path, content: str, overwrite: bool) -> None:
#     if path.exists() and not overwrite:
#         raise FileExistsError(f"Refusing to overwrite existing file: {path}")
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(content, encoding="utf-8")


# @dataclass(frozen=True)
# class ScaffoldPlan:
#     repo_root: Path
#     algo_dir: Path
#     tests_dir: Path
#     pkg_name: str
#     class_name: str


# def build_plan(repo_root: Path, name: str) -> ScaffoldPlan:
#     pkg_name = to_snake(name)
#     class_name = to_pascal(name)

#     # Adjust these paths to match your repo layout.
#     # Here we assume src/ layout and algorithms live at prt_rl/algorithms/<algo>/
#     algo_dir = repo_root / "src" / "prt_rl" / "algorithms" / pkg_name
#     tests_dir = repo_root / "tests" / "algorithms" / pkg_name

#     return ScaffoldPlan(
#         repo_root=repo_root,
#         algo_dir=algo_dir,
#         tests_dir=tests_dir,
#         pkg_name=pkg_name,
#         class_name=class_name,
#     )


# def scaffold(plan: ScaffoldPlan, overwrite: bool) -> None:
#     mapping = {"ClassName": plan.class_name, "pkg_name": plan.pkg_name}

#     write_file(plan.algo_dir / "__init__.py", INIT_TMPL.format(**mapping), overwrite)
#     write_file(plan.algo_dir / "config.py", CONFIG_TMPL.format(**mapping), overwrite)
#     write_file(plan.algo_dir / "agent.py", AGENT_TMPL.format(**mapping), overwrite)
#     write_file(plan.tests_dir / "test_import.py", TEST_TMPL.format(**mapping), overwrite)


def find_repo_root(start: Path) -> Path:
    # Very simple heuristic: walk up until we find pyproject.toml
    cur = start.resolve()
    for _ in range(20):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("Could not find repo root (pyproject.toml not found).")


def main() -> None:
    print("hello")
    # p = argparse.ArgumentParser(prog="prt-scaffold")
    # sub = p.add_subparsers(dest="cmd", required=True)

    # algo = sub.add_parser("algorithm", help="Create a new algorithm scaffold")
    # algo.add_argument("name", help="Algorithm name (e.g., PPO, td3, 'my algo')")
    # algo.add_argument("--repo-root", default=None, help="Override repo root")
    # algo.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    # args = p.parse_args()

    # repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(Path.cwd())

    # if args.cmd == "algorithm":
        # plan = build_plan(repo_root, args.name)
        # scaffold(plan, overwrite=args.overwrite)

        # print(f"Created algorithm scaffold:")
        # print(f"  package: prt_rl.algorithms.{plan.pkg_name}")
        # print(f"  path:    {plan.algo_dir}")
        # print(f"  tests:   {plan.tests_dir}")
        # return 0

    # return 1


