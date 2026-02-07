from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List


def _to_snake(name: str) -> str:
    """
    Convert a class or algorithm name to snake_case.

    Rules:
      - Preserve acronyms with digits: A3C -> a3c, TD3 -> td3
      - Preserve all-caps acronyms: PPO -> ppo, SAC -> sac
      - Convert CamelCase: SoftActorCritic -> soft_actor_critic
    """
    # If the name is all caps / digits (acronym), just lowercase it
    if name.isupper():
        return name.lower()

    # Handle CamelCase -> snake_case
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("-", "_").lower()


def _render_single_file_template(algo_pascal: str) -> str:
    """
    Returns a single-file algorithm scaffold containing:
      - Config dataclass
      - HeadSpec + PolicySpec dataclasses
      - PolicyModule implementation
      - PolicyFactory implementation
      - Agent implementation with save/load skeleton

    Minimal dependencies: dataclasses, json, pathlib, torch, numpy (optional)
    and your prt_rl primitives (BaseAgent, EnvParams, PolicyModule, heads, etc.)
    """
    return f'''
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import Optional, List, Literal, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.policies as pmod


# ----------------------------
# 1) Config
# ----------------------------

@dataclass
class {algo_pascal}Config:
    """
    Configuration for the {algo_pascal} agent.

    Add algorithm hyperparameters here.
    """
    # Example:
    # learning_rate: float = 3e-4
    pass


# ----------------------------
# 2) Policy specs
# ----------------------------
@dataclass
class {algo_pascal}PolicySpec:
    """
    Describes how to build a {algo_pascal}-compliant policy.
    """
    pass


# ----------------------------
# 3) Policy
# ----------------------------
class {algo_pascal}Policy(pmod.PolicyModule):
    def __init__(
        self,
        *,
        backbone: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, pmod.InfoDict]:
        return None, {{}}

    def forward(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        action, _ = self.act(obs, deterministic=deterministic)
        return action


# ----------------------------
# 4) Factory
# ----------------------------

class {algo_pascal}PolicyFactory(pmod.PolicyFactory[{algo_pascal}PolicySpec, {algo_pascal}Policy]):
    """
    Builds and serializes {algo_pascal}Policy from (EnvParams, {algo_pascal}PolicySpec).
    """

    def make(self, env_params: EnvParams, spec: {algo_pascal}PolicySpec) -> {algo_pascal}Policy:
        return None


    def save(self, env_params: EnvParams, spec: {algo_pascal}PolicySpec, policy: {algo_pascal}Policy, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        payload = {{
            "env_params": asdict(env_params),
            "spec": asdict(spec),
            "format_version": 1,
        }}
        (p / "spec.json").write_text(json.dumps(payload, indent=2))
        torch.save(policy.state_dict(), p / "weights.pt")

    def load(
        self,
        path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> Tuple[EnvParams, {algo_pascal}PolicySpec, {algo_pascal}Policy]:
        p = Path(path)
        payload = json.loads((p / "spec.json").read_text())
        env_params = EnvParams(**payload["env_params"])
        spec = {algo_pascal}PolicySpec(**payload["spec"])
        policy = self.make(env_params, spec)
        sd = torch.load(p / "weights.pt", map_location=map_location)
        policy.load_state_dict(sd, strict=strict)
        return env_params, spec, policy


# ----------------------------
# 5) Agent
# ----------------------------
class {algo_pascal}Agent(BaseAgent):
    def __init__(
        self,
        env_params: EnvParams,
        policy_spec: {algo_pascal}PolicySpec,
        *,
        config: {algo_pascal}Config = {algo_pascal}Config(),
        device: str = "cpu",
    ) -> None:
        self.env_params = env_params
        self.policy_spec = policy_spec
        self.config = config

        policy = {algo_pascal}PolicyFactory().make(env_params, policy_spec).to(device)
        super().__init__(policy=policy, device=device)

        # Optional optimizer example:
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              evaluator: Optional[Evaluator] = None,
              show_progress: bool = True
              ) -> None:
        """
        Train the PPO agent.

        Args:
            env (EnvironmentInterface): The environment to train on.
            total_steps (int): Total number of steps to train for.
            schedulers (Optional[List[ParameterScheduler]]): Learning rate schedulers.
            logger (Optional[Logger]): Logger for training metrics.
            evaluator (Optional[Evaluator]): Evaluator for performance evaluation.
            show_progress (bool): If True, show a progress bar during training.
        """
        logger = logger or Logger()

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        num_steps = 0

    def _save_impl(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        agent_meta = {{
            "algo": "{algo_pascal}",
            "agent_format_version": 1,
            "config": asdict(self.config),
        }}
        (path / "agent.json").write_text(json.dumps(agent_meta, indent=2))

        {algo_pascal}PolicyFactory().save(self.env_params, self.policy_spec, self.policy, path / "policy")

        # Optional optimizer save:
        # torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "{algo_pascal}Agent":
        p = Path(path)
        agent_meta = json.loads((p / "agent.json").read_text())
        if agent_meta.get("algo") != "{algo_pascal}":
            raise ValueError(f"Checkpoint algo mismatch: expected {algo_pascal}, got {{agent_meta.get('algo')}}")

        config = {algo_pascal}Config(**agent_meta.get("config", {{}}))
        env_params, policy_spec, policy = {algo_pascal}PolicyFactory().load(p / "policy", map_location=map_location)

        agent = cls(env_params=env_params, policy_spec=policy_spec, config=config, device=str(map_location))
        agent.policy = policy

        # Optional optimizer restore:
        # opt_state = torch.load(p / "optimizer.pt", map_location=map_location)
        # agent.optimizer.load_state_dict(opt_state)

        return agent
'''


def generate_single_file_algorithm(
    agent_path: str,
    *,
    repo_root: Path | None = None,
    force: bool = False,
) -> Path:
    root = repo_root or Path.cwd()
    base_pkg = root / "src" / "prt_rl"

    parts = [p for p in agent_path.strip().split("/") if p]
    if not parts:
        raise ValueError("agent_path must not be empty")

    pkg_parts = [_to_snake(p) for p in parts[:-1]]
    algo_name_raw = parts[-1]

    algo_snake = _to_snake(algo_name_raw)
    algo_pascal = algo_name_raw  # preserve user casing: PPO, DAgger, TD3

    # Ensure directories exist, but DO NOT create __init__.py
    out_dir = base_pkg
    for p in pkg_parts:
        out_dir = out_dir / p
        out_dir.mkdir(parents=True, exist_ok=True)

    target = out_dir / f"{algo_snake}.py"
    if target.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {target} (use --force)"
        )

    target.write_text(_render_single_file_template(algo_pascal))
    return target



def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prt-rl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_algo = sub.add_parser("algorithm", help="Generate a single-file algorithm scaffold under src/prt_rl/")
    p_algo.add_argument("agent_path", type=str, help='e.g. "PPO" or "imitation/DAgger"')
    p_algo.add_argument("--force", action="store_true", help="Overwrite if target exists")

    args = parser.parse_args(argv)

    if args.cmd == "algorithm":
        out = generate_single_file_algorithm(args.agent_path, force=args.force)
        print(f"Created scaffold: {out}")
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")
