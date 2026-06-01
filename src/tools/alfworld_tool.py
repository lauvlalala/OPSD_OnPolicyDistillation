"""
ALFWorld environment wrapped as a verl BaseTool.

Provides a text-based household task environment where the agent
issues actions (look, go, pick, put, etc.) and receives observations.

Requires: pip install alfworld
          export ALFWORLD_DATA=/path/to/alfworld/data
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class AlfWorldTool(BaseTool):
    """Wraps ALFWorld TextWorld environment as a verl tool."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}
        self.max_steps = config.get("max_steps", 50)
        self.data_path = config.get("data_path", os.environ.get("ALFWORLD_DATA", ""))

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a new ALFWorld environment instance."""
        if instance_id is None:
            instance_id = str(uuid4())

        try:
            import alfworld.agents.environment as environment
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        except ImportError:
            msg = "alfworld not installed. Run: pip install alfworld"
            self._instances[instance_id] = {"env": None, "done": True, "reward": 0.0, "steps": 0}
            return instance_id, ToolResponse(text=msg)

        # Initialize environment
        env = AlfredTWEnv(self.data_path)
        obs, info = env.reset()
        initial_obs = obs[0] if isinstance(obs, list) else str(obs)

        self._instances[instance_id] = {
            "env": env,
            "done": False,
            "reward": 0.0,
            "steps": 0,
            "obs": initial_obs,
        }

        return instance_id, ToolResponse(text=initial_obs)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute an action in the ALFWorld environment.

        Args:
            parameters: {"action": "go to desk 1"} or {"action": "look"}

        Returns:
            (observation, step_reward, metrics)
        """
        state = self._instances.get(instance_id)
        if state is None or state.get("done"):
            return ToolResponse(text="Episode finished."), 0.0, {"done": True}

        action = parameters.get("action", "look")
        env = state["env"]

        if env is None:
            return ToolResponse(text="Environment not available."), 0.0, {"error": "no_env"}

        # Step the environment
        obs, reward, done, info = env.step([action])
        obs_text = obs[0] if isinstance(obs, list) else str(obs)
        step_reward = float(reward[0]) if isinstance(reward, list) else float(reward)
        is_done = done[0] if isinstance(done, list) else bool(done)

        state["steps"] += 1
        state["reward"] += step_reward
        state["obs"] = obs_text

        if is_done or state["steps"] >= self.max_steps:
            state["done"] = True

        metrics = {
            "steps": state["steps"],
            "done": state["done"],
            "cumulative_reward": state["reward"],
        }

        return ToolResponse(text=obs_text), step_reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Return cumulative reward (1.0 if task completed, 0.0 otherwise)."""
        state = self._instances.get(instance_id)
        if state is None:
            return 0.0
        return state["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up environment instance."""
        state = self._instances.pop(instance_id, None)
        if state and state.get("env") is not None:
            try:
                state["env"].close()
            except Exception:
                pass
