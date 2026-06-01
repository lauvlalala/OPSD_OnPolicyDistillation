"""
WebShop environment wrapped as a verl BaseTool.

Provides a text-based web shopping environment where the agent navigates
product pages, searches, and makes purchases to fulfill user instructions.

Requires: WebShop env (from SDAR/verl-agent repo)
          See: github.com/princeton-nlp/WebShop
"""

import logging
import os
import sys
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class WebShopTool(BaseTool):
    """Wraps WebShop text environment as a verl tool."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}
        self.max_steps = config.get("max_steps", 15)
        self.data_path = config.get("data_path", "")
        self._session_counter = 0

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a new WebShop environment instance."""
        if instance_id is None:
            instance_id = str(uuid4())

        try:
            import gym
            # Add webshop to path if needed
            webshop_path = self.data_path or os.environ.get("WEBSHOP_PATH", "")
            if webshop_path and webshop_path not in sys.path:
                sys.path.insert(0, webshop_path)
            from web_agent_site.envs import WebAgentTextEnv  # noqa

            env = gym.make("WebAgentTextEnv-v0")
            obs, info = env.reset(session=self._session_counter)
            self._session_counter += 1
            initial_obs = str(obs)
        except Exception as e:
            logger.warning(f"Failed to create WebShop env: {e}")
            env = None
            initial_obs = f"WebShop environment not available: {e}"

        self._instances[instance_id] = {
            "env": env,
            "done": False,
            "reward": 0.0,
            "steps": 0,
        }

        return instance_id, ToolResponse(text=initial_obs)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute an action in WebShop.

        Args:
            parameters: {"action": "search[wireless headphones]"} or {"action": "click[Buy Now]"}
        """
        state = self._instances.get(instance_id)
        if state is None or state.get("done"):
            return ToolResponse(text="Episode finished."), 0.0, {"done": True}

        action = parameters.get("action", "")
        env = state["env"]

        if env is None:
            return ToolResponse(text="Environment not available."), 0.0, {"error": "no_env"}

        obs, reward, done, info = env.step(action)
        obs_text = str(obs)
        step_reward = float(reward)

        state["steps"] += 1
        state["reward"] = step_reward  # WebShop reward is cumulative from env

        if done or state["steps"] >= self.max_steps:
            state["done"] = True

        metrics = {
            "steps": state["steps"],
            "done": state["done"],
            "reward": state["reward"],
        }

        return ToolResponse(text=obs_text), step_reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Return final reward (task score from WebShop)."""
        state = self._instances.get(instance_id)
        if state is None:
            return 0.0
        return state["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up environment instance."""
        self._instances.pop(instance_id, None)
