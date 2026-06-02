"""
SWE Bash Tool: Execute bash commands inside per-task Docker containers.

Implements the mini-swe-agent approach — only one tool (bash), the model
figures out how to use shell commands to navigate, edit, and test code.

Requires: pip install docker
          Docker daemon running
          SWE-Gym Lite docker images pulled
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)

MAX_OUTPUT_LENGTH = 16000  # Truncate long outputs


class SWEBashTool(BaseTool):
    """Execute bash commands in a per-task Docker container for SWE tasks."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}
        self.max_output_length = config.get("max_output_length", MAX_OUTPUT_LENGTH)
        self.timeout = config.get("timeout", 120)  # seconds per command
        self.workdir = config.get("workdir", "/testbed")  # SWE-bench default
        self._docker_client = None

    def _get_docker_client(self):
        """Lazy import and create docker client (cached)."""
        if self._docker_client is None:
            import docker
            self._docker_client = docker.from_env()
        return self._docker_client

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Start a Docker container for this task instance."""
        if instance_id is None:
            instance_id = str(uuid4())

        # Get task-specific info from kwargs
        tools_kwargs = kwargs.get("tools_kwargs", {})
        bash_kwargs = tools_kwargs.get("bash", {}).get("create_kwargs", {})
        docker_image = bash_kwargs.get("docker_image", "")
        test_cmd = bash_kwargs.get("test_cmd", "")
        issue_text = bash_kwargs.get("issue_text", "")

        if not docker_image:
            self._instances[instance_id] = {"container": None, "done": True, "reward": 0.0}
            return instance_id, ToolResponse(text="Error: no docker_image specified in task.")

        try:
            client = self._get_docker_client()
            container = client.containers.run(
                docker_image,
                command="sleep infinity",
                detach=True,
                remove=False,
            )
            initial_obs = f"Repository loaded. Issue:\n{issue_text}\n\nYou can use bash commands to explore, edit, and test the code."
        except Exception as e:
            logger.error(f"Failed to start container {docker_image}: {e}")
            self._instances[instance_id] = {"container": None, "done": True, "reward": 0.0}
            return instance_id, ToolResponse(text=f"Error starting container: {e}")

        self._instances[instance_id] = {
            "container": container,
            "docker_image": docker_image,
            "test_cmd": test_cmd,
            "done": False,
            "reward": 0.0,
            "steps": 0,
        }

        return instance_id, ToolResponse(text=initial_obs)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute a bash command in the container.

        Args:
            parameters: {"command": "grep -rn 'def foo' src/"}
        """
        state = self._instances.get(instance_id)
        if state is None or state.get("done"):
            return ToolResponse(text="Session ended."), 0.0, {"done": True}

        container = state["container"]
        if container is None:
            return ToolResponse(text="No container available."), 0.0, {"error": "no_container"}

        command = parameters.get("command", "echo 'no command'")
        state["steps"] += 1

        try:
            exec_result = container.exec_run(
                ["bash", "-c", command],
                workdir=self.workdir,
            )
            output = exec_result.output.decode("utf-8", errors="replace")
            exit_code = exec_result.exit_code

            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (truncated)"

            obs = f"Exit code: {exit_code}\n{output}" if exit_code != 0 else output
        except Exception as e:
            obs = f"Execution error: {e}"

        metrics = {"steps": state["steps"], "done": state["done"]}
        return ToolResponse(text=obs), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Run tests in the container to determine if the issue is fixed."""
        state = self._instances.get(instance_id)
        if state is None:
            return 0.0

        container = state.get("container")
        test_cmd = state.get("test_cmd", "")

        if container is None or not test_cmd:
            return 0.0

        try:
            exec_result = container.exec_run(
                ["bash", "-c", test_cmd],
                workdir=self.workdir,
            )
            # reward = 1.0 if all tests pass (exit code 0)
            state["reward"] = 1.0 if exec_result.exit_code == 0 else 0.0
        except Exception as e:
            logger.warning(f"Test execution failed: {e}")
            state["reward"] = 0.0

        return state["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Stop and remove the Docker container."""
        state = self._instances.pop(instance_id, None)
        if state and state.get("container") is not None:
            try:
                state["container"].remove(force=True)
            except Exception:
                pass
