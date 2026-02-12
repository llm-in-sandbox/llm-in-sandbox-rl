"""
LLM-in-Sandbox Agent for rllm.

This agent reuses llm-in-sandbox's components and loads prompts from benchmark configs.
"""

import json
import logging
import uuid

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

# Import from llm-in-sandbox
from llm_in_sandbox.action import Action as LLMSandboxAction
from llm_in_sandbox.tools import str_replace_editor_tool, execute_bash_tool, submit_tool
from llm_in_sandbox.benchmark.runner import load_task_config


logger = logging.getLogger(__name__)


class LLMinSandboxAgent(BaseAgent):
    """
    Agent for LLM-in-Sandbox tasks.
    
    Loads prompts from benchmark configs.
    """

    def __init__(
        self,
        task_name: str = "instruct_pretrain",
        **kwargs
    ):
        """
        Initialize LLM-in-Sandbox Agent.

        Args:
            task_name: Task name for loading config (default: "instruct_pretrain")
            **kwargs: Additional arguments (ignored for forward compatibility)
        """
        self.task_name = task_name
        
        # Load prompts from config
        config = load_task_config(task_name)
        
        # Load directory configs with defaults
        self.working_dir = config.get("working_dir", "/testbed")
        self.input_dir = config.get("input_dir", "/testbed/documents")
        self.output_dir = config.get("output_dir", "/testbed")
        
        # Replace placeholders in prompts
        self.system_prompt = config["system_prompt"]
        self.system_prompt = self.system_prompt.replace("{working_dir}", self.working_dir)
        self.system_prompt = self.system_prompt.replace("{input_dir}", self.input_dir)
        self.system_prompt = self.system_prompt.replace("{output_dir}", self.output_dir)
        
        self.user_prompt_template = config["instance_prompt"]
        self.user_prompt_template = self.user_prompt_template.replace("{input_dir}", self.input_dir)
        self.user_prompt_template = self.user_prompt_template.replace("{output_dir}", self.output_dir)
        # Note: {working_dir} in user_prompt_template will be replaced in update_from_env

        self.reset()

    def reset(self):
        """Reset the agent state."""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.cur_step = None
        self.step = 0

    @property
    def trajectory(self) -> Trajectory:
        """Return the current trajectory."""
        return self._trajectory

    def update_from_env(self, observation: str, reward: float, done: bool, info: dict):
        """
        Update agent state from environment observation.

        Called after env.reset() or env.step() to incorporate the observation.
        """
        # Update prior step if exists
        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info

        # First call (from env.reset) - format with user prompt template
        if not self._trajectory.steps:
            extra_info = info.get("extra_info", {})
            if isinstance(extra_info, str):
                extra_info = json.loads(extra_info)
            
            # Load domain-specific prompt if available
            domain = extra_info.get("domain", "")
            self._load_domain_prompt(domain)
            
            # Format observation with template
            problem_statement = extra_info.get("problem_statement", observation)
            formatted_observation = self.user_prompt_template.replace(
                "{problem_statement}", problem_statement
            ).replace(
                "{working_dir}", self.working_dir
            )
            observation = formatted_observation
        else:
            observation = str(observation)

        # Add remaining steps info (before this step executes)
        max_steps = info.get("max_steps", None)
        if max_steps:
            steps_remaining = max_steps - self.step
            if steps_remaining > 0:
                observation += f"\nSteps Remaining: {steps_remaining}"
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."

        # Handle tool response format for function calling
        if self.messages and self.messages[-1]["role"] == "assistant":
            last_msg = self.messages[-1]
            tool_calls_list = last_msg.get("tool_calls", [])
            
            if tool_calls_list:
                # Extract tool call info for response
                last_tool_call = tool_calls_list[0]
                if hasattr(last_tool_call, 'name'):
                    function_name = last_tool_call.name
                    tool_call_id = getattr(last_tool_call, 'id', f'call_{uuid.uuid4().hex[:24]}')
                elif isinstance(last_tool_call, dict):
                    if 'function' in last_tool_call:
                        function_name = last_tool_call['function'].get('name', '')
                        tool_call_id = last_tool_call.get('id', f'call_{uuid.uuid4().hex[:24]}')
                    else:
                        function_name = last_tool_call.get('name', '')
                        tool_call_id = last_tool_call.get('id', f'call_{uuid.uuid4().hex[:24]}')
                else:
                    function_name = ''
                    tool_call_id = f'call_{uuid.uuid4().hex[:24]}'
                
                self.messages.append({
                    "role": "tool",
                    "content": observation,
                    "name": function_name,
                    "tool_call_id": tool_call_id,
                })
            else:
                self.messages.append({"role": "user", "content": observation})
        else:
            self.messages.append({"role": "user", "content": observation})

        self.cur_step = Step(observation=observation)

    def _load_domain_prompt(self, domain: str):
        """Load domain-specific prompts from benchmark config."""
        if not domain:
            return
        
        # Map domain to task name (e.g., "math_mini" -> "math")
        task_name = domain.replace("_mini", "")
        
        config = load_task_config(task_name)
        
        # Update directory configs if provided in domain config
        self.working_dir = config.get("working_dir", self.working_dir)
        self.input_dir = config.get("input_dir", self.input_dir)
        self.output_dir = config.get("output_dir", self.output_dir)
        
        # Replace placeholders in prompts
        self.system_prompt = config["system_prompt"]
        self.system_prompt = self.system_prompt.replace("{working_dir}", self.working_dir)
        self.system_prompt = self.system_prompt.replace("{input_dir}", self.input_dir)
        self.system_prompt = self.system_prompt.replace("{output_dir}", self.output_dir)
        
        self.messages[0]["content"] = self.system_prompt
        
        self.user_prompt_template = config["instance_prompt"]
        self.user_prompt_template = self.user_prompt_template.replace("{input_dir}", self.input_dir)
        self.user_prompt_template = self.user_prompt_template.replace("{output_dir}", self.output_dir)

    def update_from_model(self, response, **kwargs):
        """
        Update agent state from model response.

        Args:
            response: ModelOutput object with text and tool_calls
        """
        self._trajectory.steps.append(self.cur_step)

        # Handle ModelOutput object
        if not hasattr(response, 'text'):
            raise TypeError(f"Expected ModelOutput with 'text' attribute, got {type(response)}")

        response_text = response.text
        tool_calls = getattr(response, 'tool_calls', [])

        # Parse tool calls (gracefully handle missing tool_calls like llm-in-sandbox)
        if tool_calls:
            tool_call = tool_calls[0]
            if hasattr(tool_call, 'name'):
                function_name = tool_call.name
                parameters = tool_call.arguments
            else:
                function_name = tool_call.get('name', '')
                parameters = tool_call.get('arguments', {})
        else:
            # No tool_calls - return empty action, env will handle it
            function_name = ""
            parameters = {}
        action = LLMSandboxAction(function_name, parameters)
        thought = getattr(response, 'reasoning', '') or getattr(response, 'reasoning_content', '') or getattr(response, 'thinking', '') or ''

        # Update trajectory (matching swe_agent.py style)
        cur_step = self._trajectory.steps[-1]
        cur_step.thought = thought
        cur_step.action = action  # Store LLMSandboxAction directly
        cur_step.model_response = response_text

        # Build assistant message for conversation history
        # Use response.content (text without <think> and tool_call), matching swe_agent behavior
        assistant_msg = {
            "role": "assistant",
            "content": getattr(response, 'content', '') or '',
            "reasoning": thought,
        }
        
        # Format tool_calls for message
        formatted_calls = []
        for tc in tool_calls:
            if hasattr(tc, 'name'):
                formatted_calls.append({
                    "id": getattr(tc, 'id', f'call_{uuid.uuid4().hex[:24]}'),
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                    }
                })
            elif isinstance(tc, dict):
                if 'function' in tc:
                    formatted_calls.append(tc)
                else:
                    formatted_calls.append({
                        "id": tc.get('id', f'call_{uuid.uuid4().hex[:24]}'),
                        "type": "function",
                        "function": {
                            "name": tc.get('name', ''),
                            "arguments": json.dumps(tc.get('arguments', {})),
                        }
                    })
        assistant_msg["tool_calls"] = formatted_calls

        self.messages.append(assistant_msg)
        self.step += 1
        
        # Return Action wrapper for env.step() - engine expects action.action
        return Action(action=action)

    def get_action(self) -> Action:
        """Get the last action from trajectory."""
        if self._trajectory.steps and self._trajectory.steps[-1].action:
            return self._trajectory.steps[-1].action
        return None

    def get_tools(self):
        """Return tools for function calling mode."""
        return [str_replace_editor_tool, execute_bash_tool, submit_tool]

    @property
    def chat_completions(self):
        """Return messages for model input (used by workflow)."""
        return self.messages
