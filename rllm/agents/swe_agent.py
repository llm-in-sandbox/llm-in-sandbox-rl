import json
import logging
import re
import uuid

try:
    from r2egym.agenthub.action import Action as SWEAction
    from r2egym.agenthub.agent.agent import AgentArgs
    from r2egym.agenthub.tools import (
        search_tool,
        file_editor,
        r2egym_bash_execute_tool,
        finish_tool,
        str_replace_editor_tool,
        execute_bash_tool,
        submit_tool,
    )
except ImportError as e:
    raise ImportError(f"r2egym is required for SWEAgent. Please install r2egym to use this class. Error: {e}")

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.prompt_loader import get_r2egym_config_path

def parse_xml_response(response_text: str) -> tuple[str, SWEAction]:
    """
    Extracts:
    - thought: everything before the first <function=...> block
    - action: the entire first <function=...></function> block
    Returns (thought, action).
    """
    # Regex to match (non-greedily) from `<function=` up to the first `</function>`
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action = match.group(1)  # The entire <function=...></function> block
        thought = response_text[: match.start()]  # Everything before the block
    else:
        # If no match, treat entire text as "thought"
        thought = response_text
        action = ""

    # Strip leading/trailing whitespace
    thought = thought.strip()
    action = action.strip()

    # convert action to Action object
    action = SWEAction.from_string(action)

    return thought, action


logger = logging.getLogger(__name__)

class SWEAgent(BaseAgent):
    def __init__(self, use_fn_calling: bool = False, format_model_response: bool = False, scaffold: str = "openhands", **kwargs):
        """
        Initialize SWE Agent.
        
        Note: Prompts are now loaded dynamically in update_from_env() based on data_source,
        not in __init__. This allows multi-domain training with per-sample prompts.
        
        Args:
            use_fn_calling: Whether to use function calling format
            format_model_response: Whether to reformat model responses
            scaffold: Which scaffold to use ('r2egym', 'sweagent', 'openhands')
            **kwargs: Additional arguments (ignored for forward compatibility)
        """
        self.use_fn_calling = use_fn_calling
        self.format_model_response = format_model_response
        self.scaffold = scaffold
        
        assert self.scaffold == "openhands", "Currently only openhands scaffold is supported"

        self._trajectory = Trajectory()
        self.reset()

    def update_from_env(self, observation, reward, done, info):
        """
        Update agent state from environment observation.
        
        On the first step (env.reset()), this method:
        1. Extracts data_source from info['extra_info']
        2. Maps data_source to domain using map_data_source_to_domain()
        3. Dynamically loads prompts from corresponding YAML file
        4. Updates system_prompt in self.messages[0]
        5. Formats observation using the loaded user_prompt_template
        
        This enables multi-domain training where each sample uses its own prompts.
        """
        # If the first step in environment, dynamically load prompts based on data_source
        if not self._trajectory.steps:
            # Extract data_source from extra_info
            extra_info = info.get("extra_info", {})
            if isinstance(extra_info, str):
                import json
                extra_info = json.loads(extra_info)
            
            config_dir = get_r2egym_config_path(scaffold=self.scaffold)
            unified_config_path = config_dir / "edit.yaml"
            unified_agent_args = AgentArgs.from_yaml(unified_config_path)
            self.system_prompt, self.user_prompt_template = unified_agent_args.system_prompt, unified_agent_args.instance_prompt

            data_source = extra_info.get("data_source", "")
            
            if data_source:
                try:
                    # Map data_source to domain
                    domain = extra_info["domain"]
                    assert domain in ("math", "chem", "ugphysics", "medxpertqa", "aalcr", "instructpt", "read_compre"), f"Invalid domain: {domain}"
                    if domain:
                        yaml_file = config_dir / f"{domain}_solver.yaml"
                        agent_args = AgentArgs.from_yaml(yaml_file)
                        
                        # Load domain-specific prompts from YAML
                        # Signature: load_prompts_from_yaml(domain, scaffold, version="")
                        self.system_prompt, self.user_prompt_template = agent_args.system_prompt, agent_args.instance_prompt
                        
                        logger.info(f"Loaded prompts for data_source='{data_source}' -> domain='{domain}'")
                    else:
                        # Use default SWE prompts (already set in __init__)
                        logger.info(f"Using default SWE prompts for data_source='{data_source}'")
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to load prompts for data_source '{data_source}': {e}.")
                    # Keep using default prompts set in __init__
            
            # For non-fn-calling mode, we append function_description to system_prompt
            # The edit.yaml has both system_prompt and function_description as separate fields
            if not self.use_fn_calling:
                self.system_prompt += "\n\n" + unified_agent_args.function_description
                logger.info(f"Using non-fn-calling mode, system_prompt with appended function description")

            # Update the system prompt in messages (reset() was already called with old prompt)
            self.messages[0]['content'] = self.system_prompt

            # Format observation with user_prompt_template
            observation = str(observation)
            observation = self.user_prompt_template.replace("{problem_statement}", observation).replace('{working_dir}', '/testbed')
        else:
            observation = str(observation)

        max_steps = info.get("max_steps", None)
        if max_steps:
            remaining_steps = max_steps - self.step - 1
            if remaining_steps > 0:
                observation += f"\nSteps Remaining: {remaining_steps}"
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."

        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info
        
        # After model generates tool_call, the next observation should be a tool response
        # If previous message was assistant with tool call, use role="tool"
        # Otherwise use role="user" (first step or non-tool interactions)
        if self.messages and self.messages[-1]["role"] == "assistant":
            # Check if last assistant message had tool calls
            last_msg = self.messages[-1]
            tool_calls_list = last_msg.get("tool_calls", [])
            has_tool_call = len(tool_calls_list) > 0
            
            if has_tool_call:
                # Tool response format - match R2E-Gym format with name and tool_call_id
                # Extract function name and tool_call_id from the last assistant's tool_calls
                last_tool_call = tool_calls_list[0] if tool_calls_list else {}
                if hasattr(last_tool_call, 'name'):
                    function_name = last_tool_call.name
                    tool_call_id = getattr(last_tool_call, 'id', f'call_{uuid.uuid4().hex[:24]}')
                elif isinstance(last_tool_call, dict):
                    # Handle OpenAI format: {'id': '...', 'type': 'function', 'function': {'name': '...', 'arguments': '...'}}
                    if 'function' in last_tool_call:
                        function_name = last_tool_call['function'].get('name', '')
                        tool_call_id = last_tool_call.get('id', f'call_{uuid.uuid4().hex[:24]}')
                    else:
                        # Handle simple format: {'name': '...', 'arguments': {...}}
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
                # Regular user message
                self.messages.append({"role": "user", "content": observation})
        else:
            # First message or after user message - use role="user"
            self.messages.append({"role": "user", "content": observation})
        
        self.cur_step = Step(observation=observation)

    def update_from_model(self, response: str, **kwargs):
        """
        Updates the agent's internal state after an environment step.

        This function is called during environment interaction to incorporate the latest action's
        outcome into the agent's learning process.

        Args:
            response (str): The response from the model.

        Returns:
            None
        """
        self._trajectory.steps.append(self.cur_step)
        
        # Handle ModelOutput object from rollout engine
        # VerlEngine always returns ModelOutput with text, content, tool_calls, etc.
        if not hasattr(response, 'text'):
            raise TypeError(f"Expected ModelOutput object with 'text' attribute, got {type(response)}")
        
        response_text = response.text
        tool_calls = getattr(response, 'tool_calls', [])
        
        # If we have parsed tool_calls, use them directly (JSON format)
        if tool_calls:
            tool_call = tool_calls[0]  # Take first tool call
            if hasattr(tool_call, 'name'):
                function_name = tool_call.name
                parameters = tool_call.arguments
            else:
                function_name = tool_call.get('name', '')
                parameters = tool_call.get('arguments', {})
            # Use positional arguments like parse_oai_response does
            action = SWEAction(function_name, parameters)
            # Use reasoning (the <think> content) as thought, not content (which is empty after parsing)
            thought = getattr(response, 'reasoning', '') or ''
            use_json_format = True
        else:
            # No tool calls detected - this shouldn't happen with use_fn_calling=True
            # Fallback: try to parse from raw text (for debugging/error cases)
            # print(f"[WARNING] No tool_calls found in ModelOutput:\nlast 100 chars of response:\n{response_text[-100:]}")
            if not self.use_fn_calling:
                thought, action = parse_xml_response(response_text)
            else:
                thought, action = response_text, SWEAction.from_string("")
            use_json_format = False
        
        # Store the SWEAction object directly - environment expects Action object, not XML string
        assert self._trajectory.steps, "Trajectory should not be empty when update_from_model is called."

        # Update Trajectory - store the SWEAction object directly
        cur_step = self._trajectory.steps[-1]
        cur_step.thought = thought
        cur_step.action = action  # Store SWEAction object (not XML string)
        cur_step.model_response = response_text

        # Update Chat Completions
        # Store parsed components for the parser to use
        # Parser (parse_assistant) will reconstruct the output based on these fields
        # Convert tool_calls to OpenAI format to match R2E-Gym
        tool_calls_openai_format = []
        for tc in getattr(response, 'tool_calls', []):
            if hasattr(tc, 'name'):
                tc_name = tc.name
                tc_args = tc.arguments
            elif isinstance(tc, dict):
                if 'function' in tc:
                    # Already in OpenAI format
                    tool_calls_openai_format.append(tc)
                    continue
                tc_name = tc.get('name', '')
                tc_args = tc.get('arguments', {})
            else:
                continue
            # Convert to OpenAI format
            tool_calls_openai_format.append({
                'id': f'call_{uuid.uuid4().hex[:24]}',
                'type': 'function',
                'function': {
                    'name': tc_name,
                    'arguments': json.dumps(tc_args) if isinstance(tc_args, dict) else tc_args,
                }
            })
        
        assistant_msg = {
            "role": "assistant",
            "content": getattr(response, 'content', '') or '',  # Text content (without <think> and <tool_call>)
            "reasoning": getattr(response, 'reasoning', '') or '',  # <think> content
            "tool_calls": tool_calls_openai_format,  # OpenAI format tool_calls
        }
        
        # Store raw model output for debugging/training analysis
        if hasattr(response, '__dict__'):
            # Convert tool_calls to JSON-serializable format
            tool_calls_serializable = []
            for tc in getattr(response, 'tool_calls', []):
                if hasattr(tc, '__dict__'):
                    tool_calls_serializable.append({
                        'name': getattr(tc, 'name', ''),
                        'arguments': getattr(tc, 'arguments', {}),
                    })
                else:
                    tool_calls_serializable.append(tc)
            
            # Convert completion_ids to list if it's a tensor/array
            completion_ids = getattr(response, 'completion_ids', None)
            if completion_ids is not None and hasattr(completion_ids, 'tolist'):
                completion_ids = completion_ids.tolist()
            
            assistant_msg['_raw_model_output'] = {
                'text': getattr(response, 'text', ''),
                'content': getattr(response, 'content', ''),
                'tool_calls': tool_calls_serializable,
                'completion_ids': completion_ids,
            }
            
            # Also store parsed components for easy access
            assistant_msg['_parsed'] = {
                'thought': thought,
                'action_function': action.function_name,  # Function name
                'action_parameters': action.parameters,  # Parameters dict
                'use_json_format': use_json_format,
            }
        
        self.messages.append(assistant_msg)
        self.step += 1
        # Return Action wrapper containing the SWEAction object
        return Action(action=cur_step.action)  # cur_step.action is now a SWEAction object

    def get_current_state(self) -> Step:
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
    
    def get_tools(self):
        """Return tools for the current scaffold."""
        if self.scaffold == "openhands":
            if self.use_fn_calling:
                return [str_replace_editor_tool, execute_bash_tool, submit_tool]
            else:
                return []
        else:
            raise NotImplementedError("Currently only openhands scaffold is supported")             

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [
            {
                "role": "system",
                "content": 'place_holder', # Will be updated dynamically in update_from_env()
            }
        ]
        self.step = 0

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self):
        return self.messages
