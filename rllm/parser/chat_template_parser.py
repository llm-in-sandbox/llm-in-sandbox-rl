import json
import logging
import re

import torch

from rllm.tools.tool_base import Tool, ToolCall, ToolOutput

from .utils import PARSER_TEST_MESSAGES

logger = logging.getLogger(__name__)


class ChatTemplateParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generation_prompt = self._get_generation_prompt(tokenizer)

    def _get_generation_prompt(self, tokenizer):
        messages = [{"role": "assistant", "content": ""}]

        with_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        without_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        generation_prompt = with_prompt[len(without_prompt) :]

        return generation_prompt

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def verify_equivalence(self, messages, verbose=True):
        """Verify that parsing messages together is equivalent to parsing them individually.

        Args:
            messages (list): List of message dictionaries to test
            verbose (bool): Whether to print detailed information about the test

        Returns:
            bool: True if the equivalence check passes, False otherwise

        Raises:
            AssertionError: If the equivalence check fails and verbose is True
        """
        # Parse all messages together
        batch_result = self.parse(messages)

        # Parse each message individually and concatenate
        individual_results = []
        for message in messages:
            individual_results.append(self.parse([message]))

        concatenated_result = "".join(individual_results)

        # Check if results are equivalent
        is_equivalent = batch_result == concatenated_result

        if verbose and not is_equivalent:
            print("Equivalence check failed!")
            print("Batch parsing result:")
            print(batch_result)
            print("\nConcatenated individual parsing result:")
            print(concatenated_result)
            raise AssertionError("Parser failed equivalence check. See above for details.")

        return is_equivalent

    @classmethod
    def get_parser(cls, tokenizer, disable_thinking=False, tool_parser_type: str = None) -> "ChatTemplateParser":
        """Factory method to get the appropriate parser based on a string identifier.

        Args:
            tokenizer: The tokenizer to use with the parser
            disable_thinking: Whether generation prompt will disable thinking.
            tool_parser_type: Optional string to specify tool parser type.
                - Any vLLM parser name: "hermes", "llama3_json", "mistral", "qwen3_xml", etc.
                - "r1": Use R1ToolParser (for DeepSeek R1 format)
                - None: Auto-detect based on model name

        Returns:
            ChatTemplateParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Create tool_parser if specified
        tool_parser = None
        if tool_parser_type is not None:
            tool_parser_type_lower = tool_parser_type.lower()
            if tool_parser_type_lower == "r1":
                from rllm.parser.tool_parser import R1ToolParser
                tool_parser = R1ToolParser()
                logger.info(f"Using R1ToolParser (tool_parser_type={tool_parser_type})")
            else:
                # Use VLLMToolParser for any other parser type
                from rllm.parser.tool_parser import VLLMToolParser
                try:
                    tool_parser = VLLMToolParser(tokenizer, parser_name=tool_parser_type_lower)
                    logger.info(f"Using VLLMToolParser with parser_name={tool_parser_type_lower}")
                except Exception as e:
                    logger.warning(f"Failed to create VLLMToolParser({tool_parser_type}): {e}, will use default")
        
        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            logger.info(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                logger.info(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                return DeepseekQwenChatTemplateParser(tokenizer)
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                logger.info(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                return QwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking, tool_parser=tool_parser)
            elif "llama" in model_name:
                logger.info(f"Using LlamaChatTemplateParser for {tokenizer.name_or_path}")
                return LlamaChatTemplateParser(tokenizer)

        # Default to the standard parser if no specific match
        parser = ChatTemplateParser(tokenizer)
        logger.info(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        assert parser.verify_equivalence(PARSER_TEST_MESSAGES), "Parser failed equivalence check"
        return parser

    def tokenize_and_mask(self, messages):
        try:
            last_assistant_idx = max(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except ValueError:
            raise ValueError("No assistant message found in chat_completions") from None

        prompt = self.parse(messages[:last_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        response = self.parse([messages[last_assistant_idx]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
        response = response[len(self.generation_prompt) :].rstrip("\n")  # handle qwen trailing newline from eot token
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        response_mask = [1] * len(response_ids)

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask

    def tokenize_and_mask_cumulative(self, messages):
        response_ids = []
        response_mask = []

        try:
            first_assistant_idx = next(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except StopIteration:
            raise ValueError("No assistant message found in chat_completions") from None

        prompt = self.parse(messages[:first_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        for i in range(first_assistant_idx, len(messages)):
            is_asst = messages[i]["role"] == "assistant"
            if is_asst:
                response = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
                response = response[len(self.generation_prompt) :]
                ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_ids.extend(ids)
                response_mask.extend([1] * len(ids))
            else:
                response = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=True, accumulate_reasoning=False)
                ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_ids.extend(ids)
                response_mask.extend([0] * len(ids))

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask


class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        self.generation_prompt = self.assistant_token + "<think>\n"

        from rllm.parser.tool_parser import R1ToolParser

        self.tool_parser = R1ToolParser()

    def parse(self, messages: list[dict], add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        tools = tools or []
        tools_prompt_str = ""
        if tools:
            try:
                tool_schema_strs = []
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_schema_str = json.dumps(tool.json)
                    elif isinstance(tool, dict):
                        tool_schema_str = json.dumps(tool)
                    else:
                        tool_schema_str = tool
                    tool_schema_strs.append(tool_schema_str)
                tools_schema_str = "\n".join(tool_schema_strs)
                tools_prompt_str = self.tool_parser.get_tool_prompt(tools_schema_str)
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Failed to format tools: {e}")

        result = ""

        if is_first_msg:
            result += self.bos_token

        if is_first_msg and messages[0]["role"] != "system" and tools_prompt_str:
            result += self.system_token + tools_prompt_str

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message, tools_prompt_str)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, accumulate_reasoning=accumulate_reasoning)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message, tools_prompt_str=""):
        content = message["content"]

        if "# Tools" not in content and tools_prompt_str:
            content += tools_prompt_str

        return self.system_token + content

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message, accumulate_reasoning=False):
        content = (message.get("content", None) or "").strip()
        reasoning = (message.get("reasoning", None) or "").strip()
        tool_calls = message.get("tool_calls", None) or []

        if not accumulate_reasoning:
            return self.assistant_token + content + self.eos_token
        elif not reasoning:
            return self.assistant_token + "<think>\n" + content + self.eos_token
        else:
            result = self.assistant_token

            if reasoning and accumulate_reasoning:
                result += "<think>\n" + reasoning + "\n</think>\n\n"

            if content:
                result += content
                if tool_calls:
                    result += "\n"

            if tool_calls:
                try:
                    tool_calls_strs = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, ToolCall):
                            tool_call_dict = tool_call.to_dict()
                        elif isinstance(tool_call, dict) and "function" in tool_call:
                            tool_call_dict = tool_call["function"]
                        else:
                            tool_call_dict = tool_call
                        # Avoid mutating original message structures by parsing into a local variable
                        arguments_obj = tool_call_dict.get("arguments")
                        if isinstance(arguments_obj, str):
                            try:
                                arguments_obj = json.loads(arguments_obj)
                            except json.JSONDecodeError:
                                pass
                        tool_call_json = f"```json\n{json.dumps(arguments_obj)}\n```"
                        tool_call_str = f"{self.tool_parser.tool_call_begin}function{self.tool_parser.tool_sep}{tool_call_dict['name']}\n{tool_call_json}\n{self.tool_parser.tool_call_end}"
                        tool_calls_strs.append(tool_call_str)
                    joined_calls_str = "\n".join(tool_calls_strs)
                    tool_calls_str = f"{self.tool_parser.tool_calls_begin}\n{joined_calls_str}\n{self.tool_parser.tool_calls_end}"
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.error(f"Failed to format tool calls: {e}")
                    tool_calls_str = ""

                result += tool_calls_str

            result += self.eos_token
            return result

    def parse_tool(self, message):
        tool_outputs: list[ToolOutput | dict] = message.get("tool_outputs", [])

        if not tool_outputs:
            return self.user_token + self.tool_parser.tool_output_begin + "\n" + message["content"] + "\n" + self.tool_parser.tool_output_end

        else:
            try:
                tool_outputs_strs = []
                for tool_output in tool_outputs:
                    if not isinstance(tool_output, ToolOutput):
                        tool_output = ToolOutput(**tool_output)
                    tool_output_str = f"{self.tool_parser.tool_output_begin}\n{str(tool_output)}\n{self.tool_parser.tool_output_end}"
                    tool_outputs_strs.append(tool_output_str)
                tool_outputs_str = "\n".join(tool_outputs_strs)
            except Exception as e:
                logger.error(f"Failed to format tool outputs: {e}")
                tool_outputs_str = ""

            return self.user_token + tool_outputs_str

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            reasoning = reasoning.strip()
            content = content.strip()
        else:
            reasoning = None
            content = completion_text
            if content.startswith("<think>"):
                content = content[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            content = content.strip()

        tool_calls = self.tool_parser.parse(completion_text)

        begin_pattern = re.escape(self.tool_parser.tool_call_begin)
        end_pattern = re.escape(self.tool_parser.tool_call_end)
        content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL)

        wrapper_begin_pattern = re.escape(self.tool_parser.tool_calls_begin)
        wrapper_end_pattern = re.escape(self.tool_parser.tool_calls_end)
        content = re.sub(f"{wrapper_begin_pattern}.*?{wrapper_end_pattern}", "", content, flags=re.DOTALL)

        content = content.strip()

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }


class QwenChatTemplateParser(ChatTemplateParser):
    """
    Qwen Chat Template Parser that uses the tokenizer's native apply_chat_template.
    This ensures 100% compatibility with vLLM and other inference engines.
    """
    def __init__(self, tokenizer, disable_thinking=False, tool_parser=None):
        super().__init__(tokenizer)
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.disable_thinking = disable_thinking
        
        # assistant_token is used by utils.py for tokenization
        self.assistant_token = "<|im_start|>assistant\n"
        self.user_token = "<|im_start|>user\n"
        
        # generation_prompt is used by base class tokenize_and_mask methods
        if disable_thinking:
            self.generation_prompt = self.assistant_token + "<think>\n\n</think>\n\n"
        else:
            self.generation_prompt = self.assistant_token

        # Allow passing custom tool_parser from outside
        if tool_parser is not None:
            self.tool_parser = tool_parser
            print(f"[QwenChatTemplateParser] Using custom tool_parser: {tool_parser}", flush=True)
        else:
            from rllm.parser.tool_parser import VLLMToolParser
            self.tool_parser = VLLMToolParser(tokenizer, parser_name="hermes")
            print(f"[QwenChatTemplateParser] WARNING: No tool_parser_type specified in config, using default 'hermes'. "
                  f"Set 'rllm.tool_parser_type' in config to use a different parser.", flush=True)
        
        # Store tools for use in parse_completion (to match vLLM's behavior)
        # Tools are automatically stored when parse() is called with tools parameter
        self._current_tools: list[dict] = None

    def parse(self, messages: list[dict], add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        """
        Parse messages using Qwen's native chat template for maximum compatibility.
        
        The messages are converted to OpenAI format before being passed to apply_chat_template.
        - reasoning field is prepended to content as <think>...</think>
        - tool_calls are converted to OpenAI format
        - tool responses use role="tool" with tool_call_id
        - is_first_msg controls whether to add bos_token at the beginning
        """
        tools = tools or []
        
        # Store tools for parse_completion to use (matching vLLM's behavior)
        if tools:
            self._current_tools = tools
        
        # Convert messages to OpenAI-compatible format for tokenizer
        converted_messages = []
        for msg in messages:
            converted_msg = {"role": msg["role"]}
            
            if msg["role"] == "assistant":
                # Build content with reasoning if accumulate_reasoning is True
                content_parts = []
                reasoning = msg.get("reasoning", "")
                content = msg.get("content", "") or ""
                
                if reasoning and accumulate_reasoning:
                    content_parts.append(f"<think>\n{reasoning}\n</think>\n\n")
                if content:
                    content_parts.append(content)
                
                # Use empty string instead of None to avoid TypeError in Jinja template
                converted_msg["content"] = "".join(content_parts) if content_parts else ""
                
                # Convert tool_calls to OpenAI format
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    openai_tool_calls = []
                    for i, tc in enumerate(tool_calls):
                        if isinstance(tc, ToolCall):
                            tc_name = tc.name
                            tc_args = tc.arguments
                            tc_id = getattr(tc, 'id', f'call_{i}')
                        elif isinstance(tc, dict):
                            if "function" in tc:
                                # Already OpenAI format, but may need to convert arguments
                                tc_name = tc["function"].get("name", "")
                                tc_args = tc["function"].get("arguments", {})
                                tc_id = tc.get("id", f'call_{i}')
                            else:
                                tc_name = tc.get("name", "")
                                tc_args = tc.get("arguments", {})
                                tc_id = tc.get("id", f'call_{i}')
                        else:
                            continue
                        
                        # Always use dict format for arguments - compatible with both
                        # Qwen3-Instruct and Qwen3-Coder chat templates
                        if isinstance(tc_args, str):
                            try:
                                tc_args = json.loads(tc_args)
                            except json.JSONDecodeError:
                                tc_args = {"raw": tc_args}
                        
                        openai_tool_calls.append({
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc_name,
                                "arguments": tc_args
                            }
                        })
                    if openai_tool_calls:
                        converted_msg["tool_calls"] = openai_tool_calls
            
            elif msg["role"] == "tool":
                converted_msg["content"] = msg.get("content", "")
                # tool_call_id is required for tool messages
                converted_msg["tool_call_id"] = msg.get("tool_call_id", "call_0")
            
            else:
                # system, user messages - just copy content
                converted_msg["content"] = msg.get("content", "")
            
            converted_messages.append(converted_msg)
        
        # Use tokenizer's native apply_chat_template
        result = self.tokenizer.apply_chat_template(
            converted_messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        
        # Fix for Qwen3-Coder chat template bug: when parsing a single tool message,
        # the template doesn't add <|im_start|>user\n prefix. We need to add it manually.
        # Qwen3-Coder produces: "<tool_response>...</tool_response>\n<|im_end|>\n"
        # But it should be: "<|im_start|>user\n<tool_response>...</tool_response>\n<|im_end|>\n"
        if (len(converted_messages) == 1 and 
            converted_messages[0]["role"] == "tool" and 
            result.startswith("<tool_response>")):
            result = self.user_token + result
        
        # If disable_thinking and add_generation_prompt, append the thinking prefix
        if add_generation_prompt and self.disable_thinking:
            # apply_chat_template already adds <|im_start|>assistant\n
            # We need to add <think>\n\n</think>\n\n after it
            result = result + "<think>\n\n</think>\n\n"
        
        return result

    def parse_completion(self, completion_ids, tools: list[dict] = None):
        """Parse model completion to extract content, reasoning, and tool calls.
        
        Args:
            completion_ids: Token IDs of the model's completion
            tools: Optional list of tool definitions. If not provided, will use 
                   tools stored from the last parse() call. This matches vLLM's 
                   behavior where the same tools from the request are used.
        """
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            if content.endswith(self.eot_token):
                content = content[: -len(self.eot_token)]
            reasoning = reasoning.strip()
            content = content.strip()
        else:
            reasoning = None
            content = completion_text
            if content.startswith("<think>"):
                content = content[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            if content.endswith(self.eot_token):
                content = content[: -len(self.eot_token)]
            content = content.strip()

        # Use provided tools or fall back to stored tools from parse()
        # This matches vLLM's behavior where extract_tool_calls uses request.tools
        effective_tools = tools if tools is not None else self._current_tools
        tool_calls = self.tool_parser.parse(content, tools=effective_tools)

        begin_pattern = re.escape(self.tool_parser.tool_call_begin)
        end_pattern = re.escape(self.tool_parser.tool_call_end)
        content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL)
        content = content.strip()

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }


class LlamaChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = "<|begin_of_text|>"
        self.system_token = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_token = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.eot_token = "<|eot_id|>"
        self.generation_prompt = self.assistant_token

        # tool tokens
        self.tool_start_token = "<|start_header_id|>tool<|end_header_id|>\n\n"
        self.tool_end_token = "<|eot_id|>"
        self.tool_response_start_token = "<|start_header_id|>tool_response<|end_header_id|>\n\n"
        self.tool_response_end_token = "<|eot_id|>"

        # TODO: add tool parser for llama

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eot_token

    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token

    def parse_completion(self, completion_ids):
        # TODO: add parse_completion for llama
        raise NotImplementedError("LLamaChatTemplateParser does not support parse_completion")
