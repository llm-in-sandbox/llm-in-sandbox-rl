import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from rllm.tools.tool_base import ToolCall

logger = logging.getLogger(__name__)


# vLLM supported tool parsers (for reference)
VLLM_TOOL_PARSERS = [
    "hermes",        # Qwen, Hermes models
    "qwen3_xml",     # Qwen3 XML format
    "qwen3_coder",   # Qwen3 Coder
    "deepseek_v3",   # DeepSeek V3
    "deepseek_v31",  # DeepSeek V3.1
    "llama3_json",   # Llama 3 JSON
    "llama4_json",   # Llama 4 JSON
    "llama4_pythonic", # Llama 4 Pythonic
    "mistral",       # Mistral
    "internlm",      # InternLM
    "jamba",         # Jamba
    "granite",       # Granite
    "pythonic",      # Generic pythonic
    "openai",        # OpenAI format
    # ... and more, see vllm.entrypoints.openai.tool_parsers
]


class ToolParser(ABC):
    @abstractmethod
    def parse(self, model_response: str) -> list[ToolCall]:
        """Extract tool calls from the model response."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_tool_prompt(self, tools_schema: str) -> str:
        """Get the tool prompt for the model."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def get_parser(cls, tokenizer) -> "ToolParser":
        """Factory method to get the appropriate tool parser based on a string identifier.

        Args:
            tokenizer: The tokenizer to use with the parser

        Returns:
            ToolParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                print(f"[ToolParser] Using R1ToolParser for {tokenizer.name_or_path}", flush=True)
                return R1ToolParser()
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                print(f"[ToolParser] Using VLLMToolParser(hermes) for {tokenizer.name_or_path}", flush=True)
                return VLLMToolParser(tokenizer, parser_name="hermes")
        # TODO: add verfication to check equivalence of the parser with that from HuggingFace
        raise ValueError(f"No tool parser found for {tokenizer.name_or_path}")


class R1ToolParser(ToolParser):
    """Parser for R1 tool call format."""

    def __init__(self):
        """Initialize the R1 tool parser.

        Args:
            model (str): Model name for tokenizer (optional)
            tokenizer: Pre-initialized tokenizer (optional)
        """
        self.tool_calls_begin = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end = "<｜tool▁calls▁end｜>"
        self.tool_call_begin = "<｜tool▁call▁begin｜>"
        self.tool_call_end = "<｜tool▁call▁end｜>"
        self.tool_sep = "<｜tool▁sep｜>"
        self.tool_output_begin = "<｜tool▁response▁begin｜>"
        self.tool_output_end = "<｜tool_response_end｜>"

    def parse(self, model_response: str) -> list[ToolCall]:
        """Parse tool calls from model output.

        Args:
            model_output (str): Text containing tool calls

        Returns:
            ToolInputs: Parsed tool calls
        """
        tool_calls_dicts = self.parse_r1_tool_calls(model_response)

        # Convert dictionaries to ToolCall objects
        tool_calls = [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]
        return tool_calls

    def parse_r1_tool_calls(self, text: str) -> list[dict]:
        """Parse tool calls from text using the R1 special token format.

        Format:
        <｜tool▁calls▁begin｜>
        <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
        ```json
        {"param1": "value1", "param2": "value2"}
        ```
        <｜tool▁call▁end｜>
        // Additional tool calls follow the same format
        <｜tool▁calls▁end｜>

        Returns:
            list[dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """
        tool_calls = []

        # Look for individual tool calls
        call_idx = 0
        while True:
            # Find the next tool call beginning
            call_idx = text.find(self.tool_call_begin, call_idx)
            if call_idx == -1:
                break

            # Find the end of this tool call
            call_start = call_idx + len(self.tool_call_begin)
            call_end = text.find(self.tool_call_end, call_start)
            if call_end == -1:
                break

            # Extract the content of this tool call
            call_content = text[call_start:call_end].strip()

            # Parse function name
            func_prefix = "function" + self.tool_sep
            func_start = call_content.find(func_prefix)

            if func_start != -1:
                # Extract function name after the prefix up to the next newline
                func_name_start = func_start + len(func_prefix)
                func_name_end = call_content.find("\n", func_name_start)

                if func_name_end == -1:
                    function_name = call_content[func_name_start:].strip()
                else:
                    function_name = call_content[func_name_start:func_name_end].strip()
            else:
                # If function prefix not found, skip this call
                call_idx = call_end + len(self.tool_call_end)
                continue

            # Extract JSON arguments
            json_start = call_content.find("```json\n")
            if json_start == -1:
                json_start = call_content.find("```json")
                if json_start == -1:
                    call_idx = call_end + len(self.tool_call_end)
                    continue
                json_start += len("```json")
            else:
                json_start += len("```json\n")

            json_end = call_content.find("```", json_start)
            if json_end == -1:
                call_idx = call_end + len(self.tool_call_end)
                continue

            args_str = call_content[json_start:json_end].strip()

            try:
                args_json = json.loads(args_str)
            except json.JSONDecodeError:
                call_idx = call_end + len(self.tool_call_end)
                continue

            # Add this tool call to our list
            tool_calls.append({"name": function_name, "arguments": args_json})

            # Move past this call for the next iteration
            call_idx = call_end + len(self.tool_call_end)

        return tool_calls

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""
# Tools

You may call one or more functions to assist with the user query.
<tools>
{tools_schema}
</tools>

Output format for tool calls:

<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
```json
{{"param1": "value1", "param2": "value2"}}
```
<｜tool▁call▁end｜>
// Additional tool calls follow the same format
<｜tool▁calls▁end｜>
"""


class VLLMToolParser(ToolParser):
    """
    Generic wrapper for any vLLM tool parser.
    
    Usage:
        parser = VLLMToolParser(tokenizer, parser_name="hermes")
        parser = VLLMToolParser(tokenizer, parser_name="llama3_json")
        parser = VLLMToolParser(tokenizer, parser_name="mistral")
    
    Common parser choices:
        - "hermes"      : Qwen3-Instruct, Hermes models (default)
        - "qwen3_coder" : Qwen3-Coder models  
        - "llama3_json" : Llama 3 models
        - "mistral"     : Mistral models
        - "deepseek_v3" : DeepSeek V3 models
    
    All supported parsers (see VLLM_TOOL_PARSERS):
        hermes, qwen3_xml, qwen3_coder, deepseek_v3, deepseek_v31,
        llama3_json, llama4_json, llama4_pythonic, mistral, internlm,
        jamba, granite, pythonic, openai, and more...
    
    Config example (in yaml):
        rllm:
          tool_parser_type: hermes  # or qwen3_coder, llama3_json, etc.
    """
    
    def __init__(self, tokenizer, parser_name: str = "hermes"):
        """
        Initialize with a vLLM tool parser.
        
        Args:
            tokenizer: The tokenizer to use
            parser_name: Name of the vLLM parser (e.g., "hermes", "llama3_json", "mistral")
        """
        self.parser_name = parser_name
        self._vllm_parser = None
        
        # Default tool call tokens (may vary by parser)
        self.tool_call_begin = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_output_begin = "<tool_response>"
        self.tool_output_end = "</tool_response>"
        
        try:
            from vllm.entrypoints.openai.tool_parsers import ToolParserManager
            
            # Get the parser class from vLLM's registry
            parser_cls = ToolParserManager.get_tool_parser(parser_name)
            if parser_cls is None:
                raise ValueError(f"Unknown vLLM parser: {parser_name}. Available: {list(VLLM_TOOL_PARSERS)}")
            
            self._vllm_parser = parser_cls(tokenizer)
            print(f"[VLLMToolParser] Using vLLM tool parser: {parser_name}", flush=True)
            
            # Try to get tool call tokens from the parser
            if hasattr(self._vllm_parser, 'tool_call_start_token'):
                self.tool_call_begin = self._vllm_parser.tool_call_start_token
            if hasattr(self._vllm_parser, 'tool_call_end_token'):
                self.tool_call_end = self._vllm_parser.tool_call_end_token
                
        except ImportError as e:
            raise ImportError(f"vLLM is required for VLLMToolParser: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM parser '{parser_name}': {e}")

    def parse(self, model_response: str, tools: list[dict] = None) -> list[ToolCall]:
        """Parse tool calls using vLLM's parser.
        
        This method mimics vLLM's serving_chat.py behavior by creating a 
        ChatCompletionRequest-like object with the tools field.
        
        Args:
            model_response: The model's response text containing tool calls
            tools: List of tool definitions in OpenAI format. Required for parsers 
                   like qwen3_coder that need tool schemas for parameter type conversion.
                   If None, all parameters will be treated as strings.
        
        Returns:
            List of ToolCall objects
        
        Note:
            For best results (matching vLLM inference behavior), always pass the 
            same tools that were used in the prompt generation.
        """
        from vllm.entrypoints.openai.protocol import ChatCompletionToolsParam, FunctionDefinition
        
        # Convert tools to vLLM's ChatCompletionToolsParam format (same as vLLM serving_chat.py)
        tool_params = None
        if tools:
            tool_params = []
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    func = tool["function"]
                    func_def = FunctionDefinition(
                        name=func.get("name", ""),
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {})
                    )
                    tool_param = ChatCompletionToolsParam(type="function", function=func_def)
                    tool_params.append(tool_param)
        
        # Create a mock request object matching vLLM's ChatCompletionRequest interface
        # vLLM's tool_parser.extract_tool_calls(content, request) expects request.tools
        class MockRequest:
            def __init__(self, tools):
                self.tools = tools
        
        request = MockRequest(tool_params)
        
        result = self._vllm_parser.extract_tool_calls(model_response, request)
        tool_calls = []
        for tc in result.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append(ToolCall(name=tc.function.name, arguments=args))
        return tool_calls

    def get_tool_prompt(self, tools_schema: str) -> str:
        """Not used - tokenizer.apply_chat_template handles this."""
        raise NotImplementedError("VLLMToolParser uses tokenizer.apply_chat_template for tool prompts")
    
    def __repr__(self) -> str:
        return f"VLLMToolParser(parser_name='{self.parser_name}')"
    
    @staticmethod
    def list_available_parsers() -> list[str]:
        """List all available vLLM tool parsers."""
        print("Available vLLM tool parsers:")
        print("  Common choices:")
        print("    - hermes      : Qwen3-Instruct, Hermes models (default)")
        print("    - qwen3_coder : Qwen3-Coder models")
        print("    - llama3_json : Llama 3 models")
        print("    - mistral     : Mistral models")
        print("    - deepseek_v3 : DeepSeek V3 models")
        print(f"  All parsers: {VLLM_TOOL_PARSERS}")
        return VLLM_TOOL_PARSERS
