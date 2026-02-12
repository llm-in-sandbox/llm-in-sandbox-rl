from rllm.parser.chat_template_parser import ChatTemplateParser, DeepseekQwenChatTemplateParser, LlamaChatTemplateParser, QwenChatTemplateParser
from rllm.parser.tool_parser import VLLMToolParser, R1ToolParser, ToolParser

__all__ = [
    "ChatTemplateParser",
    "DeepseekQwenChatTemplateParser",
    "QwenChatTemplateParser",
    "LlamaChatTemplateParser",
    "ToolParser",
    "R1ToolParser",
    "VLLMToolParser",
]

PARSER_REGISTRY = {
    "r1": R1ToolParser,
    "hermes": VLLMToolParser,      # Qwen3-Instruct uses hermes format
    "qwen3_coder": VLLMToolParser, # Qwen3-Coder
    "qwen": VLLMToolParser,        # Alias for backward compatibility
}


def get_tool_parser(parser_name: str) -> type[ToolParser]:
    assert parser_name in PARSER_REGISTRY, f"Tool parser {parser_name} not found in {PARSER_REGISTRY}"
    return PARSER_REGISTRY[parser_name]
