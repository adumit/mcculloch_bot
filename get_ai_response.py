import typing as ta
import os

from anthropic import Anthropic

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
)

def convert_anthropic_tokens_to_cost(model, input_tokens, output_tokens):
    if model == "claude-3-haiku-20240307":
        output_cost_per_token = 1.25 / 1_000_000
        input_cost_per_token = 0.25 / 1_000_000
    elif model == "claude-3-sonnet-20240229" or model == "claude-3-5-sonnet-20240620":
        output_cost_per_token = 15 / 1_000_000
        input_cost_per_token = 3 / 1_000_000
    elif model == "claude-3-opus-20240229":
        output_cost_per_token = 75 / 1_000_000
        input_cost_per_token = 15 / 1_000_000
    else:
        raise ValueError("Model not supported")
    return input_tokens * input_cost_per_token + output_tokens * output_cost_per_token

def get_anthropic_response(sys_message: str, conversation_history: list[str], model: str, temperature: float):
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
        for i, msg in enumerate(conversation_history)
    ]
    
    message = client.messages.create(
        max_tokens=4000,
        system=sys_message,
        messages=messages,
        model=model,
        temperature=temperature,
    )
    return message.content[0].text, convert_anthropic_tokens_to_cost(
        model, message.usage.input_tokens, message.usage.output_tokens
    )

MODEL = ta.Literal[
    "opus",
    "haiku",
    "sonnet",
]

def get_ai_response(system_message: str, conversation_history: list[str], model: MODEL, temperature: float):
    if model == "opus":
        return get_anthropic_response(
            system_message, conversation_history, "claude-3-opus-20240229", temperature
        )
    elif model == "haiku":
        return get_anthropic_response(
            system_message, conversation_history, "claude-3-haiku-20240307", temperature
        )
    elif model == "sonnet":
        return get_anthropic_response(
            system_message, conversation_history, "claude-3-5-sonnet-20240620", temperature
        )
    else:
        raise ValueError("Invalid model:", model)
