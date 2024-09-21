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


def get_anthropic_response(sys_message: str, human_msg: str, model: str):
    message = client.messages.create(
        max_tokens=4000,
        system=sys_message,
        messages=[{"role": "user", "content": human_msg}],
        model=model,
    )
    return message.content[0].text, convert_anthropic_tokens_to_cost(
        model, message.usage.input_tokens, message.usage.output_tokens
    )

MODEL = ta.Literal[
    "opus",
    "haiku",
    "sonnet",
    # "gpt-4-0125-preview",
    # "gpt-3.5-turbo-0125",
    # "gpt-4-turbo-2024-04-09",
    # "google",
    # "llama-3-70b",
    # "llama-3-8b",
]


def get_ai_response(messages: list[str], model: MODEL):
    if len(messages) != 2:
        raise ValueError(f"Invalid number of messages. Currently only implemented for 2 messages. Received {len(messages)}")
    if model == "opus":
        return get_anthropic_response(
            messages[0], messages[1], "claude-3-opus-20240229"
        )
    elif model == "haiku":
        return get_anthropic_response(
            messages[0], messages[1], "claude-3-haiku-20240307"
        )
    elif model == "sonnet":
        return get_anthropic_response(
            messages[0], messages[1], "claude-3-5-sonnet-20240620"
        )
    # elif (
    #     model == "gpt-4-0125-preview"
    #     or model == "gpt-3.5-turbo-0125"
    #     or model == "gpt-4-turbo-2024-04-09"
    #     or model == "gpt-4o"
    # ):
    #     ai_model = ChatOpenAI(model=model, temperature=0.8)
    #     with get_openai_callback() as cb:
    #         response = ai_model(messages).content
    #     return response, cb.total_cost
    else:
        raise ValueError("Invalid model:", model)
