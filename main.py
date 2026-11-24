import json
import os
import sys

import arabic_reshaper
from bidi.algorithm import get_display
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

from tools import get_current_time, get_weather

# Step 1: Load API settings from the .env file into environment variables.
load_dotenv()

# Step 2: Create the OpenAI client using the API key and base URL.
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model = os.getenv("MODEL")

missing_settings = [
    name
    for name, value in (
        ("API_KEY", api_key),
        ("BASE_URL", base_url),
        ("MODEL", model),
    )
    if not value
]

if missing_settings:
    print(
        "Missing environment setting(s): " + ", ".join(missing_settings),
        file=sys.stderr,
    )
    print("Copy .env.example to .env and configure the values.", file=sys.stderr)
    raise SystemExit(1)

client = OpenAI(api_key=api_key, base_url=base_url)

available_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current time in the Asia/Baghdad timezone, formatted as "
                "YYYY-MM-DD HH:MM:SS."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather conditions for a city or named location. "
                "Use the location provided by the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or named location to look up.",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
]

# Step 3: Keep the system instruction and every message in a Python list.
conversation_history = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the conversation history to answer "
            "follow-up questions and remember details shared earlier in this conversation. "
            "Do not claim you cannot remember earlier messages in this conversation. "
            "You do not retain memory after the program ends."
            "Answer in the same language the user asks the question. If the user asks a question in Arabic language respond with Arabic language: and same in any other language."
        ),
    }
]

# Step 4: Keep accepting questions until the user interrupts the program.
try:
    while True:
        # Step 5: Read the user's question and add it to the conversation history.
        question = input("Ask a question 🙂 (Ctrl+C to exit): ")
        history_start = len(conversation_history)
        conversation_history.append({"role": "user", "content": question})

        # Step 6: Send the conversation and available tools to the selected model.
        try:
            # Continue until the model returns a user-facing answer.
            while True:
                response = client.chat.completions.create(
                    model=model,
                    messages=conversation_history,
                    tools=available_tools,
                    tool_choice="auto",
                )
                message = response.choices[0].message
                tool_calls = message.tool_calls or []

                if not tool_calls:
                    answer = message.content
                    break

                # Preserve the assistant's tool request before replying with its result.
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in tool_calls
                        ],
                    }
                )

                for tool_call in tool_calls:
                    try:
                        arguments = json.loads(
                            tool_call.function.arguments or "{}"
                        )
                        if not isinstance(arguments, dict):
                            raise ValueError("Tool arguments must be a JSON object.")

                        if tool_call.function.name == "get_current_time":
                            if arguments:
                                raise ValueError(
                                    "get_current_time does not accept any arguments."
                                )
                            tool_result = get_current_time()
                        elif tool_call.function.name == "get_weather":
                            if (
                                set(arguments) != {"location"}
                                or not isinstance(arguments["location"], str)
                            ):
                                raise ValueError(
                                    "get_weather requires a string location argument."
                                )
                            tool_result = get_weather(arguments["location"])
                        else:
                            raise ValueError(
                                "Unsupported tool requested: "
                                f"{tool_call.function.name}"
                            )
                    except (ValueError, RuntimeError) as error:
                        print(f"Tool call failed: {error}", file=sys.stderr)
                        tool_result = f"Tool call failed: {error}"

                    conversation_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )
        except (APIConnectionError, APIStatusError) as error:
            del conversation_history[history_start:]
            print(
                f"Could not reach the configured model service at {base_url}: {error}",
                file=sys.stderr,
            )
            continue

        # Step 7: Save and display the model's response for the next turn.
        if not answer:
            print("The model returned an empty response.", file=sys.stderr)
            continue
        conversation_history.append({"role": "assistant", "content": answer})
        reshaped_answer = arabic_reshaper.reshape(answer)
        print(f"😎Assistant: {get_display(reshaped_answer)}\n")
except (KeyboardInterrupt, EOFError):
    print("\nGoodbye!")
