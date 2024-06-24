import json
import ast
import os
from hdbcli import dbapi
from openai import AsyncOpenAI

import chainlit as cl

cl.instrument_openai()

api_key = os.environ.get("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

MAX_ITER = 5


def get_SAP_SQL_query(query):
    conn = dbapi.connect(
    address="111.93.61.252",
    port=30215,
    user="SAPHANADB",
    password="Final_1234"
)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    print(str(result))
    return str(result)


tools = [
    {
        "type": "function",
        "function": {
            "name": "SAP_SQL_query",
            "description": "Execute the SQL query used for ACDOCA table in SAP. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to be executed",
                    },
                },
                "required": ["query"],
            },
        },
    }
]


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful data analyst that gets the answer user wants. You can use the tool SAP_SQL_query to execute a specific query and get the desired answer. Just give the user the answer, not the actual SQL query or any other intermediate process. "}],
    )


@cl.step(type="tool")
async def call_tool(tool_call_id, name, arguments, message_history):
    arguments = ast.literal_eval(arguments)

    current_step = cl.context.current_step
    current_step.name = name
    current_step.input = arguments

    function_response = get_SAP_SQL_query(
        query=arguments.get("query"),
    )

    current_step.output = function_response
    current_step.language = "json"

    message_history.append(
        {
            "role": "function",
            "name": name,
            "content": function_response,
            "tool_call_id": tool_call_id,
        }
    )

async def call_gpt4(message_history):
    settings = {
        "model": "gpt-3.5-turbo",
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
    }

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    tool_call_id = None
    function_output = {"name": "", "arguments": ""}

    final_answer = cl.Message(content="", author="Answer")

    async for part in stream:
        new_delta = part.choices[0].delta
        tool_call = new_delta.tool_calls and new_delta.tool_calls[0]
        function = tool_call and tool_call.function
        if tool_call and tool_call.id:
            tool_call_id = tool_call.id

        if function:
            if function.name:
                function_output["name"] = function.name
            else:
                function_output["arguments"] += function.arguments
        if new_delta.content:
            if not final_answer.content:
                await final_answer.send()
            await final_answer.stream_token(new_delta.content)

    if tool_call_id:
        await call_tool(
            tool_call_id,
            function_output["name"],
            function_output["arguments"],
            message_history,
        )

    if final_answer.content:
        await final_answer.update()

    return tool_call_id


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    cur_iter = 0

    while cur_iter < MAX_ITER:
        tool_call_id = await call_gpt4(message_history)
        if not tool_call_id:
            break

        cur_iter += 1