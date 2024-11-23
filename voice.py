#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Any, Dict
import aiohttp
import os
import sys

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.anthropic import AnthropicLLMContext, AnthropicLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from runner import configure
from loguru import logger
from dotenv import load_dotenv

import dbutils
import anthropic
import base64

import tools

import asyncio
import cv2
from PIL import Image
import threading

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

video_participant_id = None

client = anthropic.Anthropic()


def wait_for(condition, img):
    system_prompt = f"""You are analyzing an image. You must answer if a condition has been met or not within the supplied image.
    Based on the objects or characteristics in the image, respond with "Yes" or "No". If you respond with "Yes", you must
    also describe what the condition is that has been met and where it is in the image. Otherwise, only respond with "No" and
    nothing else.

    EXAMPLE USER INPUT: "there is a blue object in the image"
    If you see a blue object in the image, you should respond with something like "Yes, there is a blue object in the image. It appears to be a blue water bottle on a desk and appears next to a notebook."
    If you do not see a blue object in the image, you should respond with "No".

    EXAMPLE USER INPUT: "there is a cat in the image"
    If you see a cat in the image, you should respond with something like "Yes, there is a cat in the image. It is sitting on a chair."
    If you do not see a cat in the image, you should respond with "No".

    REMEMBER, if the condition is not met only respond with "No." If the condition is met, respond with "Yes" and briefly describe the condition in two sentences.
    """

    with open("latest.jpeg", "rb") as image_file:
        image_data = image_file.read()

    base64_image = base64.b64encode(image_data).decode("utf-8")

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                    {"type": "text", "text": f"Condition: {condition}"},
                ],
            }
        ],
    )

    print(message.content[0].text)
    return message.content[0].text


async def wait_for_condition(
    function_name, tool_call_id, arguments, llm, context, result_callback
):
    condition = arguments["condition"]
    cam = cv2.VideoCapture(0)
    frame_counter = 0
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        # turn frame into base64
        cv2.resize(frame, (480, 270))

        if frame_counter % 30 == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = wait_for(condition, img)
            if "Yes" in result:
                await result_callback("Yes, the condition has been met.")
                break

        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()


async def get_current_image(
    function_name, tool_call_id, arguments, llm, context, result_callback
):
    logger.debug(f"!!! IN get_current_image {video_participant_id}, {arguments}")
    question = arguments["user_request"]
    await llm.request_image_frame(user_id=video_participant_id, text_content=question)


async def recall_item(
    function_name, tool_call_id, arguments, llm, context, result_callback
):
    item = arguments["item"]
    user_query = arguments["user_query"]

    filename = dbutils.search(item)
    response = dbutils.getResponse(filename, user_query)

    await result_callback(response)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = AnthropicLLMService(api_key=os.getenv("ANTHROPIC_API_KEY"))
        llm.register_function("get_current_image", get_current_image)
        llm.register_function("recall_item", recall_item)
        llm.register_function("wait_for_condition", wait_for_condition)

        tools = [
            {
                "name": "get_current_image",
                "description": "This will get the current user's image from the video stream. This tool should be used any time the user asks a question that may need a visual response.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_request": {
                            "type": "string",
                            "description": "The user's request that requires a visual response. For example, 'What do you see in front of you?'",
                        },
                    },
                    "required": ["user_request"],
                },
            },
            {
                "name": "recall_item",
                "description": "Recall an item that the user is looking for",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "item": {
                            "type": "string",
                            "description": "The item that the user is looking for",
                        },
                        "user_query": {
                            "type": "string",
                            "description": "The user's full query for what they are looking for. For example, 'where did I leave my glasses?'",
                        },
                    },
                    "required": ["item"],
                },
            },
            {
                "name": "wait_for_condition",
                "description": "This tool is used to wait for a condition to be met in an image. The tool will keep asking the AI to analyze the image until the condition is met. This tool should be used if the user says something like 'Tell me when you see something blue'.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "condition": {
                            "type": "string",
                            "description": "The condition to wait for in the image. For example, if the user says 'Tell me when you see a cat', the condition will be 'a cat appears in the image'.",
                        }
                    },
                    "required": ["item"],
                },
            },
        ]

        system_prompt = """\
You are a helpful assistant who converses with a user and answers questions. Respond concisely to general questions.

Your response will be turned into speech so use only simple words and punctuation.

You have access to these tools:
- get_current_image: You can use this tool to get the current user's image from the video stream. This tool should be used any time the user asks a question that may need a visual response. This should be the basic tool you use whenever a visual task is needed.
- recall_item: This is used whenever the user asks you to recall an item. This tool is used for recalling items that the user is looking for from the past.
- wait_for_condition: This tool is used to wait for a condition to be met in an image. The tool will keep asking the AI to analyze the image until the condition is met. This tool should be used if the user says something like 'Tell me when you see something blue'. When you use this tool, do not say anything to the user. Just wait for the condition to be met and then respond to the user.

- Your output is directly streamed to the user.
- Do not mention any tools you are using
- Only use the tool, and do not say anything before using the tool.
- Remember to be brief in your messages. Only say two sentences maximum.
"""
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        context = AnthropicLLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                # context_aggregator.user(),
                # llm,
                # tts,
                transport.output(),
                # context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            global video_participant_id
            video_participant_id = participant["id"]
            await transport.capture_participant_transcription(participant["id"])
            await transport.capture_participant_video(video_participant_id, framerate=0)
            # Kick off the conversation.
            await tts.say("Hi! Ask me about anything!")

        @transport.event_handler("on_transcription_message")
        async def on_transcription_message(transport, message: Dict[str, Any]):
            participant_id = message.get("participantId")
            text = message.get("text")
            is_final = message["rawResponse"]["is_final"]
            logger.info(
                f"Transcription from {participant_id}: {text} (final: {is_final})"
            )

            if is_final:
                tools.use_user_input(text)

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
