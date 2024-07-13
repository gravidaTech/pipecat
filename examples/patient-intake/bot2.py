#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.services.openai import OpenAITTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

prompt = '''
Your job is to provide medical recommendations about patients.
You will be speaking to a rural midwife or healthcare practitioner who is not medically trained.
Your job is to provide a management plan based on the World Health Organisation guidelines.
'''

async def main(room_url: str, token, patient: str):
    transport = DailyTransport(
        room_url,
        token,
        "Gravida AI",
        DailyParams(
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            transcription_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer()
        )
    )

    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="alloy"
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM called GravidaAI in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "system",
            "content": "Extract the patient details from this json: " + patient 
        }
    ]

    tma_in = LLMUserResponseAggregator(messages)
    tma_out = LLMAssistantResponseAggregator(messages)

    pipeline = Pipeline([
        transport.input(),   # Transport user input
        tma_in,              # User responses
        llm,                 # LLM
        tts,                 # TTS
        transport.output(),  # Transport bot output
        tma_out              # Assistant spoken responses
    ])

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        transport.capture_participant_transcription(participant["id"])
        # Kick off the conversation.
        messages.append(
            {"role": "system", "content": """
Start by introducing yourself.
Briefly go over the key details of the patient at hand.
Provide a short and consise plan of action in a few bullet points.
After giving that short summary, indicate to the user that they will be given a summary in the Gravida app."""
             })
        await task.queue_frames([LLMMessagesFrame(messages)])

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    (url, token, patient) = configure()
    asyncio.run(main(url, token, patient))
