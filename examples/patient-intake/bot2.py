#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import openai

from pipecat.frames.frames import (
    EndFrame, LLMMessagesFrame, ImageRawFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.services.openai import OpenAITTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from supabase import create_client, Client

from runner import configure

from PIL import Image

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

prompt = '''
Your job is to provide medical recommendations about patients.
You will be speaking to a rural midwife or healthcare practitioner who is not medically trained.
If any appointments are requested to be scheduled, say thank you and note that they will be scheduled.
Remember to stick to WHO guidelines.
'''
### SUPABASE SETUP
supabaseUrl: str = os.environ.get("SUPABASE_URL")
supabaseKey: str = os.environ.get("SUPABASE_ANON_KEY")
supabase = create_client(supabaseUrl, supabaseKey)

### LOADING IMAGE FROM FILE
scriptDir = os.path.dirname(__file__)
def loadImage(imgFile):
    imgPath = os.path.join(scriptDir, "../patient-intake", imgFile)
    with Image.open(imgPath) as img:
        image = ImageRawFrame(image=img.tobytes(), size=img.size, format=img.format)
    return image

### SETTING BOT CAMERA SIZE TO BE EQUAL TO THE IMAGE SIZE
botImage = loadImage("gravidapfp.jpg")
imageWidth = botImage.size[0]
imageHeight = botImage.size[1]

async def summarise(messages: dict):
    prompt = f"""
    Your job is to provide medical recommendations about patients. 
    You will be speaking to a rural midwife or healthcare practitioner who is not medically trained.
    The patient has had a call with a medical assistant. Here is a transcript of the call:

    {messages}

    Provide a detailed summary which can be displayed to the practitioner to help aid them in caring for the patient.
    Do not include patient medical history or patient details, only include intervention and future care advice.
    Do not include any mention of the physician's name, follow-up questions or follow-up appointments.
    Do not include any mention of the summary being provided on the Gravida app.
    """
    summary = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
            {
            "role": "system",
            "content": prompt,
            },
        ],
    )
    
    return (summary.choices[0].message.content) # returns the response of the LLM to the above prompt


async def main(room_url: str, token, patient: str):
    logger.debug("PATIENT DETAILS:" + patient)
    print("PATIENT DETAILS:" + patient)
    transport = DailyTransport(
        room_url,
        token,
        "Gravida AI",
        DailyParams(
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            transcription_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            camera_out_enabled=True,
            camera_out_height=imageWidth,
            camera_out_width=imageHeight
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
            "content": "These are the patient details: " + patient + "Based on our models, the patient is predicted to have severe preeclampsia."      
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

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, arg):
        patientID = eval(patient) ["id"] # takes patient JSON string, converts it to dictionary, and retrieves id
        summary = await summarise(messages) # retrieves summary and patientID
        supabase.table("patient summaries").insert({"id":patientID, "summary":summary}).execute() # creates new row in table
        #print(summary)

        await task.queue_frame(EndFrame()) # leave the call
         
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        transport.capture_participant_transcription(participant["id"])
        # Kick off the conversation.
        messages.append(
            {"role": "system", "content": "Start by introducing yourself and ask the user for their name politely."
             })
        
        messages.append({"role": "system", "content":"Ask if you have the right patients name."})

        messages.append({"role": "system", "content":"If so, then ask what advice the user would like about the patient."})

        messages.append({"role": "system", "content": """
If they ask for a management plan then
Provide a short and concise plan of action in a few bullet points referring to patient details when relevant.
Briefly advise on which symptoms to look out for that might need an emergency hospital admission.
After giving that short summary, indicate to the user that they will be given a general outline of the conversation in the Gravida app.
"""})

        await task.queue_frames([botImage, LLMMessagesFrame(messages)])

    runner = PipelineRunner()

    await runner.run(task)

if __name__ == "__main__":
    (url, token, patient) = configure()
    asyncio.run(main(url, token, patient))
