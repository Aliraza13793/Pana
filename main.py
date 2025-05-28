import chainlit as cl
import asyncio
from typing import cast
from agents.extensions.visualization import draw_graph
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled,RunConfig
from openai.types.responses import ResponseTextDeltaEvent
from agents.run import RunConfig
from dotenv import load_dotenv, find_dotenv

import os

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")
# set_tracing_disabled(disabled=True)

@cl.on_chat_start
async def start():

#Reference: https://ai.google.dev/gemini-api/docs/openai
    client = AsyncOpenAI( 
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",)

    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client)

    config=RunConfig(
        model=model,
        model_provider=client,
        tracing_disabled=True
)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    
#planner
    planner_expert=Agent(
         name="Planner Expert",
        instructions="You are a planner expert agent that can answer questions about planner.",
        model= model)

    Agentic_ai_expert=Agent(
        name="Agentic AI Expert",
        instructions="You are a agentic ai expert agent that can answer questions about agentic ai.",
        model=model,
        tools=[planner_expert.as_tool(tool_name="Agenticworkflow",
                                  tool_description="You are check all the tools of agents")]
)
# Web expertr
    Web_expert=Agent(
        name="Web Expert Agent",
        instructions="You are a web expert agent that can answer questions about web development from panacloud.",
 
        handoffs=[Agentic_ai_expert]
)
# mbl agnet
    Mbl_agent=Agent(
        name="Mbl Agent",
        instructions="You are a mbl agent that can answer questions about mbl.",

        handoffs=[Agentic_ai_expert]
)
#Web experrt
    panacloud_agent=Agent(
        name="Panacloud",
        instructions="You are a helpful assistant that can answer questions about Panacloud you will be given a question and you will need to answer it.",
        handoffs=[Web_expert, Mbl_agent,Agentic_ai_expert]
)
    cl.user_session.set("panacloud",panacloud_agent)
    
    await cl.Message(content="Welcome to the Panaloud Ai Assistant! How can i help you today?").send()

@cl.on_message
async def  main(message: cl.Message):
    msg= cl.Message(content="")
    await msg.send()
    
    panacloud_agent: Agent = cast(Agent,cl.user_session.get("panacloud"))
    config:RunConfig= cast(RunConfig,cl.user_session.get("config"))
    history=cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content":message.content})
    
    try:
        print("\n[CALLING_AGENT_WITH_CONTECT]\n", history, "\n")
        result= Runner.run_sync(starting_agent=panacloud_agent,input=history, run_config=config)
        response_content= result.final_output
        
        #simulate streaming by sending tokens one by one
        for token in response_content.split():
            await msg.stream_token(token + " ")
        await msg.update()
        
        #update the chat history with the assistant, response
        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("chat_history",history)
        
        #log the interaction
        
        print (f"User: {message.content}")
        print(f"Assistant:  {response_content}")
    except Exception as e:
        await msg.stream_token(f"\nError: {str(e)}")
        await msg.update()
        print(f"error: {str(e)}")

    

# if __name__=="__main__":
#     asyncio.run(main()
    
    
    