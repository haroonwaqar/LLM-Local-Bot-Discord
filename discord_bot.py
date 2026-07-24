import discord
import os
from dotenv import load_dotenv
from llm import ask_llm
import asyncio

load_dotenv()
Bot_token = os.getenv('BOT_TOKEN')

intents = discord.Intents.default()
intents.message_content = True  # required to read message text

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}. Listening for messages...")

@client.event
async def on_message(message):
    if message.author == client.user:
        return  # ignore the bot's own messages

    print(f"[{message.author}] (channel={message.channel}): {message.content}")

    user_question = message.content
    answer = await asyncio.to_thread(ask_llm, user_question)
    await message.channel.send(answer) # sends the reply
    print("Message Answered") 

client.run(Bot_token)

