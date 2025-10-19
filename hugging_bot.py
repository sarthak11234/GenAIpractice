import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command

# Hugging Face
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Load Hugging Face model & tokenizer
model_name = "microsoft/DialoGPT-large"  # You can switch to "distilgpt2" for faster/smaller
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Optional: enable GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Store previous conversation (optional)
conversation_history = ""


# Command handlers
@dp.message(Command(commands=["start", "help"]))
async def start_help(message: Message):
    help_text = (
        "Hi! I am your Telegram bot 🤖\n\n"
        "Commands:\n"
        "/start - start the bot\n"
        "/help - show this message\n"
        "/clear - clear conversation history\n"
        "Or just type anything to chat with me!"
    )
    await message.reply(help_text)


@dp.message(Command(commands=["clear"]))
async def clear_history(message: Message):
    global conversation_history
    conversation_history = ""
    await message.reply("Conversation history cleared!")


# Store previous conversation (optional)
# conversation_history as a list
conversation_history = []

@dp.message()
async def chat(message: Message):
    global conversation_history

    # Store user message
    conversation_history.append(f"User: {message.text}")

    # Keep last 4 turns
    prompt = "\n".join(conversation_history[-4:]) + "\nBot:"

    # Tokenize & generate
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1]+100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    response = response.split("\n")[0].split("User:")[0].strip()

    # Append bot response
    conversation_history.append(f"Bot: {response}")

    await message.reply(response)




# Start bot
async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
