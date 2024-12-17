import asyncio
import os
from threading import Thread

import chainlit as cl
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# Initialize Hugging Face model
load_dotenv()
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # "openai-community/gpt2"#"Qwen/Qwen2.5-Coder-7B"  #
TOKENIZER_NAME = MODEL_NAME  # Use the same tokenizer as the model
model = None
tokenizer = None


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="LLama 3.1",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
            icon="https://img-cdn.inc.com/image/upload/f_webp,c_fit,w_828,q_auto/images/panoramic/meta-llama3-inc_539927_dhgoal.jpg",
        ),
        cl.ChatProfile(
            name="GPT-4",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://picsum.photos/250",
        ),
    ]


def load_llama_once():
    global model, tokenizer, generator
    if model and tokenizer:
        return True
    print("Loading model and tokenizer...")
    try:
        login(token=os.getenv("hf_token"))
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Match input dtype for fast inference
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True)

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            use_fast=True,
            device_map="auto",
            # torch_dtype=torch.float16,
            # quantization_config=quantization_config,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            # torch_dtype=torch.float16,
            quantization_config=quantization_config,
        )

        print("Model and tokenizer loaded successfully!")

        return True
    except Exception:
        return False


@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile == "LLama 3.1":
        if load_llama_once():
            await cl.Message(
                content=f"Hello, I am {chat_profile}, how can I help you today?"
            ).send()
            cl.user_session.set(
                "message_history",
                [{"role": "system", "content": "You are a helpful assistant."}],
            )
        else:
            await cl.Message(
                content=f"Error loading {chat_profile}, sorry for the troubles!"
            ).send()


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    global model, tokenizer

    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})
    # dummy_assistant = {"role": "assistant", "content": ""}
    # message_history.append(dummy_assistant)

    msg = cl.Message(content="")

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    # cleaned_history = [
    #     {"role": msg["role"], "content": msg["content"]}
    #     for msg in message_history
    #     if msg["role"] in ["user", "assistant"]
    # ]

    prompt = message_history
    inputs = tokenizer.apply_chat_template(
        prompt, return_tensors="pt", tokenize=True, add_generation_prompt=True
    ).to(model.device)

    def generate():
        with torch.no_grad():
            model.generate(
                inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=128,
                streamer=streamer,
                temperature=0.7,
                # top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                do_sample=True,
            )

    thread = Thread(target=generate)
    thread.start()

    streamed_response = ""
    async for token in async_streamer_wrapper(streamer):
        if token.startswith("system"):
            continue
        streamed_response += token
        # await cl.Message(content=streamed_response).update()
        await msg.stream_token(token)  # cl.Message(content=streamed_response).send()

    # message_history.remove(dummy_assistant)
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()


async def async_streamer_wrapper(streamer):
    """
    Wrap the TextIteratorStreamer into an async generator.
    """
    for token in streamer:
        await asyncio.sleep(0)  # Yield control to the event loop
        yield token
