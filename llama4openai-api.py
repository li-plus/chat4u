# Adapted from https://gist.github.com/kinoc/8a042d8c5683725aa8c372274c02ea2f
import time

import torch
from flask import Flask, request
from flask.json import jsonify
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

app = Flask(__name__)

# ===== ARGUMENTS =====
model_name_or_path = "llama-wechat/"
device = "cuda"

# set up the llama model
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
model = LlamaForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()


def generate_prompt_llama(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def evaluate_llama(
    instruction,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt_llama(instruction)
    print(f"prompt: {prompt}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    print(f"top_k: {top_k}")
    print(f"num_beams: {num_beams}")
    print(f"max_new_tokens: {max_new_tokens}")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    print(f"output: {output}")
    gen_text = output.split("### Response:")[1].strip()
    print(f"gen_text: {gen_text}")
    return gen_text


def decode_kwargs(data):
    kwargs = {}
    if "n" in data:
        kwargs["num_return_sequences"] = data["n"]
    if "stop" in data:
        kwargs["early_stopping"] = True
        kwargs["stop_token"] = data["stop"]
    if "suffix" in data:
        kwargs["suffix"] = data["suffix"]
    if "presence_penalty" in data:
        kwargs["presence_penalty"] = data["presence_penalty"]
    if "frequency_penalty" in data:
        kwargs["repetition_penalty"] = data["frequency_penalty"]
    if "repetition_penalty " in data:
        kwargs["repetition_penalty"] = data["repetition_penalty "]
    if "best_of " in data:
        kwargs["num_return_sequences"] = data["best_of "]

    return kwargs


@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True)
    model_name = data["model"]
    messages = data["messages"]

    prompt = messages[-1]["content"]

    max_tokens = data.get("max_tokens", 16)
    temperature = data.get("temperature", 1.0)
    top_p = data.get("top_p", 0.75)
    top_k = data.get("top_k", 40)
    num_beams = data.get("num_beams", 1)
    max_new_tokens = data.get("max_new_tokens", 256)

    kwargs = decode_kwargs(data)

    generated_text = evaluate_llama(
        prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(generated_text))
    total_tokens = prompt_tokens + completion_tokens
    return jsonify(
        {
            "object": "chat.completion",
            "id": "dummy",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"message": {"role": "assistant", "content": generated_text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
    )


if __name__ == "__main__":
    app.run()
