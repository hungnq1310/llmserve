import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import os
import re
import numpy as np

from PIL import Image
import torch, base64
import cv2
# from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration

from dotenv import load_dotenv
import os


# MODEL_PATH = "/home/admin/doku/TensorRT_LLM/models/test/ggml-model-Q4_K_M.bin"
# MODEL_PATH = "/home/tiennv/phucth/EraX-VL-7B-V1.0/ERAX_VL_7B_Q4_K_M.gguf"

# llm = Qwen2VLForConditionalGeneration.from_pretrained(
#     model="/home/tiennv/phucth/EraX-VL-7B-V1.0",
#     tokenizer="erax-ai/EraX-VL-7B-V1.0",
#     load_in_4bit=True
#     # kwargs={"gguf_file": MODEL_PATH}
# )

from transformers import BitsAndBytesConfig


min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # hoặc load_in_4bit=True cho các mô hình lớn hơn
)
# model_path = "/home/tiennv/phucth/EraX-VL-7B-V1.0"

# tokenizer = AutoTokenizer.from_pretrained("/home/tiennv/phucth/EraX-VL-7B-V1.0")
# processor = AutoProcessor.from_pretrained(
#      pretrained_model_name_or_path="/home/tiennv/phucth/EraX-VL-7B-V1.0",
#      min_pixels=min_pixels,
#      max_pixels=max_pixels,
#  )

# llm = Qwen2VLForConditionalGeneration.from_pretrained(
#     pretrained_model_name_or_path="/home/tiennv/phucth/EraX-VL-7B-V1.0",
#     torch_dtype=torch.float16,
#     # tokenizer="erax-ai/EraX-VL-7B-V1.0",
#     # attn_implementation="flash_attention_2",
#     quantization_config=quantization_config
#     # kwargs={"gguf_file": MODEL_PATH}
# )

# biến từ file .env
load_dotenv()

model_path = os.getenv("MODEL_PATH")
load_in_8bit = os.getenv("LOAD_IN_8BIT") == "True"

# thay biến .env vào:
quantization_config = BitsAndBytesConfig(
    load_in_8bit=load_in_8bit  
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path=model_path,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)

llm = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_path,
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)


import torch
model = torch.compile(llm)
print(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
import time
# Gradio UI
def describe_image(image, question, max_length, temperature, top_p, rp):
    start_time = time.time()
    image = np.array(image.convert("RGB")) 
    buffered = cv2.imencode(".png", image)[1]

    encoded_image = base64.b64encode(buffered).decode('utf-8')
    base64_data = f"data:image;base64,{encoded_image}"
    print("Encoded image base64:", encoded_image[:100]) 

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64_data,
                },
                {
                    "type": "text",
                    "text": question
                },
            ],
        }
    ]

    tokenized_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("Tokenized Text:", tokenized_text)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[tokenized_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = model.device
    inputs = inputs.to(device)
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs.input_ids = inputs.input_ids.to(device)
    # inputsattention_mask = inputs.attention_mask.to(device)

    generation_config = model.generation_config
    generation_config.do_sample = True
    generation_config.temperature = temperature
    generation_config.top_k = 1
    generation_config.top_p = top_p
    generation_config.max_new_tokens = max_length
    generation_config.repetition_penalty = rp



    # generated_ids = model.generate(**inputs, generation_config=generation_config)
    generated_ids = model.generate(**inputs, generation_config = generation_config)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    generated_text = output_text[0]
    print(f"question:{question}, Generated text:{generated_text}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Thời gian thực thi: {elapsed_time:.2f} giây")
    return generated_text

# UI Gradio
with gr.Blocks() as demo:
    gr.Markdown("## OCR")
    with gr.Row():
        with gr.Column(scale=1):
            max_length = gr.Slider(
                label="Max Length",
                minimum=512,
                maximum=8192,
                value=2048,
                step=8,
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0,
                maximum=1,
                value=0.2,
                step=0.1,
            )
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                value=0.001,
                label="top_p",
            )
            rp = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                step=0.1,
                value=1.1,
                label="Repetition penalty",
            )
            question = gr.Textbox(
                label="Câu hỏi",
                value="nội dung bức ảnh này bằng tiếng Việt và định dạng json.",
                lines=3,
            )

        with gr.Column(scale=4):
            image_input = gr.Image(type="pil", label="Image Input", interactive=True)
            output_text = gr.Textbox(
                label="Response",
                lines=10,
                show_copy_button=True,
            )
    with gr.Row():
        submit = gr.Button(value="Submit")
        clear = gr.ClearButton([question, output_text])

    submit.click(fn=describe_image, 
                 inputs=[image_input, question, max_length, temperature, top_p, rp], 
                 outputs=[output_text])

if __name__ == "__main__":
    demo.launch(share=True, server_port = 5000) #