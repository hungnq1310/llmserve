import torch
import gradio as gr
# import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
from polyglot.detect import Detector
from vllm import LLM, SamplingParams

# MODEL_PATH = "/home/admin/doku/TensorRT_LLM/models/test/ggml-model-Q4_K_M.bin"
MODEL_PATH = "./LLaMAX3-8B-Alpaca.Q4_K_M.gguf"

llm = LLM(
    model=MODEL_PATH,
    tokenizer="LLaMAX/LLaMAX3-8B-Alpaca",
    gpu_memory_utilization=0.95,
)

def lang_detector(text):
    min_chars = 5
    if len(text) < min_chars:
        return "Input text too short"
    try:
        detector = Detector(text).language
        lang_info = str(detector)
        code = re.search(r"name: (\w+)", lang_info).group(1)
        return code
    except Exception as e:
        return f"ERROR：{str(e)}"

def Prompt_template(inst, prompt, query, src_language=None, trg_language=None):
    # inst = inst.format(src_language=src_language, trg_language=trg_language)
    instruction = f"`{inst}`"
    prompt = (
        f'{prompt}'
        f'### Instruction:\n{instruction}\n'
        f'### Input:\n{query}\n### Response:'
    )
    return prompt

# Unfinished
def chunk_text():
    pass
    

def translate(
    source_text: str, 
    # source_lang: str,
    # target_lang: str,
    inst: str, 
    prompt: str, 
    max_length: int,
    temperature: float,
    top_p: float,
    rp: float):
    
    print(f'Text is - {source_text}')
    
    # conversation = Prompt_template(inst, prompt, source_text, source_lang, target_lang)
    conversation = Prompt_template(inst, prompt, source_text)
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print(conversation)
    generate_kwargs = dict(
        # input_ids=input_ids,
        max_tokens=max_length, 
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rp,    
    )

    sampling_params = SamplingParams(**generate_kwargs)

    outputs = llm.generate(conversation, sampling_params)
    
    # resp = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # yield resp[len(prompt):]
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    return generated_text

CSS = """
    h1 {
        text-align: center;
        display: block;
        height: 10vh;
        align-content: center;
    }
    footer {
        visibility: hidden;
    }
"""

# LICENSE = """
# Model: <a href="https://huggingface.co/LLaMAX/LLaMAX3-8B-Alpaca">LLaMAX3-8B-Alpaca</a>
# """

# LANG_LIST = ['Akrikaans', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Asturian', 'Azerbaijani', \
#              'Belarusian', 'Bengali', 'Bosnian', 'Bulgarian', 'Burmese', \
#              'Catalan', 'Cebuano', 'Simplified Chinese', 'Traditional Chinese', 'Croatian', 'Czech', \
#              'Danish', 'Dutch', 'English', 'Estonian', 'Filipino', 'Finnish', 'French', 'Fulah', \
#              'Galician', 'Ganda', 'Georgian', 'German', 'Greek', 'Gujarati', \
#              'Hausa', 'Hebrew', 'Hindi', 'Hungarian', \
#              'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian', \
#              'Japanese', 'Javanese', \
#              'Kabuverdianu', 'Kamba', 'Kannada', 'Kazakh', 'Khmer', 'Korean', 'Kyrgyz', \
#              'Lao', 'Latvian', 'Lingala', 'Lithuanian', 'Luo', 'Luxembourgish', \
#              'Macedonian', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', \
#              'Nepali', 'Northern', 'Norwegian', 'Nyanja', \
#              'Occitan', 'Oriya', 'Oromo', \
#              'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', \
#              'Romanian', 'Russian', \
#              'Serbian', 'Shona', 'Sindhi', 'Slovak', 'Slovenian', 'Somali', 'Sorani', 'Spanish', 'Swahili', 'Swedish', \
#              'Tajik', 'Tamil', 'Telugu', 'Thai', 'Turkish', \
#              'Ukrainian', 'Umbundu', 'Urdu', 'Uzbek', \
#              'Vietnamese', 'Welsh', 'Wolof', 'Xhosa', 'Yoruba', 'Zulu']

chatbot = gr.Chatbot(height=600)

with gr.Blocks(theme="soft", css=CSS) as demo:
    # gr.Markdown(TITLE)
    with gr.Row():
        with gr.Column(scale=1):
            # source_lang = gr.Textbox(
            #     label="Source Lang(Auto-Detect)",
            #     value="English",
            # )
            # target_lang = gr.Dropdown(
            #     label="Target Lang",
            #     value="Spanish",
            #     choices=LANG_LIST,
            # )
            max_length = gr.Slider(
                label="Max Length",
                minimum=512,
                maximum=8192,
                value=4096,
                step=8,
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0,
                maximum=1,
                value=0.3,
                step=0.1,
            )
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=1.0,
                label="top_p",
            )
            rp = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                step=0.1,
                value=1.2,
                label="Repetition penalty",
            )
            
            inst = gr.Textbox(
                label="Instruction",
                value="Answering these following question.",
                lines=3,
            )
            prompt = gr.Textbox(
                label="Prompt",
                value=""" 'Below is an instruction that describes a task, paired with an input that provides further context. '
                'Write a response that appropriately completes the request.\n' """,
                lines=8,
            )
                
        with gr.Column(scale=4):
            source_text = gr.Textbox(
                label="Question",
                value="LLaMA is a language model with powerful multilingual capabilities without loss instruction-following capabilities. "+\
                "LLaMA supports many tasks between more than 100 languages, "+\
                "surpassing the performance of similarly scaled LLMs.",
                lines=10,
            )
            output_text = gr.Textbox(
                label="Response",
                lines=10,
                show_copy_button=True,
            )
    with gr.Row():
        submit = gr.Button(value="Submit")
        clear = gr.ClearButton([source_text, output_text])
    # gr.Markdown(LICENSE)
    
    # source_text.change(lang_detector, source_text, source_lang)
    # source_text.change(lang_detector, source_text)
    # submit.click(fn=translate, inputs=[source_text, source_lang, target_lang, inst, prompt, max_length, temperature, top_p, rp], outputs=[output_text])
    submit.click(fn=translate, inputs=[source_text, inst, prompt, max_length, temperature, top_p, rp], outputs=[output_text])

    with gr.Accordion("Tutorials", open=True):
        gr.Markdown("## Instruction")
        gr.Textbox(value="For examples: Supposed that you are the professor in education.",
                label="The instruction that describes a command.")

        gr.Markdown("\n\n\n### Prompt: Format template used to achieve different tasks.")
        gr.Textbox(value="Summarize articles and concepts into quick and easy-to-read summaries", label="1. Text Summarization")
        gr.Textbox(value="Translate text from one language to another",
                    label="2. Translation")
        gr.Textbox(value="Answer questions based on a given context",
                    label="3. Question Answering")

        gr.Markdown("\n\n\n### Question: The input text that you want to ask.")
        gr.Markdown("### Generation Configuration:")
        gr.Textbox(value="The maximum length of the generated text.", label="1. Max Length")
        gr.Textbox(value="Lower values make the output more deterministic, while higher values increase randomness.", label='2. Temperature')
        gr.Textbox(value="Lower values make the model consider fewer options, while higher values allow more diverse outputs.", label="3. Top_p")
        gr.Textbox(value="Lower values make the model consider repeated tokens less likely to be chosen again", label="4. Repetition Penalty")
        

if __name__ == "__main__":
    demo.launch(share=True)