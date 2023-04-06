import os
import sys
import time
import gradio as gr
from videocrafter_test import Text2Video
sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))

def videocrafter_demo(result_dir='./tmp/'):
    text2video = Text2Video(result_dir)
    with gr.Blocks(analytics_enabled=False) as videocrafter_iface:
        with gr.Row().style(equal_height=False):
            with gr.Tab(label="VideoCrafter"):
                input_text = gr.Text()
                model_choices=['origin','vangogh','frozen','yourname', 'coco']
                trigger_word_list=[' ','Loving Vincent style', 'frozenmovie style', 'MakotoShinkaiYourName style', 'coco style']

                with gr.Row():
                    model_index = gr.Dropdown(label='Models', elem_id=f"model", choices=model_choices, value=model_choices[0], type="index",interactive=True)
                    trigger_word=gr.Dropdown(label='Trigger Word', elem_id=f"trigger_word", choices=trigger_word_list, value=trigger_word_list[0], interactive=True)

                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=200, step=1, elem_id=f"steps", label="Sampling steps", value=50)
                    eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="eta")

                with gr.Row():
                    lora_scale = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, label='Lora Scale', value=1.0, elem_id="lora_scale")
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=15.0, elem_id="cfg_scale")

                send_btn = gr.Button("Send")

            with gr.Column():
                output_video_1 = gr.PlayableVideo()

            send_btn.click(
                fn=text2video.get_prompt, 
                inputs=[
                    input_text,
                    steps,
                    model_index,
                    eta,
                    cfg_scale,
                    lora_scale,
                    trigger_word
                ],
                outputs=[output_video_1],
            )
    return videocrafter_iface

if __name__ == "__main__":
    result_dir = os.path.join('./', 'results')
    videocrafter_iface = videocrafter_demo(result_dir)
    videocrafter_iface.launch()