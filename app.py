import os
import sys
import time
import gradio as gr
from videocrafter_test import Text2Video
sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))

def videocrafter_demo(result_dir='./tmp/'):
    text2video = Text2Video(result_dir)
    with gr.Blocks(analytics_enabled=False) as videocrafter_iface:
        gr.Markdown("<div align='center'> <h2> VideoCrafter: A Toolkit for Text-to-Video Generation and Editing </span> </h2> \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/VideoCrafter/VideoCrafter'> Github </div>")
        with gr.Row().style(equal_height=False):
            with gr.Tab(label="VideoCrafter"):
                input_text = gr.Text(label='Prompts')
                model_choices=['origin','vangogh','frozen','yourname', 'coco']

                with gr.Row():
                    model_index = gr.Dropdown(label='Models', elem_id=f"model", choices=model_choices, value=model_choices[0], type="index",interactive=True)
                    
                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=200, step=1, elem_id=f"steps", label="Sampling steps", value=50)
                    eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="eta")

                with gr.Row():
                    lora_scale = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, label='Lora Scale', value=1.0, elem_id="lora_scale")
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=15.0, elem_id="cfg_scale")

                send_btn = gr.Button("Send")

            with gr.Column():
                output_video_1 = gr.PlayableVideo()

        with gr.Row():
            examples = [
                [
                    'an elephant is walking under the sea, 4K, high definition',
                    50,
                    'origin',
                    1,
                    15,
                    1,
                ],
                [
                    'an astronaut riding a horse in outer space',
                    25,
                    'origin',
                    1,
                    15,
                    1,
                ],
                [
                    'a monkey is playing a piano',
                    25,
                    'vangogh',
                    1,
                    15,
                    1,
                ],
                [
                    'A fire is burning on a candle',
                    25,
                    'frozen',
                    1,
                    15,
                    1,
                ],
                [
                    'a horse is drinking in the river',
                    25,
                    'yourname',
                    1,
                    15,
                    1,
                ],
                [
                    'Robot dancing in times square',
                    25,
                    'coco',
                    1,
                    15,
                    1,
                ],                    

            ]
            gr.Examples(examples=examples,
                        inputs=[
                        input_text,
                        steps,
                        model_index,
                        eta,
                        cfg_scale,
                        lora_scale],
                        outputs=[output_video_1],
                        fn=text2video.get_prompt,
                        cache_examples=False)
                        #cache_examples=os.getenv('SYSTEM') == 'spaces')

            send_btn.click(
                fn=text2video.get_prompt, 
                inputs=[
                    input_text,
                    steps,
                    model_index,
                    eta,
                    cfg_scale,
                    lora_scale,
                ],
                outputs=[output_video_1],
            )
    return videocrafter_iface

if __name__ == "__main__":
    result_dir = os.path.join('./', 'results')
    videocrafter_iface = videocrafter_demo(result_dir)
    videocrafter_iface.launch()