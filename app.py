import os
import sys
import gradio as gr
from videocrafter_test import Text2Video
from videocontrol_test import VideoControl
sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))

t2v_examples = [
    ['an elephant is walking under the sea, 4K, high definition',50,'origin',1,15,1,],
    ['an astronaut riding a horse in outer space',25,'origin',1,15,1,],
    ['a monkey is playing a piano',25,'vangogh',1,15,1,],
    ['A fire is burning on a candle',25,'frozen',1,15,1,],
    ['a horse is drinking in the river',25,'yourname',1,15,1,],
    ['Robot dancing in times square',25,'coco',1,15,1,],                    
]

control_examples = [
    ['input/flamingo.mp4', 'An ostrich walking in the desert, photorealistic, 4k', 0, 50, 15, 1]
]

def videocrafter_demo(result_dir='./tmp/'):
    text2video = Text2Video(result_dir)
    videocontrol = VideoControl(result_dir)
    with gr.Blocks(analytics_enabled=False) as videocrafter_iface:
        gr.Markdown("<div align='center'> <h2> VideoCrafter: A Toolkit for Text-to-Video Generation and Editing </span> </h2> \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/VideoCrafter/VideoCrafter'> Github </div>")
        #######t2v#######
        with gr.Tab(label="Text2Video"):
            with gr.Column():
                with gr.Row().style(equal_height=False):
                    with gr.Column():
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
                    with gr.Tab(label='show'):
                        output_video_1 =  gr.Video().style(width=384)
                gr.Examples(examples=t2v_examples,
                            inputs=[input_text,steps,model_index,eta,cfg_scale,lora_scale],
                            outputs=[output_video_1],
                            fn=text2video.get_prompt,
                            cache_examples=False)
                        #cache_examples=os.getenv('SYSTEM') == 'spaces')
            send_btn.click(
                fn=text2video.get_prompt, 
                inputs=[input_text,steps,model_index,eta,cfg_scale,lora_scale,],
                outputs=[output_video_1],
            )
        #######videocontrol######
        with gr.Tab(label='VideoControl'):
            with gr.Column():
                with gr.Row():
                    # with gr.Tab(label='input'):
                    with gr.Column():
                        vc_input_video = gr.Video().style(width=256)
                        with gr.Row():
                            vc_input_text = gr.Text(label='Prompts')
                        with gr.Row():
                            vc_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="vc_eta")
                            vc_cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=15.0, elem_id="vc_cfg_scale")
                        with gr.Row():
                            vc_steps = gr.Slider(minimum=1, maximum=200, step=1, elem_id="vc_steps", label="Sampling steps", value=50)
                            frame_stride = gr.Slider(minimum=0 , maximum=8, step=1, label='Frame Stride', value=0, elem_id="vc_frame_stride")

                        vc_end_btn = gr.Button("Send")
                    with gr.Tab(label='Result'):
                        vc_output_info = gr.Text(label='Info')
                        vc_output_video = gr.Video().style(width=384)

                gr.Examples(examples=control_examples,
                            inputs=[vc_input_video, vc_input_text, frame_stride, vc_steps, vc_cfg_scale, vc_eta],
                            outputs=[vc_output_info, vc_output_video],
                            fn = videocontrol.get_video,
                            cache_examples=False
                )
            vc_end_btn.click(inputs=[vc_input_video, vc_input_text, frame_stride, vc_steps, vc_cfg_scale, vc_eta],
                            outputs=[vc_output_info, vc_output_video],
                            fn = videocontrol.get_video
            )

    return videocrafter_iface

if __name__ == "__main__":
    result_dir = os.path.join('./', 'results')
    videocrafter_iface = videocrafter_demo(result_dir)
    videocrafter_iface.launch()
    # videocrafter_iface.launch(server_name='0.0.0.0', server_port=80)