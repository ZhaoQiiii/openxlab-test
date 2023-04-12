import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
import math

import torch
from decord import VideoReader, cpu
import torchvision
from pytorch_lightning import seed_everything

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import instantiate_from_config
from lvdm.utils.saving_utils import tensor_to_mp4
from scripts.sample_text2video_adapter import load_model_checkpoint, adapter_guided_synthesis

import torchvision.transforms._transforms_video as transforms_video
from huggingface_hub import hf_hub_download


def load_video(filepath, frame_stride, video_size=(256,256), video_frames=16):
    info_str = ''
    vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
    max_frames = len(vidreader)
    # auto 

    if frame_stride != 0:
        if frame_stride * (video_frames-1) >= max_frames:
            info_str += "Warning: The user-set frame rate makes the current video length not enough, we will set it to an adaptive frame rate.\n"
            frame_stride = 0
    if frame_stride == 0:
        frame_stride = max_frames / video_frames 
        # if temp_stride < 1:
            # info_str = "Warning: The length of the current input video is less than 16 frames, we will automatically fill to 16 frames for you.\n"
    if frame_stride > 100:
        frame_stride = 100
        info_str += "Warning: The current input video length is longer than 1600 frames, we will process only the first 1600 frames.\n"
    info_str += f"Frame Stride is set to {frame_stride}"
    frame_indices = [int(frame_stride*i) for i in range(video_frames)]
    frames = vidreader.get_batch(frame_indices)
        
    ## [t,h,w,c] -> [c,t,h,w]
    frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
    frame_tensor = (frame_tensor / 255. - 0.5) * 2    
    return frame_tensor, info_str

class VideoControl:
    def __init__(self, result_dir='./tmp/') -> None:
        self.savedir = result_dir
        self.download_model()
        config_path = "models/adapter_t2v_depth/model_config.yaml"
        ckpt_path = "models/base_t2v/model.ckpt"
        adapter_ckpt = "models/adapter_t2v_depth/adapter.pth"

        config = OmegaConf.load(config_path)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.to('cuda')
        assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, ckpt_path, adapter_ckpt)
        model.eval()
        self.model = model
        self.resolution=256
        self.spatial_transform = transforms_video.CenterCropVideo(self.resolution)

    def get_video(self, input_video, input_prompt, frame_stride=0, vc_steps=50, vc_cfg_scale=15.0, vc_eta=1.0):
        ## load video
        print("input video", input_video)
        info_str = ''
        try:
            h, w, c = VideoReader(input_video, ctx=cpu(0))[0].shape
        except:
            os.remove(input_video)
            return 'please input video', None, None, None

        if h < w:
            scale = h / self.resolution
        else:
            scale = w / self.resolution
        h = math.ceil(h / scale)
        w = math.ceil(w / scale)
        try:
            video, info_str = load_video(input_video, frame_stride, video_size=(h, w), video_frames=16)
        except:
            os.remove(input_video)
            return 'load video error', None, None, None
        video = self.spatial_transform(video)
        print('video shape', video.shape)

        h, w = 32, 32
        bs = 1
        channels = self.model.channels
        frames = self.model.temporal_length
        noise_shape = [bs, channels, frames, h, w]
        
        ## inference
        start = time.time()
        prompt = input_prompt
        video = video.unsqueeze(0).to("cuda")
        with torch.no_grad():
            batch_samples, batch_conds = adapter_guided_synthesis(self.model, prompt, video, noise_shape, n_samples=1, ddim_steps=vc_steps, ddim_eta=vc_eta, unconditional_guidance_scale=vc_cfg_scale)
        batch_samples = batch_samples[0]
        os.makedirs(self.savedir, exist_ok=True)
        filename = prompt
        filename = filename.replace("/", "_slash_") if "/" in filename else filename
        filename = filename.replace(" ", "_") if " " in filename else filename
        if len(filename) > 200:
            filename = filename[:200]
        video_path = os.path.join(self.savedir, f'{filename}_sample.mp4')
        depth_path = os.path.join(self.savedir, f'{filename}_depth.mp4')
        origin_path = os.path.join(self.savedir, f'{filename}.mp4')
        tensor_to_mp4(video=video.detach().cpu(), savepath=origin_path, fps=8)
        tensor_to_mp4(video=batch_conds.detach().cpu(), savepath=depth_path, fps=8)
        tensor_to_mp4(video=batch_samples.detach().cpu(), savepath=video_path, fps=8)

        print(f"Saved in {video_path}. Time used: {(time.time() - start):.2f} seconds")
        # delete video
        (path, input_filename) = os.path.split(input_video)
        if input_filename != 'flamingo.mp4':
            os.remove(input_video)
            print('delete input video')
        # print(input_video)
        return info_str, origin_path, depth_path, video_path
    def download_model(self):
        REPO_ID = 'VideoCrafter/t2v-version-1-1'
        filename_list = ['models/base_t2v/model.ckpt',
                         "models/adapter_t2v_depth/adapter.pth",
                         "models/adapter_t2v_depth/dpt_hybrid-midas.pt"
                        ]
        for filename in filename_list:
            if not os.path.exists(filename):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./', local_dir_use_symlinks=False)




    

if __name__ == "__main__":
    vc = VideoControl('./result')
    info_str, video_path =  vc.get_video('input/flamingo.mp4',"An ostrich walking in the desert, photorealistic, 4k")