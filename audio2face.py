import requests
import os
import io
import json
import logging
import time
import subprocess
import colorama

import gradio as gr

logging.basicConfig(level=logging.INFO)

def run_system_command(command):
    logging.info(colorama.Back.GREEN + f"Running command: {command}" + colorama.Style.RESET_ALL)
    process = subprocess.Popen(command, shell=True, env=os.environ, stdout=subprocess.PIPE)
    output, error = process.communicate()

    if process.returncode != 0:
        print(colorama.Fore.RED + f"Error executing command: {error}" + colorama.Style.RESET_ALL)
    else:
        print(f"Output: {output.decode('utf-8')}")


class Audio2Face:
    def __init__(self):
        pass

    def send_job(self, audio_file_path):
        msg = "成功"
        video_no_audio_path = None
        audio_filename = os.path.basename(audio_file_path)
        audio_dir = os.path.dirname(audio_file_path)
        audio_filename_no_ext = os.path.splitext(audio_filename)[0]
        # convert audio to npy, save to data/<name>.npy
        print(f"Converting audio to npy".center(80, "#"))
        run_system_command(f"python nerf/asr.py --wav {audio_file_path} --save_feats")
        # run system command
        print(f"Generating video".center(80, "#"))
        run_system_command(f"python main.py data/chuanhu/ --workspace trial_chuanhu_torso/ -O --torso --test --aud {os.path.join(audio_dir, f'{audio_filename_no_ext}_eo.npy')}")
        video_no_audio_path = f"trial_chuanhu_torso/results/ngp_ep0039.mp4"
        # merge audio and video
        print(f"Merging audio and video".center(80, "#"))
        run_system_command(f"ffmpeg -y -i {video_no_audio_path} -i {audio_file_path} -c:v copy -c:a aac -strict experimental trial_chuanhu_torso/results/{audio_filename_no_ext}_merged.mp4")
        video_with_audio_path =  f"trial_chuanhu_torso/results/{audio_filename_no_ext}_merged.mp4"
        return video_with_audio_path, msg


a2f = Audio2Face()

with gr.Blocks(theme="JohnSmith9982/small_and_pretty") as demo:
    gr.Markdown("""<h1 align="left">🌲 Santa Audio2Face</h1>""")
    with gr.Row():
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="filepath", value="/home/john/RAD-NeRF/data/me.png", interactive=False)
                audio_in = gr.Audio(source="microphone", type="filepath")
                with gr.Row():
                    submit_btn = gr.Button("提交")
                status_display = gr.Markdown("ready")
        with gr.Row():
            with gr.Column():
                video_out = gr.Video(interactive=False).style(height="200px")

    submit_btn.click(a2f.send_job, [audio_in], [video_out, status_display])

demo.queue().launch(inbrowser=True, share=True)
