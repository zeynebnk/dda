import torch
import numpy as np

import subprocess


def evaluate_model(model_path, peft_path=None, task=None):

    subprocess.run("cd ./lm-evaluation-harness/", shell=True)

    if peft_path:
        run_command = f"lm_eval --model hf \
        --model_args pretrained={model_path},load_in_4bit=True,peft={peft_path} \
        --tasks {task} --device cpu"
    else:
        run_command = f"lm_eval --model hf \
        --model_args pretrained={model_path} \
        --tasks {task} --device cpu"

    subprocess.run(run_command, shell=True)

    subprocess.run("cd ..", shell=True)
    
