import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb


def load_model(model_path, lora_path=None):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto"
    )
    
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def train(model_path, dataset, name):

    model, tokenizer = load_model(model_path)
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,  
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./"+name,
            num_train_epochs=3
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    model.save_pretrained("./"+name)

