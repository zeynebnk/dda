import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from datasets import load_dataset
from trl import SFTTrainer


def load_model(model_path, lora_path=None):
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False
    )
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if lora_path:
        model.load_adapter(lora_path)

    
    return model, tokenizer

def train(model_path, dataset, name):

    model, tokenizer = load_model(model_path)
    
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    
    train_args = TrainingArguments(
        output_dir=name,
        num_train_epochs=3,
        optim="adamw_torch_fused",
        
        
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=train_args,
    )
    
    trainer.train()
    trainer.model.save_pretrained("./"+name)
