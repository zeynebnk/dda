import run_model
import lm_eval

from datasets import load_dataset
from huggingface_hub import login


def prep_data(config):

    if config['data_src'] == 'genQA':
        ds = load_dataset("tomg-group-umd/GenQA", split=config['data_type'] + f"[:{config['size']}]")
        
        def format_text(example):
            # Get the user question and assistant answer from the text list
            question = example['text'][0]['content']  # user's question
            answer = example['text'][1]['content']    # assistant's answer
            
            # Format for instruction tuning
            return {
                'text': f"<s>[INST] {question} [/INST] {answer} </s>"
            }
        
        # Apply the formatting
        ds = ds.map(format_text, remove_columns=ds.column_names)

    elif config['data_src'] == 'self_gen':
        ds = load_dataset("json", data_files=config['data_path'])

    return ds

def pipeline(config):
    
    if config['ft']:
        ds = prep_data(config)
        run_model.train(config['model_path'], ds, config["name"])

        lm_eval.evaluate_model(config['model_path'], "./"+config["name"], config['task'])

    else:
        lm_eval.evaluate_model(config['model_path'], None, config['task'])


baseline_mmlu_config = {
    'model_path': 'meta-llama/Meta-Llama-3-8B',
    'task': 'tinyMMLU',
    'ft': False,
    'name': 'baseline_mmlu'
} 

comparison_mmlu_config = {
    'model_path': 'meta-llama/Meta-Llama-3-8B',
    'task': 'tinyMMLU',
    'ft': True,
    'data_src': 'genQA',
    'data_type': 'mmlu',
    'size': 100,
    'name': 'comparison_mmlu'
} 

pipeline(comparison_mmlu_config)







