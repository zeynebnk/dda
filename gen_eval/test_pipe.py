import run_model
import lm_eval

from datasets import load_dataset

def prep_data(config):

    if config['data_src'] == 'genQA':
        ds = load_dataset("tomg-group-umd/GenQA", split=config['data_type'] + f"[:{config['size']}]")

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
    'model_path': 'facebook/opt-125m',
    'task': 'tinyMMLU',
    'ft': False,
    'name': 'baseline_mmlu'
} 

comparison_mmlu_config = {
    'model_path': 'facebook/opt-125m',
    'task': 'tinyMMLU',
    'ft': True,
    'data_src': 'genQA',
    'data_type': 'mmlu',
    'size': 100,
    'name': 'comparison_mmlu'
} 

pipeline(baseline_mmlu_config)







