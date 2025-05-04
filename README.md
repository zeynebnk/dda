# Model Evaluation Pipeline

This repository contains tools and scripts for evaluating language models using the lm-evaluation-harness framework.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install lm-evaluation-harness:
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

## Usage

### Running Evaluations

To evaluate a model on CPU:
```bash
python main.py \
    --model hf \
    --model_args pretrained=facebook/opt-125m \
    --tasks tinyMMLU \
    --device cpu \
    --batch_size 1 \
    --model_args_kwargs '{"torch_dtype": "float32", "max_length": 2048}'
```

To evaluate a model on GPU:
```bash
python main.py \
    --model hf \
    --model_args pretrained=facebook/opt-125m \
    --tasks tinyMMLU \
    --device cuda \
    --batch_size 4
```

## Project Structure

- `gen_eval/`: Contains scripts for model generation and evaluation
- `lm-evaluation-harness/`: The evaluation framework
- `diversity/`: Tools for measuring model diversity
- `requirements.txt`: Project dependencies

## License

MIT License 