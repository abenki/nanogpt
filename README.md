# NanoGPT

In this project, I trained a 10M parameters GPT-like model based on this great video from Andrew Karpathy: [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=960s). I used the tiny Shakespeare dataset. The final output is a model able to produce text that has the same structure as a typical Shakespeare text (but the English is far from being grammatically and syntactically correct). This project focuses only on the pretraining stage of LLM training.

## How to use
Start by creating a virtual environment and installing the dependencies:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Generate text using the last trained model:
```bash
python generate_text.py --checkpoint-path ./checkpoints/run_20241228_212938/model.pt --max-tokens 1000
```

```
--checkpoint-path:  Path to the pre-trained model file
--max-tokens:       Maximum number of tokens to generate
```

## Files overview
- ```config.py```: hyperparameters and configuration
- ```generate_text.py```: command-line interface for generating text using the pre-trained model
- ```get_metadata.py```: get some metadata about the model
- ```load.py```: load the model from a given .pt file
- ```model.py```: definition of the different components of the GPT model
- ```preprocessing.py```: loading and preprocessing of the data
- ```training.py```: training loop and model saving
- ```prototype.py```: first draft of the code, all the steps detailed above in a single file
