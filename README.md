# Shakespeare LLM #

This LLM was fine-tuned on a curated corpus of Shakespeare’s complete works, including poems, plays, and sonnets. It uses the gpt-2 model from huggingface transformers. The model learns to generate text and complete the user's passage using Shakespeare’s unique linguistic style. The trained model is uploaded to [Huggingface Hub](https://huggingface.co/amahuli/shakespeare-llm) and is accessed for inference via FastAPI from [my web UI project](https://github.com/anishamahuli/shakespeare-bot-project).

### Local Development ###

To clone repo and re-train model locally:
1. conda torch-nightly environment is recommended
2. Install dependencies in `environment.yml`. Run `check_installation.py` to confirm.
3. Run `prepare_data.py`
4. Run `train.py` (will take around an hour)
5. Use `generate_text.py` to interact with your trained model
