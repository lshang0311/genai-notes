import os

import tiktoken
from dotenv import load_dotenv


def load_env(env: str):
    if env == 'TEST':
        env_path = os.getenv('ENVIRONMENT', '.env.test')
    else:
        raise ValueError(f"Invalid environment: {env}.")

    dotenv_path = f'{env_path}'
    load_dotenv(dotenv_path=dotenv_path)
    return True


def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = tokenizer.encode(text)
    return len(tokens)
