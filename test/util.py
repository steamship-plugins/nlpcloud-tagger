import os
import logging

from src.oneai import *


def get_oneai_client(key: str = None):
    if key is not None:
        return OneAIClient(key=key)
    environ_key = os.environ.get('ONEAI_KEY')
    if environ_key is not None:
        return OneAIClient(key=environ_key)
    logging.error("No OneAI key found. Please set ONEAI_KEY")
    return None


def read_test_file(filename: str):
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', filename), 'r') as f:
        return f.read()
