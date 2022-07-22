import os
import pytest
from src.nlpcloud import *

def read_test_file(filename: str):
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', filename), 'r') as f:
        return f.read()

@pytest.fixture()
def nlpcloud() -> NlpCloudClient:
    """Returns an NLpCloudClient.

    To use, simply import this file and then write a test which takes `nlpcloud`
    as an argument.

    Example
    -------
    The client can be used by injecting a fixture as follows::

        @pytest.mark.usefixtures("nlpcloud")
        def test_something(nlpcloud):
          pass
    """
    environ_key = os.environ.get('NLPCLOUD_KEY')
    if environ_key is not None:
        return NlpCloudClient(key=environ_key)
    raise SteamshipError(message="No NLP Cloud key found. Please set the NLPCLOUD_KEY environment variable.")

