import os
import pytest
from nlpcloud.client import *
from steamship import SteamshipError

def read_test_file(filename: str):
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', filename), 'r') as f:
        return f.read()


def get_key() -> str:
    try:
        with open(RESOURCES / "api_key.json") as f:
            j = json.load(f)
            return j.get("api_key")
    except:
        print("Unable to get api_key from src/resources/api_key.json")

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

    environ_key = get_key()
    if environ_key is not None:
        return NlpCloudClient(key=environ_key)
    raise SteamshipError(message="No NLP Cloud key found. Please set the api_key variable in git ignored src/resources/api_key.json.")

