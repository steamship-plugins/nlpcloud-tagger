import os
from test import DOT_STEAMSHIP

import pytest
import toml
from steamship import SteamshipError

from nlpcloud.client import NlpCloudClient


def read_test_file(filename: str):
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "..", "test_data", filename), "r") as f:
        return f.read()


def get_key() -> str:
    try:
        secret_kwargs = toml.load(DOT_STEAMSHIP / "secrets.toml")
        return secret_kwargs.get("api_key")
    except Exception:
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
    raise SteamshipError(
        message="No NLP Cloud key found. Please set the api_key variable in git ignored src/resources/api_key.json."
    )
