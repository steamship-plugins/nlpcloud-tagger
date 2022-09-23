import logging

import pytest

from src.nlpcloud import *

__copyright__ = "Steamship"
__license__ = "MIT"

from test.util import nlpcloud, read_test_file
from nlpcloud.client import NlpCloudClient

@pytest.mark.usefixtures("nlpcloud")
def test_live_sentiment(nlpcloud: NlpCloudClient):
    result = nlpcloud.request(
        NlpCloudTask.SENTIMENT,
        NlpCloudModel.DISTILBERT_BASE_UNCASED_EMOTION,
        ["I love parsnips!"]
    )
    print(result)


@pytest.mark.usefixtures("nlpcloud")
def test_live_entities(nlpcloud: NlpCloudClient):
    result = nlpcloud.request(
        NlpCloudTask.ENTITIES,
        NlpCloudModel.EN_CORE_WEB_LG,
        ["America is a country!"]
    )
    print(result)

@pytest.mark.usefixtures("nlpcloud")
def test_tokenize_chinese(nlpcloud: NlpCloudClient):
    result = nlpcloud.request(
        NlpCloudTask.TOKENS,
        NlpCloudModel.ZH_CORE_WEB_LG,
        ["這個model會弄繁體嗎", "这个model会弄繁体吗"]
    )
    for r in result:
        for tag in r:
            print(tag.value.get('text'))

