import pytest

from nlpcloud.api_spec import NlpCloudModel, NlpCloudTask

__copyright__ = "Steamship"
__license__ = "MIT"

from test.util import nlpcloud

from nlpcloud.client import NlpCloudClient


@pytest.mark.usefixtures("nlpcloud")
def test_live_sentiment(nlpcloud: NlpCloudClient):
    result = nlpcloud.request(
        NlpCloudTask.SENTIMENT,
        NlpCloudModel.DISTILBERT_BASE_UNCASED_EMOTION,
        ["I love parsnips!"],
    )
    print(result)


@pytest.mark.usefixtures("nlpcloud")
def test_live_entities(nlpcloud: NlpCloudClient):
    result = nlpcloud.request(
        NlpCloudTask.ENTITIES, NlpCloudModel.EN_CORE_WEB_LG, ["America is a country!"]
    )
    print(result)


@pytest.mark.usefixtures("nlpcloud")
def test_tokenize_chinese(nlpcloud: NlpCloudClient):
    texts = ["這個model會弄繁體嗎", "这个model会弄繁体吗"]
    result = nlpcloud.request(
        NlpCloudTask.TOKENS,
        NlpCloudModel.ZH_CORE_WEB_LG,
        texts,
    )
    for r, t in zip(result, texts):
        assert len(r) == 5 or len(r) == 6
        for tag in r:
            assert tag.value.get('text') in t
