import os

import pytest
from steamship import Block
from steamship.data.file import File
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.service import PluginRequest

from api import NlpCloudTaggerPlugin

__copyright__ = "Steamship"
__license__ = "MIT"

from nlpcloud.api_spec import NlpCloudModel, NlpCloudTask


def _read_test_file_lines(filename: str) -> File:
    folder = os.path.dirname(os.path.abspath(__file__))
    lines = []
    with open(os.path.join(folder, "..", "test_data", "inputs", filename), "r") as f:
        lines = list(map(lambda line: line, f.read().split("\n")))
    return lines


def _read_test_file(filename: str) -> File:
    lines = _read_test_file_lines(filename)
    blocks = list(map(lambda line: Block(text=line), lines))
    return File(blocks=blocks)


def _file_from_string(string: str) -> File:
    lines = string.split("\n")
    blocks = list(map(lambda line: Block(text=line), lines))
    return File(blocks=blocks)


@pytest.fixture
def parser():
    parser = NlpCloudTaggerPlugin(
        config={
            "task": NlpCloudTask.TOKENS.value,
            "model": NlpCloudModel.EN_CORE_WEB_LG.value,
        }
    )
    return parser


@pytest.fixture
def language_detector():
    parser = NlpCloudTaggerPlugin(
        config={
            "task": NlpCloudTask.LANGUAGE_DETECTION.value,
            "model": NlpCloudModel.PYTHON_LANGDETECT.value,
        }
    )
    return parser

@pytest.fixture
def embedder():
    embedder = NlpCloudTaggerPlugin(
        config={
            "task": NlpCloudTask.EMBEDDINGS.value,
            "model": NlpCloudModel.PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2.value,
        }
    )
    return embedder


def test_embed_english_sentence(embedder):
    sentence = "Hello there"
    FILE = "roses.txt"

    file = _read_test_file(FILE)

    NUM_BLOCKS = 5  # Includes the empty lines

    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = embedder.run(request)

    for block in response.file.blocks:
        for tag in block.tags:
            assert tag.kind == 'embedding'
            assert tag.value.get('value') is not None
            assert len(tag.value.get('value')) == 768


def test_parse_english_sentence(parser):
    """Test an end-to-end run on the general structure of the full request-response"""
    FILE = "roses.txt"

    file = _read_test_file(FILE)

    NUM_BLOCKS = 5  # Includes the empty lines

    assert len(file.blocks) == NUM_BLOCKS

    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = parser.run(request)

    assert response.file.blocks is not None
    assert len(response.file.blocks) == NUM_BLOCKS

    line1 = response.file.blocks[0]

    num_tokens = len(file.blocks[0].text.split(" ")) + 1

    print(f"{num_tokens} {line1.text}")

    tags = line1.tags

    found_tokens = 0
    for tag in tags:
        if tag.kind == "doc" and tag.name == "token":
            found_tokens += 1

    assert found_tokens == num_tokens


@pytest.mark.parametrize(
    "file,language",
    [
        (_file_from_string("Hi there!"), "en"),
        (_file_from_string("你好！你叫什麼名字？"), "zh-cn"),  # Yikes!
        (
            _file_from_string("你好！你叫什么名字？"),
            "zh-cn",
        ),  # It looks like it doesn't distinguish!
        (_file_from_string("こんにちは"), "ja"),
        (_file_from_string("नमस्ते"), "hi"),
    ],
)
def test_detect_language(language_detector, file, language):
    # def test_detect_language(language_detector):
    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = language_detector.run(request)
    assert len(response.file.blocks) == 1
    assert len(response.file.blocks[0].tags) == 1
    assert response.file.blocks[0].tags[0].kind == "language"
    assert response.file.blocks[0].tags[0].name == language
    assert response.file.blocks[0].tags[0].value.get("score") is not None


@pytest.mark.parametrize(
    "model,file,language,num_blocks,num_tokens",
    [
        (NlpCloudModel.EN_CORE_WEB_LG.value, _file_from_string("Hi there"), "en", 1, 2),
        (
            NlpCloudModel.ZH_CORE_WEB_LG.value,
            _file_from_string("你好！你叫什麼名字？"),
            "zh-cn",
            1,
            7,
        ),
        (NlpCloudModel.JA_CORE_NEWS_LG.value, _file_from_string("こんにちは"), "ja", 1, 1),
    ],
)
def test_tokenize_some(model, file, language, num_blocks, num_tokens):
    parser = NlpCloudTaggerPlugin(
        config={"task": NlpCloudTask.TOKENS.value, "model": model}
    )

    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = parser.run(request)

    assert response.file.blocks is not None
    assert len(response.file.blocks) == num_blocks

    line1 = response.file.blocks[0]
    tags = line1.tags

    found_tokens = 0
    for tag in tags:
        if tag.kind == "doc" and tag.name == "token":
            found_tokens += 1

    assert found_tokens == num_tokens
