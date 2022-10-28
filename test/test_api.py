import os
from typing import List

import pytest
from steamship import Block
from steamship.data.file import File
from steamship.data.tags import DocTag, Tag, TagKind, TagValue
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.request import PluginRequest

from api import NlpCloudTaggerPlugin, NlpCloudTaggerPluginConfig
from tagger.span import Granularity

__copyright__ = "Steamship"
__license__ = "MIT"

from nlpcloud.api_spec import NlpCloudModel, NlpCloudTask


def _read_test_file_lines(filename: str) -> List[str]:
    folder = os.path.dirname(os.path.abspath(__file__))
    lines = []
    with open(os.path.join(folder, "..", "test_data", "inputs", filename), "r") as f:
        lines = list(map(lambda line: line, f.read().split("\n")))
    return lines


def _read_test_file(filename: str) -> File:
    lines = _read_test_file_lines(filename)
    blocks = list(map(lambda t: Block(id=t[0], text=t[1]), enumerate(lines)))
    return File(id="XYZ", blocks=blocks)


def _file_from_string(string: str) -> File:
    lines = string.split("\n")
    blocks = list(map(lambda t: Block(id=t[0], text=t[1]), enumerate(lines)))
    return File(id="XYZ", blocks=blocks)


@pytest.fixture
def parser():
    parser = NlpCloudTaggerPlugin(
        config={
            "task": NlpCloudTask.TOKENS.value,
            "model": NlpCloudModel.EN_CORE_WEB_LG.value,
        }
    )
    return parser

def test_serialization():
    config = {
        "task": NlpCloudTask.LANGUAGE_DETECTION.value,
        "model": NlpCloudModel.PYTHON_LANGDETECT.value,
    }
    obj = NlpCloudTaggerPluginConfig(**config)
    assert obj.task == NlpCloudTask.LANGUAGE_DETECTION
    assert obj.model == NlpCloudModel.PYTHON_LANGDETECT
    assert obj.task.value == NlpCloudTask.LANGUAGE_DETECTION.value
    assert obj.model.value == NlpCloudModel.PYTHON_LANGDETECT.value


def test_embed_english_sentence():
    FILE = "roses.txt"

    embedder_block_text = NlpCloudTaggerPlugin(
        config={
            "task": NlpCloudTask.EMBEDDINGS.value,
            "model": NlpCloudModel.PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2.value,
        }
    )

    file = _read_test_file(FILE)

    NUM_BLOCKS = 5  # Includes the empty lines

    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = embedder_block_text.run(request)

    for block in response.file.blocks:
        for tag in block.tags:
            assert tag.kind == TagKind.EMBEDDING
            assert tag.value.get(TagValue.VECTOR_VALUE) is not None
            assert len(tag.value.get(TagValue.VECTOR_VALUE)) == 768

    embedder_tokens_text = NlpCloudTaggerPlugin(
        config={
            "task": NlpCloudTask.EMBEDDINGS.value,
            "model": NlpCloudModel.PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2.value,
            "granularity": Granularity.TAG,
            "kind_filter": TagKind.DOCUMENT,
            "name_filter": DocTag.TOKEN,
        }
    )

    # Add the tokens.
    for block in file.blocks:
        start_idx = 0
        tokens = block.text.split(" ")
        for token in tokens:
            block.tags.append(Tag(
                file_id=file.id,
                block_id=block.id,
                kind=TagKind.DOCUMENT,
                name=TagKind.TOKEN,
                start_idx=start_idx,
                end_idx=start_idx+len(token)
            ))
            start_idx += len(token)

    request2 = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response2 = embedder_tokens_text.run(request2)

    for (block_in, block_out) in zip(file.blocks, response2.file.blocks):
        tags_in, tags_out = block_in.tags, block_out.tags
        assert len(tags_out) == len(tags_in)
        for tag_1, tag_2 in zip(tags_in, tags_out):
            assert tag_1.kind == TagKind.DOCUMENT
            assert tag_2.kind == TagKind.EMBEDDING
            assert tag_1.start_idx == tag_2.start_idx
            assert tag_1.end_idx == tag_2.end_idx

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
        if tag.kind == DocTag.DOCUMENT and tag.name == DocTag.TOKEN:
            found_tokens += 1

    assert found_tokens == num_tokens


@pytest.mark.parametrize("granularity", [
    Granularity.FILE,
    Granularity.BLOCK_TEXT,
    Granularity.BLOCK
])
@pytest.mark.parametrize(
    "file,language",
    [
        (_file_from_string("Hi there!"), "en"),
        (_file_from_string("你好！你叫什麼名字？"), "zh-cn"),  # Yikes!
        (
            _file_from_string("你好！你叫什 x么名字？"),
            "zh-cn",
        ),  # It looks like it doesn't distinguish!
        (_file_from_string("こんにちは"), "ja"),
        (_file_from_string("नमस्ते"), "hi"),
    ],
)
def test_detect_language(granularity, file, language):
    language_detector = NlpCloudTaggerPlugin(
        config={
            "task": NlpCloudTask.LANGUAGE_DETECTION.value,
            "model": NlpCloudModel.PYTHON_LANGDETECT.value,
            "granularity": granularity,
        }
    )

    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = language_detector.run(request)

    if granularity == Granularity.BLOCK_TEXT or granularity == Granularity.BLOCK:
        assert not response.file.tags
        assert len(response.file.tags) == 0
        assert len(response.file.blocks) == 1
        assert len(response.file.blocks[0].tags) > 0
        assert response.file.blocks[0].tags[0].kind == "language"
        assert response.file.blocks[0].tags[0].name == language
        assert response.file.blocks[0].tags[0].value.get("score") is not None
        if granularity == Granularity.BLOCK_TEXT:
            assert response.file.blocks[0].tags[0].start_idx == 0
            assert response.file.blocks[0].tags[0].end_idx == len(file.blocks[0].text)
        else:
            assert response.file.blocks[0].tags[0].start_idx is None
            assert response.file.blocks[0].tags[0].end_idx is None
        assert response.file.blocks[0].tags[0].block_id == file.blocks[0].id
        assert response.file.blocks[0].tags[0].file_id == file.id
    elif granularity == Granularity.FILE:
        assert response.file.tags
        assert len(response.file.tags) > 0
        assert len(response.file.blocks) == 1
        assert len(response.file.blocks[0].tags) == 0
        assert response.file.tags[0].kind == "language"
        assert response.file.tags[0].name == language
        assert response.file.tags[0].value.get("score") is not None
        assert response.file.tags[0].start_idx == None
        assert response.file.tags[0].end_idx == None
        assert response.file.tags[0].block_id == None
        assert response.file.tags[0].file_id == file.id


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
        if tag.kind == DocTag.DOCUMENT and tag.name == DocTag.TOKEN:
            found_tokens += 1

    assert found_tokens == num_tokens
