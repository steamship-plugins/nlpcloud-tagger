import os
from typing import List

import pytest
from pydantic import ValidationError
from steamship import Block, DocTag, Steamship, SteamshipError
from steamship.data.file import File
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.service import PluginRequest

from src.api import OneAITaggerPlugin

__copyright__ = "Steamship"
__license__ = "MIT"


def _read_test_file_lines(filename: str) -> List[str]:
    folder = os.path.dirname(os.path.abspath(__file__))
    blocks = []
    with open(os.path.join(folder, '..', 'test_data', filename), 'r') as f:
        blocks = list(map(lambda line: Block(text=line), f.read().split('\n')))
    return File(blocks=blocks)


def test_tagger():
    """Test an end-to-end run on the general structure of the full request-response"""
    tagger = OneAITaggerPlugin()
    rose_file = _read_test_file_lines('call_center.txt')
    request = PluginRequest(data=BlockAndTagPluginInput(
        file=rose_file
    ))
    response = tagger.run(request)

    assert (response.error is None)
    assert (response.data is not None)

    assert (response.data.blocks is not None)
    assert (len(response.data.blocks) == 3)

    # A Poem
    para1 = response.data.blocks[0]
    assert (para1.type == DocTag.doc)
    assert (para1.children is not None)
    for kit in para1.children:
        print(kit.text)
    assert (len(para1.children) == 1)
    assert (para1.children[0].text == "A Poem.")
    assert (para1.children[0].tokens is not None)
    assert (len(para1.children[0].tokens) == 2)
    assert (para1.children[0].type == DocTag.sentence)

    # Roses are red. Violets are blue.
    para2 = response.data.blocks[1]
    assert (para2.type == DocTag.doc)
    assert (para2.children is not None)
    assert (len(para2.children) == 2)
    assert (para2.children[0].text == "Roses are red.")
    assert (para2.children[1].text == "Violets are blue.")
    assert (para2.children[1].tokens[1].text == "are")

    # Sugar is sweet, and I love you.
    para3 = response.data.blocks[2]
    assert (para3.children is not None)
    assert (len(para3.children) == 1)
    assert (para3.children[0].text == "Sugar is sweet, and I love you.")
    assert (para3.children[0].tokens[2].text == "sweet,")

def test_plugin_fails_with_bad_config():
    # With no client, no config
    with pytest.raises(TypeError):
        # TODO: How do we ignore the PyCharm "parameter unfilled" error below?
        OneAITaggerPlugin()

    # With no config
    with pytest.raises(TypeError):
        # TODO: How do we ignore the PyCharm "parameter unfilled" error below?
        OneAITaggerPlugin(client=Steamship())

    # With None config
    with pytest.raises(SteamshipError):
        # TODO: How do we ignore the PyCharm "parameter not what was expected" error below?
        OneAITaggerPlugin(client=Steamship(), config=None)

    # With empty dict config
    with pytest.raises(SteamshipError):
        OneAITaggerPlugin(client=Steamship(), config=dict())

    valid_config = {
        "api_key": "foo",
        "input_type": "conversation",
        "skills": "foo"
    }

    # Missing a required config field
    for key in valid_config.keys():
        bad_config = valid_config.copy()
        del bad_config[key]
        with pytest.raises(ValidationError):
            OneAITaggerPlugin(client=Steamship(), config=bad_config)

    # Using a bad input_type
    bad_config = valid_config.copy()
    bad_config["input_type"] = "recording_of_dolphin_clicks"
    with pytest.raises(ValidationError):
        OneAITaggerPlugin(client=Steamship(), config=bad_config)
