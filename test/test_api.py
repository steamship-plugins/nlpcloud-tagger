import os
from typing import List

import pytest
from pydantic import ValidationError
from steamship import Block, DocTag, Steamship, SteamshipError
from steamship.data.file import File
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.service import PluginRequest

from api import NlpCloudTaggerPlugin, NlpCloudTaggerPluginConfig

__copyright__ = "Steamship"
__license__ = "MIT"

from nlpcloud.api_spec import NlpCloudModel, NlpCloudTask


def _read_test_file_lines(filename: str) -> File:
    folder = os.path.dirname(os.path.abspath(__file__))
    lines = []
    with open(os.path.join(folder, '..', 'test_data', 'inputs', filename), 'r') as f:
        lines = list(map(lambda line: line, f.read().split('\n')))
    return lines


def _read_test_file(filename: str) -> File:
    lines = _read_test_file_lines(filename)
    blocks = list(map(lambda line: Block(text=line), lines))
    return File(blocks=blocks)


def test_tagger():
    """Test an end-to-end run on the general structure of the full request-response"""
    tagger = NlpCloudTaggerPlugin(config={
        "task": NlpCloudTask.TOKENS.value,
        "model": NlpCloudModel.EN_CORE_WEB_LG.value
    })

    file = _read_test_file('weird_languages_pg.txt')
    lines = _read_test_file_lines('weird_languages_pg.txt')

    assert len(file.blocks) == 11

    request = PluginRequest(data=BlockAndTagPluginInput(
        file=file
    ))
    response = tagger.run(request)

    assert (response.file.blocks is not None)
    assert (len(response.file.blocks) == 11)

    # A Poem
    para1 = response.file.blocks[0]
    line1 = lines[0]

    print(para1)


# def test_plugin_fails_with_bad_config():
#     # With no client, no config
#     with pytest.raises(TypeError):
#         # TODO: How do we ignore the PyCharm "parameter unfilled" error below?
#         OneAITaggerPlugin()
#
#     # With no config
#     with pytest.raises(TypeError):
#         # TODO: How do we ignore the PyCharm "parameter unfilled" error below?
#         OneAITaggerPlugin(client=Steamship())
#
#     # With None config
#     with pytest.raises(SteamshipError):
#         # TODO: How do we ignore the PyCharm "parameter not what was expected" error below?
#         OneAITaggerPlugin(client=Steamship(), config=None)
#
#     # With empty dict config s
#     with pytest.raises(SteamshipError):
#         OneAITaggerPlugin(client=Steamship(), config=dict())
#
#     valid_config = {
#         "api_key": "foo",
#         "input_type": "conversation",
#         "skills": "foo"
#     }
#
#     # Missing a required config field
#     for key in valid_config.keys():
#         bad_config = valid_config.copy()
#         del bad_config[key]
#         with pytest.raises(ValidationError):
#             OneAITaggerPlugin(client=Steamship(), config=bad_config)
#
#     # Using a bad input_type
#     bad_config = valid_config.copy()
#     bad_config["input_type"] = "recording_of_dolphin_clicks"
#     with pytest.raises(ValidationError):
#         OneAITaggerPlugin(client=Steamship(), config=bad_config)
