"""Steamship NLPCloud Plugin
"""

from typing import List, Optional, Type

from steamship import Block, Tag
from steamship.base.error import SteamshipError
from steamship.data.file import File
from steamship.invocable import Config, create_handler
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import \
    BlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest
from steamship.plugin.tagger import Tagger

from nlpcloud.api_spec import (NlpCloudModel, NlpCloudTask,
                               validate_task_and_model)
from nlpcloud.client import NlpCloudClient


class NlpCloudTaggerPluginConfig(Config):
    api_key: Optional[str]
    task: NlpCloudTask
    model: NlpCloudModel

    class Config:
        use_enum_values = False


class NlpCloudTaggerPlugin(Tagger):
    config: NlpCloudTaggerPluginConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We know from docs.nlpcloud.com that only certain task<>model pairings are valid.
        validate_task_and_model(self.config.task, self.config.model)

    def config_cls(self) -> Type[NlpCloudTaggerPluginConfig]:
        return NlpCloudTaggerPluginConfig

    def run(
        self, request: PluginRequest[BlockAndTagPluginInput]
    ) -> BlockAndTagPluginOutput:
        # TODO: Ensure base Tagger class checks to make sure this is not None
        file = request.data.file

        client = NlpCloudClient(key=self.config.api_key)
        if client is None:
            raise SteamshipError(message="Unable to create NlpCloudClient.")

        output = BlockAndTagPluginOutput(file=File.CreateRequest())

        for block in request.data.file.blocks:
            # Create tags for that block via OneAI and add them
            tags_lists: List[List[Tag.CreateRequest]] = client.request(
                model=self.config.model,
                task=self.config.task,
                inputs=[block.text],
            )

            tags = tags_lists[0] or []

            # Create an output block for this block
            output_block = Block.CreateRequest(id=block.id, tags=tags)

            # Attach the output block to the response
            output.file.blocks.append(output_block)

        return output


handler = create_handler(NlpCloudTaggerPlugin)
