"""Steamship NLPCloud Plugin
"""

from typing import Any, Dict, Optional, Type

from steamship import Block, Steamship
from steamship.app import App, create_handler
from steamship.base.error import SteamshipError
from steamship.data.file import File
from steamship.plugin.blockifier import Config
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import BlockAndTagPluginOutput
from steamship.plugin.service import PluginRequest
from steamship.plugin.tagger import Tagger

from src.nlpcloud import (
    NlpCloudClient,
    NlpCloudModel,
    NlpCloudTask,
    validate_task_and_model,
)


class NlpCloudTaggerPluginConfig(Config):
    api_key: Optional[str]
    task: NlpCloudTask
    model: NlpCloudModel


class NlpCloudTaggerPlugin(Tagger, App):
    def __init__(self, client: Steamship, config: Dict[str, Any]):
        super().__init__(client, config)

        # This plugin requires configuration
        if self.config is None:
            raise SteamshipError(
                message="Missing Plugin Instance configuration dictionary."
            )

        # The api_key must not be none
        if self.config.api_key is None:
            raise SteamshipError(
                message="Missing `api_key` field in Plugin configuration dictionary."
            )

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
            # Create an output block for this block
            output_block = Block.CreateRequest(id=block.id)

            # Create tags for that block via OneAI and add them
            request = NlpCloudRequest(
                model=self.config.model,
                task=self.config.task,
                text=block.text,
            )
            response = client.request(request)
            if response:
                output_block.tags = response.to_tags()

            # Attach the output block to the response
            output.file.blocks.append(output_block)

        return output


handler = create_handler(NlpCloudTaggerPlugin)