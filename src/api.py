"""Steamship NLPCloud Plugin
"""

import logging
import pathlib
from typing import List, Optional, Type

import toml
from steamship import Block, Tag
from steamship.app import App, create_handler
from steamship.base.error import SteamshipError
from steamship.data.file import File
from steamship.plugin.blockifier import Config
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import \
    BlockAndTagPluginOutput
from steamship.plugin.service import PluginRequest
from steamship.plugin.tagger import Tagger

from nlpcloud.api_spec import (NlpCloudModel, NlpCloudTask,
                               validate_task_and_model)
from nlpcloud.client import NlpCloudClient


class NlpCloudTaggerPluginConfig(Config):
    api_key: Optional[str]
    task: NlpCloudTask
    model: NlpCloudModel


class NlpCloudTaggerPlugin(Tagger, App):
    def __init__(self, **kwargs):
        secret_kwargs = toml.load(
            str(pathlib.Path(__file__).parent / ".steamship" / "secrets.toml")
        )
        config = kwargs["config"] or {}
        kwargs["config"] = {
            **secret_kwargs,
            **{k: v for k, v in config.items() if v != ""},
        }
        super().__init__(**kwargs)
        logging.info(self.config)

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
