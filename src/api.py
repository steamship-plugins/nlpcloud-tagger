"""Steamship NLPCloud Plugin
"""

from typing import List, Optional, Type

from steamship import Tag
from steamship.base.error import SteamshipError
from steamship.invocable import Config, Invocable, create_handler
from steamship.plugin.request import PluginRequest

from nlpcloud.api_spec import (NlpCloudModel, NlpCloudTask,
                               validate_task_and_model)
from nlpcloud.client import NlpCloudClient
from tagger.span import Granularity, Span
from tagger.span_tagger import SpanStreamingConfig, SpanTagger


class NlpCloudTaggerPluginConfig(Config):
    api_key: Optional[str]
    task: NlpCloudTask
    model: NlpCloudModel

    class Config:
        use_enum_values = False



class NlpCloudTaggerPlugin(SpanTagger, Invocable):
    config: NlpCloudTaggerPluginConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We know from docs.nlpcloud.com that only certain task<>model pairings are valid.
        validate_task_and_model(self.config.task, self.config.model)

    def config_cls(self) -> Type[NlpCloudTaggerPluginConfig]:
        return NlpCloudTaggerPluginConfig

    def get_span_streaming_args(self) -> SpanStreamingConfig:
        return SpanStreamingConfig(
            granularity=Granularity.BLOCK_TEXT
        )

    def tag_spans(self, request: PluginRequest[List[Span]]) -> List[Tag.CreateRequest]:
        client = NlpCloudClient(key=self.config.api_key)
        if client is None:
            raise SteamshipError(message="Unable to create NlpCloudClient.")

        all_tags = []
        for span in request.data:
            # Create tags for that block via OneAI and add them
            tags_lists: List[List[Tag.CreateRequest]] = client.request(
                model=self.config.model,
                task=self.config.task,
                inputs=[span.text],
            )

            tags = tags_lists[0] or []

            for tag in tags:
                tag.file_id = span.file_id
                if span.granularity != Granularity.FILE:
                    tag.block_id = span.block_id
                if span.granularity == Granularity.BLOCK_TEXT or span.granularity == Granularity.TAG:
                    tag.start_idx = span.start_idx
                    tag.end_idx = span.end_idx
                else:
                    tag.start_idx = None
                    tag.end_idx = None

                all_tags.append(tag)
        return all_tags


handler = create_handler(NlpCloudTaggerPlugin)
