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

    granularity: Granularity = Granularity.BLOCK
    kind_filter: Optional[str] = None
    name_filter: Optional[str] = None

    class Config:
        use_enum_values = False


class NlpCloudTaggerPlugin(SpanTagger, Invocable):
    config: NlpCloudTaggerPluginConfig
    client: NlpCloudClient

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We know from docs.nlpcloud.com that only certain task<>model pairings are valid.
        validate_task_and_model(self.config.task, self.config.model)

        self.client = NlpCloudClient(key=self.config.api_key)
        if self.client is None:
            raise SteamshipError(message="Unable to create NlpCloudClient.")

    def config_cls(self) -> Type[NlpCloudTaggerPluginConfig]:
        return NlpCloudTaggerPluginConfig

    def get_span_streaming_args(self) -> SpanStreamingConfig:
        return SpanStreamingConfig(
            granularity=self.config.granularity,
            kind_filter=self.config.kind_filter,
            name_filter=self.config.name_filter
        )

    def tag_span(self, request: PluginRequest[Span]) -> List[Tag.CreateRequest]:
        tags_lists: List[List[Tag.CreateRequest]] = self.client.request(
            model=self.config.model,
            task=self.config.task,
            inputs=[request.data.text],
        )
        tags = tags_lists[0] or []
        return tags


handler = create_handler(NlpCloudTaggerPlugin)
