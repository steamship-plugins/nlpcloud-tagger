from abc import ABC, abstractmethod
from typing import List, Optional

from steamship import File, SteamshipError
from steamship.base.model import CamelModel
from steamship.invocable import InvocableResponse, post
from steamship.invocable.plugin_service import PluginService
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import \
    BlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest

from build.deps.steamship import Block
from tagger.span import Granularity, Span


class SpanStreamingConfig(CamelModel):
    granularity: Granularity
    kind_filter: Optional[str]
    name_filter: Optional[str]

class Tagger(PluginService[BlockAndTagPluginInput, BlockAndTagPluginOutput], ABC):
    """An implementation of a Tagger that permits implementors to care only about Spans."""

    def run(
        self, request: PluginRequest[BlockAndTagPluginInput]
    ) -> InvocableResponse[BlockAndTagPluginOutput]:
        args = self.get_span_streaming_args()
        spans = list(
            Span.stream_from(
                file=request.data.file,
                granularity=args.granularity,
                kind_filter=args.kind_filter,
                name_filter=args.name_filter
            )
        )
        output_tags = self.tag_spans(
            PluginRequest(
                data=spans,
                context=request.context,
                status=request.status,
                is_status_check=request.is_status_check
            )
        )

        # Now prepare the results. There's a bit of bookkeeping we have to do to make sure this is
        # structured properly with respect to the current BlockAndTag contract.
        block_lookup = {}
        output = BlockAndTagPluginOutput(file=File.CreateRequest(), tags=[])
        for block in request.data.file.blocks:
            output_block = Block.CreateRequest(id=block.id, tags=[])
            block_lookup[block.id] = output_block
            output.file.blocks.append(output_block)

        # Go through each span and add to the appropriate place.
        for tag in output_tags:
            if tag.file_id is None:
                raise SteamshipError(message="All Tags should have a file_id field")
            if args.granularity == Granularity.FILE:
                if tag.block_id is not None:
                    raise SteamshipError(message="A tag with a granularity of FILE should not have a block_id field")
                output.file.tags.append(tag)
            elif args.granularity == Granularity.TAG or args.granularity == Granularity.BLOCK:
                if args.granularity.BLOCK:
                    if tag.start_idx is not None:
                        raise SteamshipError(message="A Tag with a granularity of BLOCK should not have a start_idx field")
                    if tag.end_idx is not None:
                        raise SteamshipError(message="A Tag with a granularity of BLOCK should not have a end_idx field")
                    if tag.block_id is None:
                        raise SteamshipError(message="Error: a Tag with a granularity of BLOCK or TAG should have a block_id field")
                    if tag.block_id not in block_lookup:
                        raise SteamshipError(message=f"Error: the referenced block_id {tag.block_id} was not among the input Blocks")
                    # Phew. Ok.
                    block_lookup[tag.block_id].tags.append(tag)

        # Finally, we can return the output
        return output


    @abstractmethod
    def get_span_streaming_args(self) -> SpanStreamingConfig:
        """This is a kludge to let the implementor return the required information for extracting Spans from the
        BlockAndTagPluginInput. Right now these have to be provided via the Config block on the plugin."""
        raise NotImplementedError()

    @abstractmethod
    def tag_spans(self, request: PluginRequest[List[Span]]) -> List[Tag.CreateRequest]:
        """The plugin author now just has to implement tagging over the provided spans."""
        raise NotImplementedError()

    @post("tag")
    def run_endpoint(self, **kwargs) -> InvocableResponse[BlockAndTagPluginOutput]:
        """Exposes the Tagger's `run` operation to the Steamship Engine via the expected HTTP path POST /tag"""
        return self.run(PluginRequest[BlockAndTagPluginInput].parse_obj(kwargs))
