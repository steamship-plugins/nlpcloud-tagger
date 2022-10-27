from enum import Enum
from typing import Generator, List, Optional

from steamship import Block, File, Tag
from steamship.base.model import CamelModel


class Granularity(str, Enum):
    FILE = "file"
    BLOCK = "block"
    TAG = "tag"

def _tag_matches(tag: Tag, kind_filter: str = None, name_filter: str = None) -> bool:
    """Returns whether the tag matches the provided filter."""
    return (
        (kind_filter is None or tag.kind == kind_filter) and
        (name_filter is None or tag.name == name_filter)
    )

def _tags_match(tags: Optional[List[Tag]], kind_filter: str = None, name_filter: str = None) -> bool:
    """Returns whether one of the tags matches the provided filter."""
    if tags is None:
        return False
    for tag in tags:
        if _tag_matches(tag, kind_filter=kind_filter, name_filter=name_filter):
            return True
    return False

def _file_matches(file: File, kind_filter: str = None, name_filter: str = None) -> bool:
    """Returns whether one of the file tags matches the provided filter."""
    if not kind_filter and not name_filter:
        return True
    return _tags_match(file.tags, kind_filter=kind_filter, name_filter=name_filter)

def _block_matches(block: Block, kind_filter: str = None, name_filter: str = None) -> bool:
    """Returns whether one of the block tags matches the provided filter."""
    if not kind_filter and not name_filter:
        return True
    return _tags_match(block.tags, kind_filter=kind_filter, name_filter=name_filter)


class TextInput(CamelModel):
    file_id: str
    block_id: Optional[str]
    granularity: Granularity
    text: str
    start_idx: Optional[int]
    end_idx: Optional[int]
    tag_kind: Optional[str]
    tag_name: Optional[str]


def stream_tags(
        file: File = None,
        granularity: Granularity = None,
        kind_filter: str = None,
        name_filter: str = None
) -> Generator[TextInput, None, None]:
    """Steams units of work to be provided as input to the tagger.

    Attributes
    ----------
    file : File
        The provided file, from the Steamship Engine, from which units of processing come.
    granularity : Granularity
        The desired granularity of the units of work: the entire file, blocks, or text covered by tags
    kind_filter : str
        Whether to filter the unit of granularity for those matching coverage by Tags of that kind
    name_filter : str
        Whether to filter the unit of granularity for those matching coverage by Tags of that name
    """
    if not file:
        return

    if granularity == Granularity.FILE:
        if _file_matches(file, kind_filter=kind_filter, name_filter=name_filter):
            all_text = "\n".join([block.text for block in file.blocks or [] if block.text])
            yield TextInput(
                file_id = file.id,
                block_id = None,
                granularity = Granularity.FILE,
                text = all_text,
                start_idx = None,
                end_idx = None,
                tag_kind = kind_filter,
                tag_name = name_filter
            )
    elif granularity == Granularity.BLOCK:
        if not file.blocks:
            return
        for block in file.blocks:
            if _block_matches(block, kind_filter=kind_filter, name_filter=name_filter):
                yield TextInput(
                    file_id=file.id,
                    block_id=None,
                    granularity=Granularity.BLOCK,
                    text=block.text,
                    start_idx=0,
                    end_idx=len(block.text),
                    tag_kind=kind_filter,
                    tag_name=name_filter
                )
    elif granularity == Granularity.TAG:
        if not file.blocks:
            return
        for block in file.blocks:
            if not block.tags:
                continue
            for tag in block.tags:
                if _tag_matches(tag, kind_filter=kind_filter, name_filter=name_filter):
                    yield TextInput(
                        file_id=file.id,
                        block_id=None,
                        granularity=Granularity.TAG,
                        text=block.text[tag.start_idx:tag.end_idx],
                        start_idx=tag.start_idx,
                        end_idx=tag.end_idx,
                        tag_kind=kind_filter,
                        tag_name=name_filter
                    )
