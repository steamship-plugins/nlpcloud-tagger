import logging
import time
from typing import Dict, List, Optional

import requests
from steamship import DocTag, SteamshipError, TagKind
from steamship.data.tags.tag import Tag

from nlpcloud.api_spec import (NlpCloudModel, NlpCloudTask,
                               validate_task_and_model)


def nlp_cloud_response_to_steamship_tag(
    task: NlpCloudTask, input_text: str, response: Dict
) -> List[Tag.CreateRequest]:
    if task == NlpCloudTask.ENTITIES:
        # {type: str, start: int, end: int}
        return [
            Tag.CreateRequest(
                kind=TagKind.ent,
                name=tag.get("type"),
                start_idx=tag.get("start"),
                end_idx=tag.get("end"),
            )
            for tag in response.get("entities", [])
        ]
    elif task == NlpCloudTask.NOUNS_CHUNKS:
        # {text: str, root_text: str, root_dep: str, root_head_text: str}
        # Assume: start, end is first match of text in input
        ret = []
        for tag in response.get("noun_chunks", []):
            text = tag.get("text")
            try:
                start = input_text.index(text)
                end = start + len(text)
                ret.append(
                    Tag.CreateRequest(
                        kind=TagKind.pos,
                        name="noun_chunk",
                        start_idx=start,
                        end_idx=end,
                        value=tag,
                    )
                )
            except Exception:
                logging.error(
                    f"Text: {text} was not found for 'noun_chunk' in input text."
                )
        return ret
    elif task == NlpCloudTask.CLASSIFICATION:
        # { scores: [float], labels: [str]}
        ret = []
        scores = response.get("scores", [])
        for index, label in enumerate(response.get("labels", [])):
            if index > 0 and index < len(scores):
                score = scores[index]
                ret.append(
                    Tag.CreateRequest(kind="label", name=label, value={"score": score})
                )
        return ret
    elif task == NlpCloudTask.INTENT_CLASSIFICATION:
        # {intent}
        intent = response.get("intent")
        if intent is None:
            return []
        return [Tag.CreateRequest(kind="intent", name=intent)]
    elif task == NlpCloudTask.KEY_PHRASE_EXTRACTION:
        # {keywords_and_keyphrases: [keyword]}
        phrases = response.get("keywords_and_keyphrases")
        if phrases is None:
            return []
        return [Tag.CreateRequest(kind="keyword", name=phrase) for phrase in phrases]
    elif task == NlpCloudTask.LANGUAGE_DETECTION:
        # {languages: [{code: score}]}
        ret = []
        languages = response.get("languages", [])
        for language in languages:
            for language_code in language:
                score = language[language_code]
                ret.append(
                    Tag.CreateRequest(
                        kind="language", name=language_code, value={"score": score}
                    )
                )
        return ret
    elif task == NlpCloudTask.SENTENCE_DEPENDENCIES:
        # UGH. The start & end are TOKEN indices not char indices..
        # TODO: Have to reconstruct the char indices in the input text given the list of words.
        # {sentence_dependencies: {sentence: str, dependencies: {words: [{text: str, tag: str}], args: [{start, end, label, text, dir: str]}}
        raise SteamshipError(
            message=f"Result processing of NLPCloud task {task} is not yet implemented"
        )
    elif task == NlpCloudTask.SENTIMENT:
        # {scored_labels: [ {label, score} ]}
        return [
            Tag.CreateRequest(
                kind=label.get("emotion"),
                name=label.get("label"),
                value={
                    "score": label.get("score")
                },
            )
            for label in response.get("scored_labels", [])
        ]
    elif task == NlpCloudTask.TOKENS:
        # {tokens: [{ start, end, index, text, lemma, ws_after}]}
        return [
            Tag.CreateRequest(
                kind=TagKind.doc,
                name=DocTag.token,
                start_idx=token.get("start"),
                end_idx=token.get("end"),
                value={
                    "ws_after": token.get("ws_after"),
                    "lemma": token.get("lemma"),
                    "text": token.get("text"),
                },
            )
            for token in response.get("tokens", [])
        ]
    elif task == NlpCloudTask.EMBEDDINGS:
        #: {embeddings: [float]}
        raise SteamshipError(
            message=f"Result processing of NLPCloud task {task} is not yet implemented"
        )


def nlp_cloud_requests(
    task: NlpCloudTask,
    inputs: List[str],
    labels: Optional[List[str]] = None,
    multi_class: Optional[bool] = False,
) -> List[dict]:
    if task in [
        NlpCloudTask.ENTITIES,
        NlpCloudTask.NOUNS_CHUNKS,
        NlpCloudTask.INTENT_CLASSIFICATION,
        NlpCloudTask.KEY_PHRASE_EXTRACTION,
        NlpCloudTask.LANGUAGE_DETECTION,
        NlpCloudTask.SENTENCE_DEPENDENCIES,
        NlpCloudTask.SENTIMENT,
        NlpCloudTask.TOKENS,
    ]:
        # {text: str}
        return [{"text": s} for s in inputs]
    elif task == NlpCloudTask.CLASSIFICATION:
        # {text: str, multi_class: bool, labels: str[]}
        if labels is None:
            raise SteamshipError(
                f"Task type {task} requires non-null `labels` setting to be configured."
            )
        return [
            {"text": s, "labels": labels, "multi_class": multi_class} for s in inputs
        ]
    elif task == NlpCloudTask.EMBEDDINGS:
        #: {sentences: [str]}
        return [{"sentences": inputs}]

    raise SteamshipError(f"Unable to prepare NLP Cloud input for task type {task}.")


class NlpCloudClient:
    URL = "https://api.nlpcloud.io/v1"

    def __init__(self, key: str):
        self.key = key

    def request(
        self, task: NlpCloudTask, model: NlpCloudModel, inputs: List[str], **kwargs
    ) -> List[List[Tag.CreateRequest]]:
        """Performs an NlpCloud request. Throw a SteamshipError in the event of error or empty response.

        See: https://docs.nlpcloud.io/
        """

        validate_task_and_model(task, model)

        headers = {
            "Authorization": f"Token {self.key}",
            "Content-Type": "application/json",
        }
        url = f"{NlpCloudClient.URL}/{model.value}/{task.value}"
        input_dict = nlp_cloud_requests(task, inputs, **kwargs)

        ret = []
        for json_body, text_input in zip(input_dict, inputs):
            time.sleep(1.1)
            response = requests.post(url=url, headers=headers, json=json_body)

            if not response.ok:
                raise SteamshipError(
                    message="Request to NLP Cloud failed. Code={}. Body={}".format(
                        response.status_code, response.text
                    )
                )

            response_dict = response.json()
            if not response_dict:
                raise SteamshipError(
                    message="Request from NLP Cloud could not be interpreted as JSON."
                )

            ret.append(
                nlp_cloud_response_to_steamship_tag(task, text_input, response_dict)
            )
        return ret
