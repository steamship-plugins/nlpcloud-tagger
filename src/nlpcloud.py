import dataclasses
import logging
from enum import Enum
from typing import List, Optional, Dict, TypedDict

import requests
from steamship import SteamshipError, TagKind, DocTag
from steamship.data.tags.tag import Tag
import time

class NlpCloudTask(str, Enum):
    ENTITIES = "entities"
    NOUNS_CHUNKS = "nouns-chunks"
    CLASSIFICATION = "classification"
    INTENT_CLASSIFICATION = "intent-classification"
    KEY_PHRASE_EXTRACTION = "kw-kp-extraction"
    LANGUAGE_DETECTION = "langdetection"
    SENTENCE_DEPENDENCIES = "sentence-dependencies"
    SENTIMENT = "sentiment"
    TOKENS = "tokens"
    EMBEDDINGS = "embeddings"

class NlpCloudModel(str, Enum):
    PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2 = "paraphrase-multilingual-mpnet-base-v2"
    BART_LARGE_MNLI_YAHOO_ANSWERS = "bart-large-mnli-yahoo-answers"
    XLM_ROBERTA_LARGE_XNLI = "xlm-roberta-large-xnli"
    DISTILBERT_BASE_UNCASED_FINETUNED_SST_2_ENGLISH = "distilbert-base-uncased-finetuned-sst-2-english"
    DISTILBERT_BASE_UNCASED_EMOTION = "distilbert-base-uncased-emotion"
    FINBERT = "finbert"
    GPT_J = "gpt-j"
    FAST_GPT_J = "fast-gpt-j"
    FINETUNED_GPT_NEOX_20B = "finetuned-gpt-neox-20b"
    PYTHON_LANGDETECT = "python-langdetect"
    EN_CORE_WEB_LG = "en_core_web_lg"
    JA_GINZA_ELECTRA = "ja_ginza_electra"
    JA_CORE_NEWS_LG = "ja_core_news_lg"
    FR_CORE_NEWS_LG = "fr_core_news_lg"
    ZH_CORE_WEB_LG = "zh_core_web_lg"
    DA_CORE_NEWS_LG = "da_core_news_lg"
    NL_CORE_NEWS_LG = "nl_core_news_lg"
    DE_CORE_NEWS_LG = "de_core_news_lg"
    EL_CORE_NEWS_LG = "el_core_news_lg"
    IT_CORE_NEWS_LG = "it_core_news_lg"
    LT_CORE_NEWS_LG = "lt_core_news_lg"
    NB_CORE_NEWS_LG = "nb_core_news_lg"
    PL_CORE_NEWS_LG = "pl_core_news_lg"
    PT_CORE_NEWS_LG = "pt_core_news_lg"
    RO_CORE_NEWS_LG = "ro_core_news_lg"
    ES_CORE_NEWS_LG = "es_core_news_lg"

VALID_TASK_MODELS = {
    NlpCloudTask.ENTITIES: [
        NlpCloudModel.FAST_GPT_J,
        NlpCloudModel.FINETUNED_GPT_NEOX_20B,
        NlpCloudModel.EN_CORE_WEB_LG,
        NlpCloudModel.JA_GINZA_ELECTRA,
        NlpCloudModel.JA_CORE_NEWS_LG,
        NlpCloudModel.FR_CORE_NEWS_LG,
        NlpCloudModel.ZH_CORE_WEB_LG,
        NlpCloudModel.DA_CORE_NEWS_LG,
        NlpCloudModel.NL_CORE_NEWS_LG,
        NlpCloudModel.DE_CORE_NEWS_LG,
        NlpCloudModel.EL_CORE_NEWS_LG,
        NlpCloudModel.IT_CORE_NEWS_LG,
        NlpCloudModel.LT_CORE_NEWS_LG,
        NlpCloudModel.NB_CORE_NEWS_LG,
        NlpCloudModel.PL_CORE_NEWS_LG,
        NlpCloudModel.PT_CORE_NEWS_LG,
        NlpCloudModel.RO_CORE_NEWS_LG,
        NlpCloudModel.ES_CORE_NEWS_LG,
    ],
    NlpCloudTask.NOUNS_CHUNKS: [
        NlpCloudModel.EN_CORE_WEB_LG,
        NlpCloudModel.JA_GINZA_ELECTRA,
        NlpCloudModel.JA_CORE_NEWS_LG,
        NlpCloudModel.FR_CORE_NEWS_LG,
        NlpCloudModel.ZH_CORE_WEB_LG,
        NlpCloudModel.DA_CORE_NEWS_LG,
        NlpCloudModel.NL_CORE_NEWS_LG,
        NlpCloudModel.DE_CORE_NEWS_LG,
        NlpCloudModel.EL_CORE_NEWS_LG,
        NlpCloudModel.IT_CORE_NEWS_LG,
        NlpCloudModel.LT_CORE_NEWS_LG,
        NlpCloudModel.NB_CORE_NEWS_LG,
        NlpCloudModel.PL_CORE_NEWS_LG,
        NlpCloudModel.PT_CORE_NEWS_LG,
        NlpCloudModel.RO_CORE_NEWS_LG,
        NlpCloudModel.ES_CORE_NEWS_LG,
    ],
    NlpCloudTask.CLASSIFICATION: [
        NlpCloudModel.BART_LARGE_MNLI_YAHOO_ANSWERS,
        NlpCloudModel.XLM_ROBERTA_LARGE_XNLI,
        NlpCloudModel.FAST_GPT_J,
        NlpCloudModel.FINETUNED_GPT_NEOX_20B,
    ],
    NlpCloudTask.INTENT_CLASSIFICATION: [
        NlpCloudModel.FAST_GPT_J,
    ],
    NlpCloudTask.KEY_PHRASE_EXTRACTION: [
        NlpCloudModel.FAST_GPT_J,
        NlpCloudModel.FINETUNED_GPT_NEOX_20B,
    ],
    NlpCloudTask.LANGUAGE_DETECTION: [
        NlpCloudModel.PYTHON_LANGDETECT,
    ],
    NlpCloudTask.SENTENCE_DEPENDENCIES: [
        NlpCloudModel.EN_CORE_WEB_LG,
        NlpCloudModel.JA_GINZA_ELECTRA,
        NlpCloudModel.JA_CORE_NEWS_LG,
        NlpCloudModel.FR_CORE_NEWS_LG,
        NlpCloudModel.ZH_CORE_WEB_LG,
        NlpCloudModel.DA_CORE_NEWS_LG,
        NlpCloudModel.NL_CORE_NEWS_LG,
        NlpCloudModel.DE_CORE_NEWS_LG,
        NlpCloudModel.EL_CORE_NEWS_LG,
        NlpCloudModel.IT_CORE_NEWS_LG,
        NlpCloudModel.LT_CORE_NEWS_LG,
        NlpCloudModel.NB_CORE_NEWS_LG,
        NlpCloudModel.PL_CORE_NEWS_LG,
        NlpCloudModel.PT_CORE_NEWS_LG,
        NlpCloudModel.RO_CORE_NEWS_LG,
        NlpCloudModel.ES_CORE_NEWS_LG,
    ],
    NlpCloudTask.SENTIMENT: [
        NlpCloudModel.DISTILBERT_BASE_UNCASED_FINETUNED_SST_2_ENGLISH,
        NlpCloudModel.DISTILBERT_BASE_UNCASED_EMOTION,
    ],
    NlpCloudTask.TOKENS: [
        NlpCloudModel.EN_CORE_WEB_LG,
        NlpCloudModel.JA_GINZA_ELECTRA,
        NlpCloudModel.JA_CORE_NEWS_LG,
        NlpCloudModel.FR_CORE_NEWS_LG,
        NlpCloudModel.ZH_CORE_WEB_LG,
        NlpCloudModel.DA_CORE_NEWS_LG,
        NlpCloudModel.NL_CORE_NEWS_LG,
        NlpCloudModel.DE_CORE_NEWS_LG,
        NlpCloudModel.EL_CORE_NEWS_LG,
        NlpCloudModel.IT_CORE_NEWS_LG,
        NlpCloudModel.LT_CORE_NEWS_LG,
        NlpCloudModel.NB_CORE_NEWS_LG,
        NlpCloudModel.PL_CORE_NEWS_LG,
        NlpCloudModel.PT_CORE_NEWS_LG,
        NlpCloudModel.RO_CORE_NEWS_LG,
        NlpCloudModel.ES_CORE_NEWS_LG,
    ],
    NlpCloudTask.EMBEDDINGS: [
        NlpCloudModel.PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2,
        NlpCloudModel.GPT_J
    ]
}

class NlpCloudOutputLabel(TypedDict):
    start: Optional[int]
    end: Optional[int]
    index: Optional[int]
    text: Optional[str]
    lemma: Optional[str]
    ws_after: Optional[str]


def nlp_cloud_response_to_steamship_tag(task: NlpCloudTask, input_text: str, response: Dict) -> List[Tag.CreateRequest]:
    if task == NlpCloudTask.ENTITIES:
        # {type: str, start: int, end: int}
        return [
            Tag.CreateRequest(kind=TagKind.ent, name=tag.get("type"), start_idx=tag.get("start"), end_idx=tag.get("end"))
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
                ret.append(Tag.CreateRequest(kind=TagKind.pos, name="noun_chunk", start_idx=start, end_idx=end, value=tag))
            except:
                logging.error(f"Text: {text} was not found for 'noun_chunk' in input text.")
        return ret
    elif task == NlpCloudTask.CLASSIFICATION:
        # { scores: [float], labels: [str]}
        ret = []
        scores = response.get("scores", [])
        for index, label in enumerate(response.get("labels", [])):
            if index > 0 and index < len(scores):
                score = scores[index]
                ret.append(Tag.CreateRequest(kind="label", name=label, value={"score": score}))
        return ret
    elif task == NlpCloudTask.INTENT_CLASSIFICATION:
        # {intent}
        intent = response.get("intent")
        if intent is None:
            return []
        return [
            Tag.CreateRequest(kind="intent", name=intent)
        ]
    elif task == NlpCloudTask.KEY_PHRASE_EXTRACTION:
        # {keywords_and_keyphrases: [keyword]}
        phrases = response.get("keywords_and_keyphrases")
        if phrases is None:
            return []
        return [
            Tag.CreateRequest(kind="keyword", name=phrase)
            for phrase in phrases
        ]
    elif task == NlpCloudTask.LANGUAGE_DETECTION:
        # {languages: [{code: score}]}
        ret = []
        languages = response.get("languages", [])
        for language in languages:
            for language_code in language:
                score = language[language_code]
                ret.append(Tag.CreateRequest(kind="language", name=language, value={"score": score}))
        return ret
    elif task == NlpCloudTask.SENTENCE_DEPENDENCIES:
        # UGH. The start & end are TOKEN indices not char indices..
        # TODO: Have to reconstruct the char indices in the input text given the list of words.
        # {sentence_dependencies: {sentence: str, dependencies: {words: [{text: str, tag: str}], args: [{start, end, label, text, dir: str]}}
        raise SteamshipError(message=f"Result processing of NLPCloud task {task} is not yet implemented")
    elif task == NlpCloudTask.SENTIMENT:
        # {scored_labels: [ {label, score} ]}
        raise SteamshipError(message=f"Result processing of NLPCloud task {task} is not yet implemented")
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
                    "text": token.get("text")
                }
            )
            for token in response.get("tokens", [])
        ]
    elif task == NlpCloudTask.EMBEDDINGS:
        #: {embeddings: [float]}
        raise SteamshipError(message=f"Result processing of NLPCloud task {task} is not yet implemented")


def nlp_cloud_requests(task: NlpCloudTask, inputs: List[str], labels: Optional[List[str]] = None, multi_class: Optional[bool] = False) -> List[dict]:
    if task in [
        NlpCloudTask.ENTITIES,
        NlpCloudTask.NOUNS_CHUNKS,
        NlpCloudTask.INTENT_CLASSIFICATION,
        NlpCloudTask.KEY_PHRASE_EXTRACTION,
        NlpCloudTask.LANGUAGE_DETECTION,
        NlpCloudTask.SENTENCE_DEPENDENCIES,
        NlpCloudTask.SENTIMENT,
        NlpCloudTask.TOKENS
    ]:
        # {text: str}
        return [{"text": s} for s in inputs]
    elif task == NlpCloudTask.CLASSIFICATION:
        # {text: str, multi_class: bool, labels: str[]}
        if labels is None:
            raise SteamshipError(f"Task type {task} requires non-null `labels` setting to be configured.")
        return [{"text": s, "labels": labels, "multi_class": multi_class} for s in inputs]
    elif task == NlpCloudTask.EMBEDDINGS:
        #: {sentences: [str]}
        return [{"sentences": inputs}]

    raise SteamshipError(f"Unable to prepare NLP Cloud input for task type {task}.")

def validate_task_and_model(task: NlpCloudTask, model: NlpCloudModel):
    # We know from docs.nlpcloud.com that only certain task<>model pairings are valid.
    if task in VALID_TASK_MODELS:
        if model not in VALID_TASK_MODELS[task]:
            raise SteamshipError(
                message=f"Model {model.value} is not compatible with task {task.value}. Valid models for this task are: {[m.value for m in VALID_TASK_MODELS[task]]}.")


class NlpCloudClient:
    URL = "https://api.nlpcloud.io/v1"

    def __init__(self, key: str):
        self.key = key

    def request(self, task: NlpCloudTask, model: NlpCloudModel, inputs: List[str], **kwargs) -> List[List[Tag.CreateRequest]]:
        """Performs an NlpCloud request. Throw a SteamshipError in the event of error or empty response.

        See: https://docs.nlpcloud.io/
        """

        validate_task_and_model(task, model)

        headers = {
            "Authorization": f"Token {self.key}",
            "Content-Type": "application/json"
        }
        url = f"{NlpCloudClient.URL}/{model.value}/{task.value}"
        input_dict = nlp_cloud_requests(task, inputs, **kwargs)

        ret = []
        for json_body, text_input in zip(input_dict, inputs):
            time.sleep(1.1)
            response = requests.post(url=url, headers=headers, json=json_body)

            if not response.ok:
                raise SteamshipError(
                    message="Request to NLP Cloud failed. Code={}. Body={}".format(response.status_code, response.text)
                )

            response_dict = response.json()
            if not response_dict:
                raise SteamshipError(
                    message="Request from NLP Cloud could not be interpreted as JSON."
                )

            ret.append(nlp_cloud_response_to_steamship_tag(task, text_input, response_dict))
        return ret
