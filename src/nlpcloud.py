import dataclasses
import logging
from enum import Enum
from typing import List, Optional, Dict, TypedDict

import requests
from steamship import SteamshipError, TagKind
from steamship.data.tags.tag import Tag

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

class NlpCloudRequest:
    model: NlpCloudModel
    task: NlpCloudTask
    text: str

    def __init__(self, model: NlpCloudModel, task: NlpCloudTask, text: str):
        self.model = model
        self.task = task
        self.text = text


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
            Tag.CreateRequest(kind=TagKind.ent, name=tag.get("type"), start=tag.get("start"), end=tag.get("end"))
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
                ret.append(Tag.CreateRequest(kind=TagKind.pos, name="noun_chunk", start=start, end=end, value=tag))
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
        raise SteamshipError(message=f"Result processing of NLPCloud task {task} is not yet implemented")
    elif task == NlpCloudTask.EMBEDDINGS:
        #: {embeddings: [float]}
        raise SteamshipError(message=f"Result processing of NLPCloud task {task} is not yet implemented")



@dataclasses.dataclass
class OneAiOutputBlock:
    block_id: str
    generating_step: str
    origin_block: str
    origin_span: List[int]
    text: str
    labels: List[OneAiOutputLabel]

    @staticmethod
    def from_dict(d: dict) -> Optional["OneAiOutputBlock"]:
        if d is None:
            return None

        return OneAiOutputBlock(
            block_id=d.get("block_id", None),
            generating_step=d.get("generating_step", None),
            origin_block=d.get("origin_block", None),
            origin_span=d.get("origin_span", None),
            text=d.get("text", None),
            labels=[OneAiOutputLabel.from_dict(l) for l in d.get("labels", [])]
        )


@dataclasses.dataclass
class OneAiResponse:
    input_text: str
    status: str
    error: str
    output: List[OneAiOutputBlock]

    @staticmethod
    def from_dict(d: dict) -> Optional["OneAiResponse"]:
        if d is None:
            return None

        return OneAiResponse(
            input_text=d.get("input_text", None),
            status=d.get("status", None),
            error=d.get("error", None),
            output=[OneAiOutputBlock.from_dict(b) for b in d.get("output", [])]
        )

    def to_tags(self) -> List[Tag.CreateRequest]:
        tags: List[Tag.CreateRequest] = []
        if self.output and self.output[0]:
            for label in self.output[0].labels:
                tags.append(label.to_steamship_tag())
        return tags


class NlpCloudClient:
    URL = "https://api.nlpcloud.io/v1"

    def __init__(self, key: str):
        self.key = key

    def request(self, request: NlpCloudRequest) -> OneAiResponse:
        """Performs an NlpCloud request. Throw a SteamshipError in the event of error or empty response.

        See: https://docs.nlpcloud.io/
        """

        headers = {
            "Authorization": f"Token {self.key}",
            "Content-Type": "application/json"
        }
        url = f"{NlpCloudClient.URL/{request.model.value}/{request.task.value}"

        request_dict = dataclasses.asdict(request)
        response = requests.post(
            url=OneAIClient.URL,
            headers=headers,
            json=request_dict,
        )

        if not response.ok:
            raise SteamshipError(
                message="Request to OneAI failed. Code={}. Body={}".format(response.status_code, response.text)
            )

        response_dict = response.json()
        if not response_dict:
            raise SteamshipError(
                message="Request from OneAI could not be interpreted as JSON."
            )

        try:
            ret = OneAiResponse.from_dict(response_dict)
            if not ret:
                raise SteamshipError(
                    message="Request from OneAI could not be interpreted as a OneAIResponse object."
                )
            return ret
        except Exception as ex:
            raise SteamshipError(
                message="Request from OneAI could not be interpreted as a OneAIResponse object. Exception: {}".format(
                    ex),
                error=ex
            )
