"""Collection of object specifications used to communicate with the NLPCloud API."""
from enum import Enum
from typing import Optional, TypedDict

from steamship import SteamshipError


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
    DISTILBERT_BASE_UNCASED_FINETUNED_SST_2_ENGLISH = (
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
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
        NlpCloudModel.GPT_J,
    ],
}


class NlpCloudOutputLabel(TypedDict):
    start: Optional[int]
    end: Optional[int]
    index: Optional[int]
    text: Optional[str]
    lemma: Optional[str]
    ws_after: Optional[str]


def validate_task_and_model(task: NlpCloudTask, model: NlpCloudModel):
    # We know from docs.nlpcloud.com that only certain task<>model pairings are valid.
    if task in VALID_TASK_MODELS:
        if model not in VALID_TASK_MODELS[task]:
            raise SteamshipError(
                message=f"Model {model.value} is not compatible with task {task.value}."
                    "Valid models for this task are: {[m.value for m in VALID_TASK_MODELS[task]]}."
            )
