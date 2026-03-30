import os

from transformers import AutoTokenizer, CLIPTextModelWithProjection

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # needed to suppress warning about potential deadlock

_TOKENIZER_NAME = "openai/clip-vit-large-patch14"  # "openai/clip-vit-base-patch32"
_CACHE_DIR = os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip"))
_lang_emb_model = None
_tz = None

LANG_EMB_OBS_KEY = "lang_emb"


def _ensure_clip_loaded():
    global _lang_emb_model, _tz
    if _lang_emb_model is None:
        _lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
            _TOKENIZER_NAME,
            cache_dir=_CACHE_DIR,
        ).eval()
    if _tz is None:
        _tz = AutoTokenizer.from_pretrained(_TOKENIZER_NAME, TOKENIZERS_PARALLELISM=True)


def get_lang_emb(lang):
    if lang is None:
        return None

    _ensure_clip_loaded()

    tokens = _tz(
        text=lang,                   # the sentence to be encoded
        add_special_tokens=True,             # Add [CLS] and [SEP]
        max_length=25,  # maximum length of a sentence
        padding="max_length",
        return_attention_mask=True,        # Generate the attention mask
        return_tensors="pt",               # ask the function to return PyTorch tensors
    )
    lang_emb = _lang_emb_model(**tokens)['text_embeds'].detach()[0]

    return lang_emb


def get_lang_emb_shape():
    return list(get_lang_emb('dummy').shape)
