import logging
from pathlib import Path
from ..core.config import settings
from ..utils.file_utils import ensure_dir, dir_is_empty, _has_cuda
import subprocess
from ..utils.transcribe_utils import _format_timestamp

import ctranslate2
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_loaded_translators = {}

def _hf_model_dir(model_id: str) -> Path:
    """Thư mục chứa HuggingFace model gốc (để convert)."""
    return Path(settings.STORAGE_DIR) / settings.TRANSLATE_DIR / "hf_models" / model_id

def _ct2_model_dir(model_id: str) -> Path:
    """Thư mục chứa CTranslate2 model đã convert."""
    return Path(settings.STORAGE_DIR) / settings.TRANSLATE_DIR / "ctranslate2" / model_id

def ensure_model_downloaded(model_id: str):
    """Đảm bảo model HuggingFace đã được tải về local."""
    model_dir = _hf_model_dir(model_id)
    if not model_dir.exists() or dir_is_empty(str(model_dir)):
        logger.info("Downloading HuggingFace model %s to %s...", model_id, model_dir)
        snapshot_download(
            repo_id=f"{model_id}",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False
        )
    return model_dir

def ensure_model_converted(model_id: str):
    model_dir = _ct2_model_dir(model_id)
    ensure_dir(str(model_dir))
    logger.info("Ensuring Directory exists: %s", model_dir)

    if dir_is_empty(str(model_dir)):
        logger.info("Converting model %s to CTranslate2 format at %s...", model_id, model_dir)
        hf_model_path = ensure_model_downloaded(model_id)
        cmd = [
            "ct2-transformers-converter",
            "--model", str(hf_model_path),
            "--output_dir", str(model_dir),
            "--force"
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Error converting model: %s", e)
            raise
    return model_dir

def get_translator(model_id: str):
    if model_id in _loaded_translators:
        return _loaded_translators[model_id]

    model_dir = ensure_model_converted(model_id)

    try:
        device = "cuda" if _has_cuda() else "cpu"
        translator = ctranslate2.Translator(str(model_dir), device=device)
    except Exception as e:
        logger.error("Failed to load translator: %s", e)
        raise

    _loaded_translators[model_id] = translator
    return translator

#envit5-translation
def translate_text(inputs: list[str], model_id: str = "envit5-translation") -> list[str]:
    """
    Dịch list các câu (inputs) sang tiếng Việt hoặc tiếng anh.
    """
    translator = get_translator(model_id)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(_hf_model_dir(model_id))

    tokenized = [tokenizer.tokenize(text) for text in inputs]
    results = translator.translate_batch(tokenized)
    outputs = []
    for res in results:
        hyp = res.hypotheses[0]
        outputs.append(tokenizer.convert_tokens_to_string(hyp))
    return outputs

def translate_segments(segments: list[dict], language: str, model_id: str = "envit5-translation"):
    """
    Dịch list segments (mỗi segment có start, end, text).
    Trả về dict gồm:
      - "segments": list[dict] với start, end giữ nguyên, text được dịch
      - "srt_str": chuỗi SRT
    """
    if not segments:
        return "", []

    translator = get_translator(model_id)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(_hf_model_dir(model_id))
    texts = [f"{language}: {seg['text']}" for seg in segments]
    tokenized = [tokenizer.tokenize(text) for text in texts]
    results = translator.translate_batch(tokenized)

    outputs = []
    for seg, res in zip(segments, results):
        hyp = res.hypotheses[0]
        translated_text = tokenizer.convert_tokens_to_string(hyp)
        outputs.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": translated_text[3:]
        })

    # build SRT string
    srt_lines = []
    for i, seg in enumerate(outputs, 1):
        start_time = _format_timestamp(seg["start"])
        end_time = _format_timestamp(seg["end"])
        srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{seg['text'][3:]}\n")
    srt_str = "\n".join(srt_lines)

    return srt_str, outputs
