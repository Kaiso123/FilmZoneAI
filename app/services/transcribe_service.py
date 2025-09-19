import logging
from pathlib import Path
from core.config import settings
from models.transcibeModels import TranscribeResponse
from utils.file_utils import ensure_dir, dir_is_empty, _has_cuda
from utils.transcribe_utils import _regroup_segments_using_word_timestamps, _apply_padding, _format_timestamp, load_audio_any
import tempfile
import os

logger = logging.getLogger(__name__)

# cache các model đã load
_loaded_models = {}

def _model_dir(model_id: str) -> Path:
    return Path(settings.STORAGE_DIR) / settings.TRANSCRIBE_DIR / model_id

def ensure_model_downloaded(model_id: str):
    """
    Nếu thư mục rỗng/không tồn tại thì gọi hàm download từ faster_whisper.
    NOTE: import download function **chỉ** khi cần.
    """
    model_dir = _model_dir(model_id)
    ensure_dir(str(model_dir))
    if dir_is_empty(str(model_dir)):
        logger.info("Model dir empty. Attempting to download model: %s -> %s", model_id, model_dir)
        try:
            from faster_whisper.utils import download_model
        except Exception:
            # fallback: try getattr on package
            try:
                import faster_whisper as fw
                download_model = getattr(fw, "download_model", None)
            except Exception:
                download_model = None

        if download_model is None:
            logger.warning("No download_model available in faster_whisper; ensure model files are present.")
            return

        download_model(size_or_id=model_id, output_dir=str(model_dir))
    else:
        logger.info("Model directory already present: %s", model_dir)

def load_model(model_id: str):
    """
    Load model **on demand**. Không import WhisperModel toàn cục.
    """
    if model_id in _loaded_models:
        return _loaded_models[model_id]

    ensure_model_downloaded(model_id)

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        logger.exception("Cannot import WhisperModel: %s", e)
        raise

    device = "cuda" if _has_cuda() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model_path = str(_model_dir(model_id))
    logger.info("Loading WhisperModel from %s (device=%s compute=%s)", model_path, device, compute_type)

    try:
        model = WhisperModel(model_path, device=device, compute_type=compute_type)
    except MemoryError as e:
        logger.exception("MemoryError while loading model: %s", e)
        raise
    except Exception as e:
        logger.exception("Error while loading model: %s", e)
        raise

    _loaded_models[model_id] = model
    return model

def transcribe(audio_path: str, model_id: str = "turbo") -> TranscribeResponse:
    """
    Transcribe audio -> trả về nội dung text và các đoạn (segments).
    """
    model = load_model(model_id)
    segments, info = model.transcribe(
        audio_path,
        condition_on_previous_text=True,
        temperature=0,
        word_timestamps=True,
        hallucination_silence_threshold=1
    )
    segment_list = list(segments)
    text = " ".join(seg.text for seg in segment_list)
    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segment_list]
    return TranscribeResponse(
        text=text,
        language=info.language,
        segments=segments,
    )

def transcribe2srt(audio_path: str, model_id: str = "turbo",
                   vad_threshold: float = 0.2,
                   min_speech_ms: int = 500,
                   min_silence_ms: int = 1000) -> str:
    """
    Transcribe audio -> trả về nội dung dạng SRT (string).
    Sử dụng Silero VAD để cắt đoạn trước, fallback về model.transcribe toàn file nếu VAD không khả dụng.
    """
    # 1) load VAD (deferred)
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        import librosa
        import soundfile as sf
        vad_available = True
    except Exception as e:
        logger.warning("silero_vad/librosa/soundfile not available (%s). Falling back to direct transcribe without VAD.", e)
        vad_available = False

    model = load_model(model_id)

    all_segments = []

    if vad_available:
        # load audio nếu file là video
        audio_data, sr = load_audio_any(audio_path, sr=16000)
        vad_model = load_silero_vad()
        speech_timestamps = get_speech_timestamps(
            audio_data,
            vad_model,
            sampling_rate=sr,
            threshold=vad_threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms
        )

        # nếu không detect speech, fallback transcribe full
        if not speech_timestamps:
            logger.info("VAD detected no speech segments, fall back to whole-file transcription.")
            vad_available = False
        else:
            detected_lang = None
            # transcribe each VAD segment and add offset
            for ts in speech_timestamps:
                start_sample = ts["start"]
                end_sample = ts["end"]
                seg_audio = audio_data[start_sample:end_sample]
                start_offset = start_sample / sr

                # write temp wav
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmpf.close()
                tmp_path = tmpf.name
                try:
                    sf.write(tmp_path, seg_audio, sr)
                    # transcribe segment (use path)
                    if detected_lang is None:
                        segments, info = model.transcribe(
                            tmp_path,
                            condition_on_previous_text=True,
                            temperature=0,
                            word_timestamps=True,
                            hallucination_silence_threshold=1
                        )
                        detected_lang = info.language
                    else:
                        segments, info = model.transcribe(
                            tmp_path,
                            condition_on_previous_text=True,
                            temperature=0,
                            language=detected_lang,   
                            word_timestamps=True,
                            hallucination_silence_threshold=1
                        )
                    segment_list = list(segments)
                    # offset timestamps
                    for segment in segment_list:
                        # some segments might be dict-like; handle both
                        if hasattr(segment, "start"):
                            segment.start = float(segment.start) + start_offset
                            segment.end = float(segment.end) + start_offset
                        # words
                        if hasattr(segment, "words") and segment.words:
                            for w in segment.words:
                                # ensure numeric
                                if hasattr(w, "start"):
                                    w.start = float(w.start) + start_offset
                                if hasattr(w, "end"):
                                    w.end = float(w.end) + start_offset
                    all_segments.extend(segment_list)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    if not vad_available:
        # direct full-file transcribe (no VAD)
        segments, info = model.transcribe(
            audio_path,
            temperature=0,
            word_timestamps=True,
            condition_on_previous_text=True,
            hallucination_silence_threshold=1,
        )
        all_segments = list(segments)

    # regroup using word timestamps (the function will handle segments without words)
    regrouped_both, regrouped_sentence = _regroup_segments_using_word_timestamps(all_segments,
                                                        max_gap=0.8,
                                                        max_chars=150,
                                                        max_words=30)
    padded_both = _apply_padding(regrouped_both)
    padded_sentence = _apply_padding(regrouped_sentence)

    # build SRT string
    srt_parts = []
    idx = 1
    for seg in padded_both:
        start_time = _format_timestamp(seg["start"])
        end_time = _format_timestamp(seg["end"])
        text = seg["text"].strip()
        if not text:
            continue
        srt_parts.append(f"{idx}\n{start_time} --> {end_time}\n{text}\n")
        idx += 1

    srt_str = "\n".join(srt_parts)
    return srt_str, padded_both, info.language, padded_sentence
