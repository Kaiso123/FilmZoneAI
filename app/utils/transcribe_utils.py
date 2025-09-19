import re
from types import SimpleNamespace
import os
import tempfile
import subprocess
import shutil
import librosa


# If input is video file (mp4, mkv, mov...), extract audio wav (16kHz, mono) using ffmpeg.
def load_audio_any(path: str, sr: int = 16000):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".mp4", ".mkv", ".mov", ".avi"]:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        wav_path = tmp.name
        cmd = ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), wav_path]
        subprocess.run(cmd, check=True)
        audio, _ = librosa.load(wav_path, sr=sr)
        os.remove(wav_path)
        return audio, sr
    else:
        return librosa.load(path, sr=sr)

# Format a timestamp in seconds to SRT format (HH:MM:SS,mmm)
def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# Convert segments to text
def _words_to_text(words):
    return "".join(w.word for w in words).strip()

# Split words into parts based on punctuation and other rules
def _split_words_by_punctuation(words, min_words_after_comma=3, split_by_comma=True):
    """
    Chia nhỏ danh sách từ (words) thành các phần (part) dựa trên dấu câu và các quy tắc khác.
    Nếu split_by_comma=True, chia cả theo dấu phẩy; nếu False, chỉ chia theo dấu kết câu.
    """
    sentence_end_re = re.compile(r"[.?!…]+$")
    parts, buf = [], []
    for i, w in enumerate(words):
        buf.append(w)
        w_text = w.word.strip()
        if sentence_end_re.search(w_text):
            parts.append(buf); buf = []; continue
        if split_by_comma:
            remaining = len(words) - (i + 1)
            if (w_text.endswith(",") or w_text == ",") and remaining >= min_words_after_comma and len(buf) > 3:
                parts.append(buf); buf = []; continue
    if buf: parts.append(buf)
    return parts

# Regroup segments using word-level timestamps
def _regroup_segments_using_word_timestamps(raw_segments,
                                           max_gap=0.8,
                                           max_chars=120,
                                           max_words=30,
                                           min_words_after_comma=3):
    """
    Sắp xếp lại các đoạn (segment) sử dụng timestamp của từng từ (word-level timestamps).
    Trả về hai danh sách: một chia theo cả dấu kết câu và dấu phẩy, một chỉ chia theo dấu kết câu.
    """
    normalized_segments = []
    for seg in raw_segments:
        words = getattr(seg, "words", None)
        if not words or len(words) == 0:
            pseudo = SimpleNamespace(text=seg.text.strip() or "", start=seg.start, end=seg.end)
            words = [pseudo] if pseudo.text else []
        normalized_segments.append(SimpleNamespace(
            start=seg.start,
            end=seg.end,
            words=list(words),
            text=getattr(seg, "text", "").strip()
        ))

    groups, buffer = [], None
    for seg in normalized_segments:
        seg_words = seg.words or []
        if buffer is None:
            buffer = {"start": seg.start, "end": seg.end, "words": list(seg_words)}
            continue
        gap = seg.start - buffer["end"]
        candidate_words = buffer["words"] + seg_words
        candidate_text = _words_to_text(candidate_words)
        if gap <= max_gap and len(candidate_words) <= max_words and len(candidate_text) <= max_chars:
            buffer["end"] = seg.end
            buffer["words"].extend(seg_words)
        else:
            groups.append(buffer)
            buffer = {"start": seg.start, "end": seg.end, "words": list(seg_words)}
    if buffer: groups.append(buffer)

    split_by_both, split_by_sentence, seen_both, seen_sentence = [], [], set(), set()
    for group in groups:
        # Split by both sentence-ending punctuation and commas
        parts_both = _split_words_by_punctuation(group["words"], min_words_after_comma, split_by_comma=True)
        for part in parts_both:
            if not part: continue
            text = _words_to_text(part).strip()
            start, end = float(part[0].start), float(part[-1].end)
            key = (text, round(start, 2), round(end, 2))
            if key in seen_both: 
                continue
            seen_both.add(key)
            if text: 
                split_by_both.append({"start": start, "end": end, "text": text})

        # Split by sentence-ending punctuation only
        parts_sentence = _split_words_by_punctuation(group["words"], min_words_after_comma, split_by_comma=False)
        for part in parts_sentence:
            if not part: continue
            text = _words_to_text(part).strip()
            start, end = float(part[0].start), float(part[-1].end)
            key = (text, round(start, 2), round(end, 2))
            if key in seen_sentence: 
                continue
            seen_sentence.add(key)
            if text: 
                split_by_sentence.append({"start": start, "end": end, "text": text})

    split_by_both.sort(key=lambda x: x["start"])
    split_by_sentence.sort(key=lambda x: x["start"])
    return split_by_both, split_by_sentence

# Apply padding to segments to avoid abrupt cuts
def _apply_padding(segments, max_gap=1.0):
    """
    Thêm khoảng đệm (padding) vào các đoạn để tránh bị cắt ngang câu
    hoặc các đoạn bị ngắt quãng đột ngột.
    giúp cho phụ đề mượt mà hơn khi hiển thị.
    """
    padded = []
    for i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        if i + 1 < len(segments):
            next_start = segments[i + 1]["start"]
            end += min((next_start - end) * 0.8, max_gap)
            if end < start: end = seg["end"]
        padded.append({"start": start, "end": end, "text": seg["text"]})
    return padded