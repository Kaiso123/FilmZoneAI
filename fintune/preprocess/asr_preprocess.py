from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
import datasets
import torch
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
import librosa
import numpy as np
import random
from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, List, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def load_dataset_from_files_or_hf(path: str, split: Optional[str] = None, language: str = "en", test_mode: bool = False):
    """
    Load dataset từ local file hoặc HuggingFace Hub.
    - Khi test_mode=True:
        + Luôn dùng streaming (tránh tải toàn bộ parquet)
        + Chỉ lấy 5 mẫu đầu tiên
    """
    from itertools import islice
    from datasets import Dataset

    p = Path(path)
    if p.exists():
        # detect format
        if p.suffix.lower() in [".csv"]:
            file_format = "csv"
        elif p.suffix.lower() in [".json", ".jsonl"]:
            file_format = "json"
        elif p.suffix.lower() in [".tsv"]:
            file_format = "csv"
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")

        if file_format == "csv":
            ds = datasets.load_dataset("csv", data_files=str(p))
        elif file_format == "json":
            ds = datasets.load_dataset("json", data_files=str(p))
        else:
            raise ValueError("Unsupported format")

        # Load audio for local files
        def load_audio(example):
            audio_path = example["audio_path"]
            if Path(audio_path).exists():
                audio_array, sr = librosa.load(audio_path, sr=16000)
                example["audio"] = {"array": audio_array, "sampling_rate": sr}
            else:
                logger.warning(f"Audio file not found: {audio_path}")
                example["audio"] = {"array": np.array([]), "sampling_rate": 16000}
            return example

        ds = ds.map(load_audio)
        ds = ds if split is None else ds[split]
        if test_mode:
            ds = ds.select(range(min(5, len(ds))))
        return ds

    else:
        # HuggingFace dataset
        streaming = test_mode 
        logger.info(f"Loading HF dataset '{path}' with streaming={streaming}")
        ds = datasets.load_dataset(path, language, split=split or "train", streaming=streaming)

        if test_mode:
            from itertools import islice
            from datasets import Dataset
            logger.info("Test mode active.")
            ds = Dataset.from_list(list(islice(ds, 5)))

        return ds



def validate_dataset(ds, sample_n: int = 5):
    """
    Validate dataset for Whisper ASR (có cột 'audio' và 'sentence').
    Kiểm tra:
      - Có cột audio và sentence không
      - Thống kê null/empty transcripts và invalid audio
      - Trả summary dict (số lượng, sample records, độ dài trung bình audio/transcript)
    """
    info = {}
    if isinstance(ds, datasets.DatasetDict):
        if "train" in ds:
            dataset = ds["train"]
        else:
            dataset = ds
    else:
        dataset = ds

    if "audio" not in dataset.column_names or "sentence" not in dataset.column_names and "text" not in dataset.column_names:
        raise ValueError("Dataset phải có cột 'audio' và 'sentence'")
    if "sentence" in dataset.column_names:
        sentence = "sentence"
    else:
        sentence = "text"
    total = len(dataset)
    info["n_records"] = total

    null_audio = 0
    null_sentence = 0
    audio_lengths = []  # seconds
    transcript_lengths = []

    for ex in dataset:
        audio = ex.get("audio", {})
        audio_array = audio.get("array", np.array([]))
        sentence = ex.get(sentence, "").strip()


        if len(audio_array) == 0:
            null_audio += 1
        else:
            audio_lengths.append(len(audio_array) / 16000.0)  # seconds at 16kHz

        if not sentence:
            null_sentence += 1

        transcript_lengths.append(len(sentence.split()))

    info["null_audio"] = null_audio
    info["null_sentence"] = null_sentence
    info["avg_audio_len_sec"] = np.mean(audio_lengths) if audio_lengths else 0
    info["avg_transcript_len"] = np.mean(transcript_lengths) if transcript_lengths else 0

    # sample vài record
    indices = random.sample(range(total), min(sample_n, total))
    info["sample_records"] = [dataset[i] for i in indices]

    return info


def prepare_for_whisper_asr(
    dataset,
    processor_name_or_path: str,
    max_audio_length: int = 30 * 16000,  # 30 seconds at 16kHz
    max_target_length: int = 448,  # Whisper max labels
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1
):
    """
    Chuẩn bị dataset để fine-tune Whisper cho ASR task.
    - dataset: Dataset hoặc DatasetDict HuggingFace, cột 'audio' và 'sentence'
    - processor_name_or_path: checkpoint processor (feature_extractor + tokenizer)
    - max_audio_length: max audio samples (trim/pad)
    - max_target_length: max label tokens
    - train_split, val_split, test_split: tỷ lệ chia tập train, validation, test
    Trả về: (DatasetDict với train/validation/test, processor)
    """
    try:
        processor = WhisperProcessor.from_pretrained(processor_name_or_path)
    except Exception as e:
        logger.warning(f"Không thể load processor từ {processor_name_or_path}: {e}")
        logger.info("Sử dụng tokenizer của model gốc 'openai/whisper'")
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    # Kiểm tra và chuẩn hóa dataset
    if isinstance(dataset, datasets.DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            dataset = dataset

    # Filter invalid samples (empty audio/sentence)
    try:
        dataset = dataset.filter(lambda x: len(x["audio"]["array"]) > 0 and x["sentence"].strip() != "")
    except Exception as e:
        dataset = dataset.filter(lambda x: len(x["audio"]["array"]) > 0 and x["text"].strip() != "")

    if len(dataset) < 10: 
        train_size = max(1, int(0.6 * len(dataset)))
        val_size = max(1, int(0.2 * len(dataset)))
        dataset_dict = dataset.train_test_split(test_size=len(dataset) - train_size, seed=42)["train"].train_test_split(test_size=val_size / (len(dataset) - train_size), seed=42)
        dataset_dict = datasets.DatasetDict({
            "train": dataset_dict["train"],
            "validation": dataset_dict["test"],
            "test": dataset  
        })
    else:
        train_val = dataset.train_test_split(test_size=(val_split + test_split), shuffle=True, seed=42)
        val_test = train_val["test"].train_test_split(test_size=test_split / (val_split + test_split), shuffle=True, seed=42)
        dataset_dict = datasets.DatasetDict({
            "train": train_val["train"],
            "validation": val_test["train"],
            "test": val_test["test"]
        })
        
    # Chia dataset thành train, validation, test
    train_val = dataset.train_test_split(test_size=(val_split + test_split), shuffle=True, seed=42)
    val_test = train_val["test"].train_test_split(test_size=test_split / (val_split + test_split), shuffle=True, seed=42)

    dataset_dict = datasets.DatasetDict({
        "train": train_val["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })

    def preprocess_function(examples):
        sentence_col = "sentence" if "sentence" in examples else "text"
        targets = examples[sentence_col]  

        # Extract audio arrays từ list of dicts (HF audio format)
        audio_arrays = []
        for audio_data in examples["audio"]:
            if isinstance(audio_data, dict):
                arr = audio_data.get("array")
                if arr is None and "path" in audio_data:  
                    path = audio_data["path"]
                    if Path(path).exists():
                        arr, _ = librosa.load(path, sr=16000)
                    else:
                        arr = np.array([])
                audio_arrays.append(arr or np.array([]))
            elif isinstance(audio_data, str):  
                arr, _ = librosa.load(audio_data, sr=16000)
                audio_arrays.append(arr)
            else:
                raise TypeError(f"Unsupported audio type: {type(audio_data)}")
            
        # Feature extraction
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
        )
        # Tokenize labels
        labels = tokenizer(
            targets,

        )
        inputs["labels"] = labels["input_ids"]
        inputs["audio_arrays"] = audio_arrays  
        
        return inputs

    tokenized_dict = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        tokenized_dict[split] = dataset_dict[split].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_dict[split].column_names
        )

    return tokenized_dict, processor

@dataclass
class CustomDataCollatorForASR:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
    
def make_data_collator(processor, decoder_start_token_id):
    """
    Return a suitable data collator for ASR training (pad features and labels).
    """
    return CustomDataCollatorForASR(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id
    )