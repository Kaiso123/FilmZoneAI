from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
import datasets
from transformers import AutoTokenizer
import math
import random

logger = logging.getLogger(__name__)


def load_dataset_from_files(path: str, src_col: str = "src", tgt_col: str = "tgt", file_format: Optional[str]=None, s: str = None):
    """
    Load dataset from a local file (json/csv/tsv) or a HF dataset id.
    Returns a datasets.Dataset or datasets.DatasetDict (if a split file).
    """
    p = Path(path)
    if p.exists():
        # detect format
        if file_format is None:
            suffix = p.suffix.lower()
            if suffix in [".csv"]:
                file_format = "csv"
            elif suffix in [".json", ".jsonl"]:
                file_format = "json"
            elif suffix in [".tsv"]:
                file_format = "csv"
            else:
                raise ValueError("Unsupported file type: %s" % suffix)
        if file_format == "csv":
            return datasets.load_dataset("csv", data_files=str(p))
        elif file_format == "json":
            return datasets.load_dataset("json", data_files=str(p))
        else:
            raise ValueError("Unsupported format")
    else:
        # assume HF dataset id
        return datasets.load_dataset(path, s)


def validate_dataset(ds, src_lang="en", tgt_lang="vi", sample_n: int = 5):
    """
    Validate dataset kiểu HuggingFace Translation (có cột 'translation': {lang: text}).
    Kiểm tra:
      - Có cột translation không
      - Có key src_lang và tgt_lang trong translation không
      - Thống kê null/empty
      - Trả summary dict (số lượng, sample records, độ dài trung bình)
    """
    info = {}
    if isinstance(ds, datasets.DatasetDict):
        if "train" in ds:
            dataset = ds["train"]  # Nếu là DatasetDict, lấy split train
        else:
            dataset = ds

    if "translation" not in dataset.column_names:
        raise ValueError("Dataset không có cột 'translation'")

    total = len(dataset)
    info["n_records"] = total

    null_src = 0
    null_tgt = 0
    lengths_src = []
    lengths_tgt = []

    for ex in dataset:
        trans = ex.get("translation", {})
        src_text = trans.get(src_lang, "")
        tgt_text = trans.get(tgt_lang, "")

        if not src_text or src_text.strip() == "":
            null_src += 1
        if not tgt_text or tgt_text.strip() == "":
            null_tgt += 1

        lengths_src.append(len(src_text.split()))
        lengths_tgt.append(len(tgt_text.split()))

    info["null_src"] = null_src
    info["null_tgt"] = null_tgt
    info["avg_len_src"] = sum(lengths_src) / len(lengths_src) if lengths_src else 0
    info["avg_len_tgt"] = sum(lengths_tgt) / len(lengths_tgt) if lengths_tgt else 0

    # sample vài record
    import random
    info["sample_records"] = [dataset[i] for i in 3]

    return info



def prepare_for_t5_translation(dataset,
                               tokenizer_name_or_path: str,
                               src_lang: str = "en",
                               tgt_lang: str = "vi",
                               max_source_length: int = 128,
                               max_target_length: int = 128,
                               prefix: str = "translate",
                               train_split: float = 0.7,
                               val_split: float = 0.2,
                               test_split: float = 0.1):
    """
    Chuẩn bị dataset để fine-tune T5 cho translation task.
    - dataset: Dataset hoặc DatasetDict HuggingFace, cột 'translation'
    - tokenizer_name_or_path: checkpoint tokenizer
    - src_lang: ngôn ngữ nguồn (vd 'en')
    - tgt_lang: ngôn ngữ đích (vd 'vi')
    - prefix: prefix cho input (vd "translate English to Vietnamese: ")
    - train_split, val_split, test_split: tỷ lệ chia tập train, validation, test
    Trả về: (DatasetDict với train/validation/test, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    # prefix instruction
    if prefix:
        task_prefix = f"{prefix} {src_lang} to {tgt_lang}: "
    else:
        task_prefix = ""

    # Kiểm tra và chuẩn hóa dataset
    if isinstance(dataset, datasets.DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]  # Nếu là DatasetDict, lấy split train
        else:
            dataset = dataset

    # Chia dataset thành train, validation, test
    train_val = dataset.train_test_split(test_size=(val_split + test_split), shuffle=True, seed=42)
    val_test = train_val["test"].train_test_split(test_size=test_split/(val_split + test_split), shuffle=True, seed=42)
    
    dataset_dict = datasets.DatasetDict({
        "train": train_val["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })

    def preprocess_function(examples):
        inputs = [task_prefix + ex[src_lang] for ex in examples["translation"]]
        targets = [ex[tgt_lang] for ex in examples["translation"]]

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
            padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                truncation=True,
                padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize từng split
    tokenized_dict = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        tokenized_dict[split] = dataset_dict[split].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_dict[split].column_names
        )

    return tokenized_dict, tokenizer


def make_data_collator(tokenizer, model=None):
    """
    Return a suitable DataCollatorForSeq2Seq for T5 training. If model provided,
    pass to DataCollatorForSeq2Seq for label padding.
    """
    from transformers import DataCollatorForSeq2Seq
    return DataCollatorForSeq2Seq(tokenizer, model=model)
