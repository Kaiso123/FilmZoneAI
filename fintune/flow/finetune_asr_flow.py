
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

from ..preprocess.asr_preprocess import (
    load_dataset_from_files_or_hf,
    validate_dataset,
    prepare_for_whisper_asr
)
from ..train.asr_train import finetune_whisper, evaluate_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(args: argparse.Namespace):
    """
    Chạy toàn bộ pipeline training.
    """
    # Bước 1: Load dataset
    logger.info(f"Loading dataset from: {args.dataset_path}")
    dataset = load_dataset_from_files_or_hf(
        path=args.dataset_path,
        split=args.split,
        language=args.language,
        test_mode=args.test_mode
    )

    # Bước 2: Validate dataset
    logger.info("Validating dataset...")
    validation_info = validate_dataset(dataset, sample_n=5)
    # logger.info(f"Validation summary: {validation_info}")
    if validation_info["null_audio"] > 0 or validation_info["null_sentence"] > 0:
        logger.warning("Có samples invalid, sẽ filter trong prepare.")

    # Bước 3: Prepare dataset
    logger.info("Preparing dataset for Whisper...")
    tokenized_datasets, processor = prepare_for_whisper_asr(
        dataset=dataset,
        processor_name_or_path=args.model_name,
        max_audio_length=args.max_audio_length,
        max_target_length=args.max_target_length,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )
    logger.info(f"Dataset prepared: { {k: len(v) for k, v in tokenized_datasets.items()} } samples")

    # Bước 4: Finetune model
    logger.info(f"Finetuning Whisper model: {args.model_name}")
    model, train_metrics = finetune_whisper(
        model_name_or_path=args.model_name,
        tokenized_datasets=tokenized_datasets,
        processor=processor,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        use_peft=args.use_peft,
        peft_config={
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
            "dropout": args.lora_dropout
        } if args.use_peft else None,
        load_in_8bit=args.load_in_8bit,
        report_to=args.report_to,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing
    )
    logger.info(f"Training completed. Metrics: {train_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quản lý training ASR model.")
    
    # Dataset args
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset file or HF dataset name (e.g., 'mozilla-foundation/common_voice_11_0')")
    parser.add_argument("--split", type=str, default=None, help="Split name (e.g., 'train')")
    parser.add_argument("--language", type=str, default="en", help="Language code for HF dataset")
    parser.add_argument("--test_mode", action="store_true", default=True, help="test flow with small subset")

    # Model args
    parser.add_argument("--model_name", type=str, default=".\\storage\\models\\transcribe\\turbo", help="Pretrained Whisper model name/path")
    parser.add_argument("--output_dir", type=str, default="./whisper_output", help="Output directory for trained model")
    
    # Prepare args
    parser.add_argument("--max_audio_length", type=int, default=30 * 16000, help="Max audio length in samples")
    parser.add_argument("--max_target_length", type=int, default=448, help="Max target token length")
    parser.add_argument("--train_split", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Per device eval batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging steps")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16")
    parser.add_argument("--use_peft", action="store_true", default=False, help="Use PEFT/LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Load in 8-bit")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report to (e.g., tensorboard)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--eval_sample_size", type=int, default=None, help="Sample size for eval (for quick test)")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)

    #python -m fintune.flow.finetune_asr_flow --dataset_path openslr/librispeech_asr --language clean --output_dir .\fintune\outputs\asr --split test --test_mode --epochs 1 --batch_size 2 --use_peft