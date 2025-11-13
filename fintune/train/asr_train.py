import os
import logging
from typing import Optional, Dict, Tuple
import torch
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
import evaluate
import numpy as np
from ..preprocess.asr_preprocess import make_data_collator

logger = logging.getLogger(__name__)


def evaluate_model(trainer, processor, eval_dataset, max_length: int = 448, num_beams: int = 4, sample_size: int = None):
    """
    Evaluate Whisper model on validation dataset using WER.
    
    Args:
        trainer: HuggingFace Seq2SeqTrainer
        processor: WhisperProcessor để decode
        eval_dataset: Dataset tokenized
        max_length: max length khi generate
        num_beams: beam size
        sample_size: nếu muốn eval nhanh, chỉ lấy N sample
    
    Returns:
        dict: {"wer": float, "preds": [...], "refs": [...]}
    """
    if sample_size:
        eval_dataset = eval_dataset.select(range(min(sample_size, len(eval_dataset))))

    logger.info("Generating predictions for evaluation...")
    preds = []
    refs = []

    for batch_start in range(0, len(eval_dataset), trainer.args.per_device_eval_batch_size):
        batch = eval_dataset[batch_start: batch_start + trainer.args.per_device_eval_batch_size]

        # convert to tensors
        input_features = torch.tensor(np.array(batch["input_features"])).float().to(trainer.model.device)

        # inputs = processor(
        #     audio=batch["audio_arrays"],
        #     sampling_rate=processor.feature_extractor.sampling_rate,
        #     padding="longest",
        #     return_attention_mask=True,
        #     truncation=False
        # )
        # input_features = torch.tensor(inputs.input_features).float().to(trainer.model.device)
        outputs = trainer.model.generate(
            input_features,
            max_length=max_length,
            task="transcribe"
        )

        decoded_preds = processor.batch_decode(outputs, skip_special_tokens=True)
        print("outputs:", outputs)
        print("Decoded Predictions:", decoded_preds)

        # clean labels (-100 -> pad_token_id)
        labels = np.array(batch["labels"])
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = [l.strip().lower() for l in processor.batch_decode(labels, skip_special_tokens=True)]

        preds.extend(decoded_preds)
        refs.extend(decoded_labels)

    # compute WER
    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(predictions=preds, references=refs)

    metrics = {
        "wer": wer_score,
        "preds": preds[:5],
        "refs": refs[:5]
    }

    logger.info(f"Evaluation results: WER={wer_score:.2f}")
    return metrics




def finetune_whisper(
    model_name_or_path: str,
    tokenized_datasets: Dict,
    processor: WhisperProcessor,
    output_dir: str,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    logging_steps: int = 25,
    fp16: bool = True,
    use_peft: bool = False,
    peft_config: Optional[Dict] = None,
    load_in_8bit: bool = True,
    report_to: str = "tensorboard",
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True
) -> Tuple[WhisperForConditionalGeneration, Dict]:
    """
    Train (or finetune) a Whisper model with optimizations.
    Returns (model, metrics_summary).
    """

    model_args = {}
    if load_in_8bit:
        model_args.update({"load_in_8bit": True, "device_map": "auto"})

    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, **model_args)
    except Exception as e:
        logger.warning(f"Không thể load model từ {model_name_or_path}: {e}")
        logger.info("Sử dụng model gốc 'openai/turbo'")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo", **model_args)
    
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if use_peft:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if load_in_8bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=peft_config.get("r", 16),
            lora_alpha=peft_config.get("alpha", 32),
            target_modules=peft_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "out_proj"]),
            lora_dropout=peft_config.get("dropout", 0.05),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None
        model.generation_config.language = None
        logger.info("PEFT/LoRA enabled")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16 and torch.cuda.is_available(),
        save_safetensors=True,
        predict_with_generate=True,
        logging_steps=logging_steps,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        remove_unused_columns=False,  
        report_to=[report_to],
        dataloader_num_workers=4,
        generation_max_length=448,
        generation_num_beams=4
    )

    data_collator = make_data_collator(processor, model.config.decoder_start_token_id)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir, safe_serialization=False)

    metrics = {"train_loss": train_result.training_loss}
    metrics = {}
    if tokenized_datasets.get("test") is not None:
        metrics.update(evaluate_model(trainer, processor, tokenized_datasets["test"]))


    return model, metrics