import os 
import logging 
from typing import Optional, Dict, Tuple 
import torch 
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq) 
import evaluate 
import numpy as np
from data_utils import make_data_collator

logger = logging.getLogger(__name__)

def evaluate_model(trainer, tokenizer, eval_dataset, max_length: int = 128, num_beams: int = 4, sample_size: int = None):
    """
    Evaluate seq2seq model on validation dataset.
    
    Args:
        trainer: HuggingFace Seq2SeqTrainer
        tokenizer: tokenizer để decode
        eval_dataset: Dataset tokenized
        max_length: max length khi generate
        num_beams: beam size
        sample_size: nếu muốn eval nhanh, chỉ lấy N sample
    
    Returns:
        dict: {"sacrebleu": float, "chrf": float, "preds": [...], "refs": [...]}
    """
    logger = logging.getLogger(__name__)

    
    if sample_size:
        eval_dataset = eval_dataset.select(range(min(sample_size, len(eval_dataset))))

    
    logger.info("Generating predictions for evaluation...")
    preds = []
    refs = []

    for batch_start in range(0, len(eval_dataset), trainer.args.per_device_eval_batch_size):
        batch = eval_dataset[batch_start: batch_start + trainer.args.per_device_eval_batch_size]

        # convert to tensors
        input_ids = torch.tensor(batch["input_ids"]).to(trainer.model.device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(trainer.model.device)

        outputs = trainer.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams
        )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # clean labels (-100 -> pad_token_id)
        labels = np.array(batch["labels"])
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds.extend(decoded_preds)
        refs.extend([[r] for r in decoded_labels])  

    # compute metrics
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    bleu_score = sacrebleu.compute(predictions=preds, references=refs)["score"]
    chrf_score = chrf.compute(predictions=preds, references=refs)["score"]

    metrics = {
        "sacrebleu": bleu_score,
        "chrf": chrf_score,
        "preds": preds[:5],  
        "refs": refs[:5]
    }

    logger.info(f"Evaluation results: BLEU={bleu_score:.2f}, chrF={chrf_score:.2f}")
    return metrics

def finetune_t5(
    model_name_or_path: str,
    tokenized_datasets,
    tokenizer,
    output_dir: str,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    num_train_epochs: int = 1,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    logging_steps: int = 10,
    fp16: bool = True,
    use_peft: bool = False,
    peft_config: Optional[Dict] = None,
    load_in_8bit: bool = False,
    report_to: str = "tensorboard"
) -> Tuple[str, Dict]:
    """
    Train (or finetune) a T5 model.
    Returns (model, metrics_summary).
    """

    model_args = {}
    if load_in_8bit:
        model_args.update({"load_in_8bit": True, "device_map": "auto"})

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_args)

    if use_peft:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if load_in_8bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=peft_config.get("r", 8),
            lora_alpha=peft_config.get("alpha", 32),
            target_modules=peft_config.get("target_modules", ["q", "v"]),
            lora_dropout=peft_config.get("dropout", 0.1),
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)
        logger.info("PEFT/LoRA enabled")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16 and torch.cuda.is_available(),
        save_safetensors=False,
        predict_with_generate=True,
        logging_steps=logging_steps,
        remove_unused_columns=True,
        report_to=[report_to],
    )

    data_collator = make_data_collator(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    # trainer.save_model(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)

    metrics = {"train_loss": train_result.training_loss}
    # if tokenized_datasets.get("test") is not None:
    #     metrics.update(evaluate_model(trainer, tokenizer, tokenized_datasets["test"]))

    return model, metrics
