import sys
sys.path.insert(0,"fintune//utils")
from data_utils import load_dataset_from_files, validate_dataset, prepare_for_t5_translation
from train_utils import finetune_t5
from zenml import step, pipeline, log_metadata
from zenml.logger import get_logger
from typing import Tuple, Any
from typing_extensions import Annotated
from zenml.types import HTMLString
import matplotlib.pyplot as plt
import base64
import io

logger = get_logger(__name__)

@step
def read_data(dataset_path: str) -> Annotated[Any, "data"]:
    logger.info(f"Reading data from {dataset_path}")
    data = load_dataset_from_files(dataset_path)
    logger.info(f"Loaded {len(data)} samples")
    info = validate_dataset(data)
    logger.info(f"Dataset info:\n{info}")

    # Log metadata for dataset artifact
    log_metadata({
        "n_records": info["n_records"],
        "null_src": info["null_src"],
        "null_tgt": info["null_tgt"],
        "avg_len_src": info["avg_len_src"],
        "avg_len_tgt": info["avg_len_tgt"]
    })

    return data

@step
def prepare_data(data, tokenizer_name: str = "t5-small", src_lang: str = "en", tgt_lang: str = "fr") -> Tuple[Annotated[Any, "tokenized"], Annotated[Any, "tokenizer"], Annotated[HTMLString, "viz"]]:
    logger.info(f"Preparing data with tokenizer {tokenizer_name}")
    tokenized, tokenizer = prepare_for_t5_translation(data, tokenizer_name, src_lang=src_lang, tgt_lang=tgt_lang)
    
    # Log metadata for tokenized artifact
    log_metadata({
        "tokenizer_name": tokenizer_name,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "train_size": len(tokenized["train"]),
        "validation_size": len(tokenized["validation"]),
        "test_size": len(tokenized["test"]),
        "max_source_length": 128,  
        "max_target_length": 128
    })
    
    # Visualize 
    split_sizes = {
        "train": len(tokenized["train"]),
        "validation": len(tokenized["validation"]),
        "test": len(tokenized["test"])
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(split_sizes.values(), labels=split_sizes.keys(), autopct='%1.1f%%')
    ax.set_title("Dataset Split Proportions")
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Create HTML with embedded image
    html_viz = f'''
    <div style="text-align: center;">
        <h3>Dataset Splits Visualization</h3>
        <img src="data:image/png;base64,{image_base64}" 
             style="max-width: 100%; height: auto;">
    </div>
    '''
    
    viz = HTMLString(html_viz)
    
    return tokenized, tokenizer, viz

@step
def train_model(model_id: str, tokenized, tokenizer, output_dir: str) -> Tuple[Annotated[Any, "model"], Annotated[Any, "metrics"], Annotated[HTMLString, "viz"]]:
    logger.info(f"Training model {model_id}")
    tokenized = tokenized  
    model, metrics = finetune_t5(
        model_name_or_path=model_id,
        tokenized_datasets=tokenized,
        tokenizer=tokenizer,
        output_dir=output_dir,
        num_train_epochs=1
    )
    logger.info(f"Training completed with metrics: {metrics}")
    
    # Log metadata for model and metrics artifacts
    log_metadata({
        "model_id": model_id,
        "output_dir": output_dir,
        "train_loss": metrics.get("train_loss", 0.0),
    })
    
    # Visualize 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Train Loss"], [metrics.get("train_loss", 0.0)])
    ax.set_title("Training Loss")
    ax.set_ylabel("Loss")
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Create HTML with embedded image
    html_viz = f'''
    <div style="text-align: center;">
        <h3>Training Metrics Visualization</h3>
        <img src="data:image/png;base64,{image_base64}" 
             style="max-width: 100%; height: auto;">
    </div>
    '''
    
    viz = HTMLString(html_viz)
    
    return model, metrics, viz

@pipeline
def t5_finetune_pipeline(model_id: str = "storage/models/translate/hf_models/t5-small", dataset_path: str = "fintune/data/test_data.json", output_dir: str = "fintune/outputs/checkpoints"):
    data = read_data(dataset_path)
    tokenized, tokenizer, viz = prepare_data(data, model_id)
    model, metrics, viz = train_model(model_id, tokenized, tokenizer, output_dir)
    

if __name__ == "__main__":
    result = t5_finetune_pipeline()