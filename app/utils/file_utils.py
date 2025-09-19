from pathlib import Path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def dir_is_empty(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return True
    return next(p.iterdir(), None) is None

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False