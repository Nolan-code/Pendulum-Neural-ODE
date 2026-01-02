from .mlp_vectorfield import VectorFieldMLP
from .hnn import HNN
from .lnn import LNN

MODEL_REGISTRY = {
    "mlp": VectorFieldMLP,
    "hnn": HNN,
    "lnn": LNN,
}

def build_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](**kwargs)
