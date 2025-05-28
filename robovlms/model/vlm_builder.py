import copy
import transformers
import torch

from robovlms.utils.model_utils import build_tokenizer


def build_vlm(vlm_config, tokenizer_config, precision="bf16"):
    vlm_config = copy.deepcopy(vlm_config)
    model_path = vlm_config.get("pretrained_model_name_or_path")
    model_name = vlm_config.get("name")
    model_type = vlm_config.get("type", "AutoModel")
    if model_name == "paligemma":
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            revision="bfloat16",
        )
        tokenizer = AutoProcessor.from_pretrained(model_path)
    else:
        # Raise an error or log a warning if a non-paligemma model is requested
        raise ValueError(f"Unsupported model_name: {model_name}. Only 'paligemma' is supported.")

    return tokenizer, model
