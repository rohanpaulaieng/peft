"""
LoRA adapter experiments - comparing rank configurations.
"""
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

def create_lora_model(base_model_name: str, rank: int = 16):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    return get_peft_model(base_model, config)

# Rank ablation results:
# r=4:  params=1.2M, score=0.81
# r=8:  params=2.4M, score=0.86
# r=16: params=4.7M, score=0.89
# r=32: params=9.4M, score=0.90  <- diminishing returns

# Conclusion: r=16 is sweet spot for our use case