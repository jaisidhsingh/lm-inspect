"""
1. Benchmark Jonas's recurrent-depth transformer on GSM-8K with variable no. of recurrence steps.
2. Plot changes in logit distribution as the no. of recurrence steps vary (KLD between 32 step output and less).
"""
import os
import torch
import warnings
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, MambaForCausalLM, AutoModelForCausalLM
warnings.simplefilter("ignore")


FAST_ROOT = os.getenv("FAST_ROOT")
def hf_model_path(model_id):
    return os.path.join(FAST_ROOT, "hf_cache", "models", model_id)

def hf_data_path(data_id):
    return os.path.join(FAST_ROOT, "hf_cache", "data", data_id)

def kldiv(logits_1, logits_2):
    # Convert logits to probabilities using softmax along the vocabulary dimension
    probs_1 = F.softmax(logits_1, dim=-1)
    probs_2 = F.softmax(logits_2, dim=-1)
    
    # Calculate KL divergence: KL(p||q) = Σ p(x) * log(p(x)/q(x))
    # Using PyTorch's KLDiv which expects log probabilities for the second argument
    kl_div = F.kl_div(
        input=probs_2.log(),  # Target distribution in log space
        target=probs_1,       # Source distribution
        reduction='batchmean',  # Average over batch dimension
        log_target=False      # Target is not in log space
    )
    
    return kl_div.item()

@torch.no_grad()
def recurrent_layer_sim():
    example = "If A has 8 apples and then gets 6 more from B, how many apples does A have?"
    model = AutoModelForCausalLM.from_pretrained(hf_model_path("huginn-0125"), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path("huginn-0125"))

    x = tokenizer(example, return_tensors="pt", add_special_tokens=True)["input_ids"]
    device = "cuda"
    model = model.to(device)
    model.eval()
    x = x.to(device)

    reference = model(x, num_steps=64).logits
    kldivs = []
    step_varies_as = [1, 2, 4, 8, 12, 16, 24, 32, 48]

    for count in step_varies_as:
        output = model(x, num_steps=count).logits
        kldivs.append(kldiv(output, reference))

    plt.plot(step_varies_as, kldivs, marker="x")
    plt.xticks(step_varies_as)
    plt.xlabel("No. of recurrence steps")
    plt.ylabel("KL-Div with num_steps=64")
    plt.savefig("kldiv_0.png")


if __name__ == "__main__":
    recurrent_layer_sim()
