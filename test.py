import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, MambaForCausalLM, AutoModelForCausalLM
import warnings
warnings.simplefilter("ignore")


FAST_ROOT = os.getenv("FAST_ROOT")
def hf_model_path(model_id):
    return os.path.join(FAST_ROOT, "hf_cache", model_id)

@torch.no_grad()
def mamba_layer_sim(example):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path("mamba-130m-hf"))
    model = MambaForCausalLM.from_pretrained(hf_model_path("mamba-130m-hf"))

    x = tokenizer(example, return_tensors="pt")["input_ids"]

    device = "cuda"
    model = model.to(device)
    x = x.to(device)

    # pass through lookup table
    x = model.backbone.embeddings(x)
    layer_outputs = []

    for i in range(24):
        x = model.backbone.layers[i](x)
        layer_outputs.append(x)

    layer_outputs = torch.stack(layer_outputs, dim=0).view(24, -1)
    layer_outputs /= layer_outputs.norm(dim=-1, keepdim=True)

    sim = (layer_outputs @ layer_outputs.T).cpu().numpy()
    return sim

@torch.no_grad()
def llm_layer_sim(example):
    model = AutoModelForCausalLM.from_pretrained(hf_model_path("gpt2"))
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path("gpt2"))
    
    x = tokenizer(example, return_tensors="pt")["input_ids"]

    device = "cuda"
    model = model.to(device)
    x = x.to(device)

    # pass through lookup table
    position_ids = torch.arange(0, x.shape[-1], dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)

    x = model.transformer.wte(x) + model.transformer.wpe(position_ids)
    layer_outputs = []

    for i in range(12):
        x = model.transformer.h[i](x)[0]
        layer_outputs.append(x)

    layer_outputs = torch.stack(layer_outputs, dim=0).view(12, -1)
    layer_outputs /= layer_outputs.norm(dim=-1, keepdim=True)

    sim = (layer_outputs @ layer_outputs.T).cpu().numpy()
    return sim

@torch.no_grad()
def recurrent_layer_sim(example):
    model = AutoModelForCausalLM.from_pretrained(hf_model_path("huginn-0125"))
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path("huginn-0125"))

    x = tokenizer(example, return_tensors="pt")["input_ids"]
    device = "cuda"
    model = model.to(device)
    x = x.to(device)

    freqs_cis = model.freqs_cis[:, : x.shape[1]]
    x = model.transformer.wte(x)
    # pass through P
    for idx, block in enumerate(model.transformer.prelude, start=1):
        x, _ = block(x, freqs_cis, idx, None, None, False)

    def initialise_state(e, model):
        u = torch.randn_like(e)
        std = model.config.init_values["std"]
        torch.nn.init.trunc_normal_(u, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if model.emb_scale != 1:
            u = u * model.emb_scale
        return u

    s = initialise_state(x, model) 

    layer_outputs = []
    for step in range(4):
        s = model.transformer.adapter(torch.cat([s, x], dim=-1))
        for ridx, block in enumerate(model.transformer.core_block, start=1):
            s, _ = block(s, freqs_cis, idx + ridx, None, None, False) 
            layer_outputs.append(s)

    layer_outputs = torch.stack(layer_outputs, dim=0).view(4 * 4, -1)
    layer_outputs /= layer_outputs.norm(dim=-1, keepdim=True)

    sim = (layer_outputs @ layer_outputs.T).cpu().numpy()
    plt.imshow(sim)
    plt.xticks([i for i in range(16)])
    plt.yticks([i for i in range(16)])
    plt.title("Cosine similarity between outputs of the 4-layers recurrent block\nover 4 recurrent steps.")
    plt.xlabel("Layer index")
    plt.ylabel("Layer index")
    plt.savefig("recurrent_layer_sims.png")

def plot_sims(sim1, sim2):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(sim1)
    axes[0].set_title("Mamba 130M")
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Layer index")
    axes[0].set_xticks([i for i in range(0, 24, 6)])
    axes[0].set_yticks([i for i in range(0, 24, 6)])
    
    axes[1].imshow(sim2)
    axes[1].set_title("GPT2 120M")
    axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel("Layer index")
    axes[1].set_xticks([i for i in range(0, 12, 3)])
    axes[1].set_yticks([i for i in range(0, 12, 3)])

    plt.suptitle("Cosine similarities between layer outputs")
    plt.tight_layout()
    plt.savefig("mamba_vs_gpt2_layer_sims.png")


if __name__ == "__main__":
    example = "This is an example sentence."
    recurrent_layer_sim(example)
    sim1 = mamba_layer_sim(example)
    sim2 = llm_layer_sim(example)
    plot_sims(sim1, sim2)
