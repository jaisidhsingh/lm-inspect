import torch
from datasets import load_dataset
from transformers import AutoTokenizer, MambaForCausalLM


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MambaForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def collect_layerwise_outputs(model, input_seq):
    num_layers = len(model.backbone.layers)
    layerwise_outputs = {}

    x = input_seq
    for i in range(num_layers):
        x = model.backbone.layers[i](x)
        layerwise_outputs[i] = x
    
def tokenwise_cosine_sim(data):
    pass
