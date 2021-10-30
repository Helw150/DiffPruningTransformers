from diffpruningtransformers import __version__
from diffpruningtransformers import DiffPruningTransformer
from transformers import AutoConfig, AutoTokenizer, AutoModel

import torch as pt


def test_version():
    assert __version__ == "0.1.0"


def test_roberta_diff_prune():
    config = AutoConfig.from_pretrained("roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    encodings = tokenizer("This is a test. A what? A test.", return_tensors="pt")
    base_model = AutoModel.from_config(config)
    diff_model = DiffPruningTransformer(base_model, "cpu")
    outputs = diff_model(encodings["input_ids"], encodings["attention_mask"])
    assert pt.isclose(outputs[1], pt.tensor(1.23811448e08))
    base_model.eval()
    diff_model.eval()
    outputs = diff_model(encodings["input_ids"], encodings["attention_mask"])
    base_outputs = base_model(encodings["input_ids"], encodings["attention_mask"])
    assert pt.all(outputs[0] == base_outputs[0])
