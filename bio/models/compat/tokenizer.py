"""Select the dictionaries matching the original 30M Geneformer/LangCell models."""
import inspect
from bio.io.paths import model_resource


def tokenizer_options(model_cfg, tokenizer_base):
    model = model_cfg.get('model', 'Geneformer')
    options = {
        "token_dictionary_file": model_resource(model_cfg, model, 'token_dictionary_gc30M.pkl', 'token_dictionary_file'),
        "gene_median_file": model_resource(model_cfg, model, 'gene_median_dictionary_gc30M.pkl', 'gene_median_file'),
    }
    # New Geneformer releases default to V2; these checkpoints require V1 encoding.
    parameters = inspect.signature(tokenizer_base.__init__).parameters
    if "model_version" in parameters:
        options["model_version"] = "V1"
    if "model_input_size" in parameters:
        options["model_input_size"] = 2048
    if "special_token" in parameters:
        options["special_token"] = False
    return options
