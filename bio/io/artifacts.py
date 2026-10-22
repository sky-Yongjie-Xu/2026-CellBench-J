"""Versioned prediction artifact locations."""
from bio.metrics import METRIC_VERSION

def prediction_directory(cfgs):
    """Do not reuse predictions trained with an older head/loss or metric protocol."""
    import hashlib
    import json
    from bio.io.embeddings import fingerprint
    from bio.io.paths import output_path
    signature, _ = fingerprint(cfgs)
    protocol = dict(signature=signature, metric_version=METRIC_VERSION,
                    topk=cfgs.get('topk', 1), evaluation=cfgs.get('evaluation_cfg', {}))
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:24]
    return output_path(cfgs, 'perturbation_results', 'v2', key)
