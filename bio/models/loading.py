"""Reject partially initialized pretrained encoders while allowing unused heads."""
import torch


def verify_backbone(model, state, prefixes):
    missing, mismatched = [], []
    parameters = dict(model.named_parameters())
    for key, value in parameters.items():
        if not key.startswith(tuple(prefixes)):
            continue
        if key not in state:
            missing.append(key)
        elif state[key].shape != value.shape:
            mismatched.append(key)
        elif not torch.equal(value.detach().cpu(), state[key].detach().cpu()):
            mismatched.append(key)
    report = dict(missing_keys=missing, mismatched_keys=mismatched,
                  checked_parameters=sum(k.startswith(tuple(prefixes)) for k in parameters))
    if report['checked_parameters'] == 0 or missing or mismatched:
        raise ValueError(f'Pretrained backbone is incomplete: {report}')
    model._cellbench_loading_report = report
    return report
