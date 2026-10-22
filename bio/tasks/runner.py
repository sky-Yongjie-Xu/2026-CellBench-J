"""Dispatch configured tasks without knowing model-specific preprocessing."""
TASK_METHODS = {
    'embedding_extraction': 'embedding_extraction',
    'clustering': 'no_finetuning_cluster',
    'correction': 'no_finetuning_correction',
    'reconstruction': 'adapter_finetuning_reconstruction',
    'perturbation': 'adapter_finetuning_perturbation',
}


def run_model(cfgs, training='adapter_finetuning'):
    from bio.models import get_model
    task = cfgs['model_cfg']['tasks']
    if task == 'annotation':
        if training not in ('adapter_finetuning', 'full_finetuning'):
            raise ValueError(f'Unknown training phase: {training}')
        method = training + '_annotation'
    elif task in TASK_METHODS:
        method = TASK_METHODS[task]
    else:
        raise ValueError(f'Unknown task: {task}')
    model = get_model(cfgs['model_cfg']['model'])(cfgs, training)
    return getattr(model, method)(cfgs)


def run_vcc_eval(cfgs):
    import pickle
    from bio.io.artifacts import prediction_directory
    from bio.metrics import METRIC_VERSION
    from bio.metrics.vcc import evaluate_vcc
    from bio.io.results import log_metrics
    filename = f"{cfgs['data_cfg']['dataset_name']}_{cfgs['model_cfg']['model']}_seed{cfgs['seed']}_topk{cfgs['topk']}.pkl"
    path = prediction_directory(cfgs) / filename
    if not path.is_file():
        raise FileNotFoundError(f'Run perturbation prediction with metric v2 first: {path}')
    with path.open('rb') as handle:
        data = pickle.load(handle)
    if data.get('metric_version') != METRIC_VERSION:
        raise ValueError('Prediction artifact has an incompatible metric protocol')
    metrics = evaluate_vcc(data['recon'], data['truth'],
                           cfgs['data_cfg'].get('evaluation_expression_scale', 'dataset_defined'),
                           num_threads=cfgs.get('evaluation_cfg', {}).get('vcc_threads', 1))
    metrics['metric_version'] = METRIC_VERSION
    log_metrics(cfgs, metrics, task='perturbation_vcc')
    return metrics
