"""Command line entry point; importing this module does not start an experiment."""
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description='CellBench single-cell representation benchmark')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--cfgs', help='Model YAML; omit in all-task mode to run all registered models')
    parser.add_argument('--phase', default='adapter_finetuning', choices=['adapter_finetuning', 'full_finetuning'])
    parser.add_argument('--task', choices=['embedding_extraction', 'clustering', 'annotation', 'reconstruction', 'perturbation', 'correction', 'all'])
    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--seed', default=41, type=int)
    parser.add_argument('--hvg', default=None, type=int)
    parser.add_argument('--dataset_name', default=None)
    parser.add_argument('--dataset', help='Run all applicable tasks for this dataset unless --task is specified')
    parser.add_argument('--dry-run', action='store_true', help='Preview the all-task plan without running models')
    parser.add_argument('--fail-fast', action='store_true', help='Stop the suite after its first failed task')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--log_to_file', action='store_true', help='Retained for existing command files')
    parser.add_argument('--iter', default=0, help='Retained for existing command files')
    parser.add_argument('--vcc-eval', action='store_true', help='Evaluate saved perturbation predictions')
    return parser


def main(argv=None):
    parser = build_parser()
    opt = parser.parse_args(argv)
    if opt.dataset:
        if opt.dataset_name and opt.dataset_name != opt.dataset:
            parser.error('--dataset and --dataset_name disagree')
        opt.dataset_name = opt.dataset
        opt.task = opt.task or 'all'
    if opt.task != 'all' and (opt.dry_run or opt.fail_fast):
        parser.error('--dry-run and --fail-fast require --task all or --dataset')
    # Set the GPU selector before importing torch or any adapter.
    if opt.gpu is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    from bio.config import config_loader
    if opt.task == 'all':
        if opt.vcc_eval:
            parser.error('--vcc-eval is for single-task evaluation; the suite schedules VCC automatically')
        from bio.runtime.suite import run_suite
        from bio.io.paths import project_path
        if opt.cfgs:
            opt.cfgs = str(project_path(opt.cfgs))
        try:
            return run_suite(opt)
        except ValueError as exc:
            parser.error(str(exc))
    opt.cfgs = opt.cfgs or 'configs/scmulan.yaml'
    cfgs = config_loader(opt.cfgs)
    from bio.runtime.seeds import init_seeds
    from bio.io.results import start_run
    from bio.tasks.runner import run_model, run_vcc_eval
    cfgs.update(topk=opt.topk, seed=opt.seed, phase=opt.phase)
    if opt.task is not None:
        cfgs['model_cfg']['tasks'] = opt.task
    if opt.dataset_name is not None:
        cfgs['data_cfg']['dataset_name'] = opt.dataset_name
    if opt.hvg is not None:
        cfgs['data_cfg']['hvg'] = opt.hvg
    if opt.vcc_eval and cfgs['model_cfg']['tasks'] != 'perturbation':
        parser.error('--vcc-eval requires --task perturbation')
    start_run(cfgs)
    init_seeds(cfgs['seed'])
    if opt.vcc_eval:
        run_vcc_eval(cfgs)
    else:
        run_model(cfgs, opt.phase)
    return 0
