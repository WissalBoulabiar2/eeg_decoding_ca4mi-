"""
Entry point for CA4MI-MP (multi-prototype contributions).
The original entry point main/main.py is unchanged and still runs the original CA4MI.

Examples:
    python main/main_mp.py --config ./configs/ca4mi_mp.yml --phase 1
    python main/main_mp.py --config ./configs/ca4mi_mp.yml --phase 3 --set n_modes=2 adaptive_tau=2.0
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from omegaconf import OmegaConf
from trainer_mp import EEGTrainerMP
import utils

# Cumulative phases: each phase adds one contribution on top of the previous one.
PHASES = {
    # Phase 0: single prototype per class + random memory (CA4MI prototype logic, class labels kept)
    0: dict(n_modes=1, memory_selection='random', class_balanced_memory='no',
            adaptive_proto='no', proto_distance='euclidean', proto_contrastive_reg=0.0),
    # Phase 1: multi-prototype anchors
    1: dict(n_modes=3),
    # Phase 2: diversity-aware + class-balanced memory
    2: dict(memory_selection='diversity', class_balanced_memory='yes'),
    # Phase 3: adaptive prototype consistency
    3: dict(adaptive_proto='yes'),
    # Phase 4: prototype uncertainty (mu, Sigma) + Mahalanobis distance
    4: dict(proto_distance='mahalanobis'),
    # Phase 5: contrastive prototype loss (pull + push)
    5: dict(proto_contrastive_reg=0.01),
}


def phase_settings(phase):
    settings = {}
    for p in range(phase + 1):
        settings.update(PHASES[p])
    return settings


def main():
    parser = argparse.ArgumentParser(description='CA4MI-MP: multi-prototype continual adaptation for MI-EEG')
    parser.add_argument('--config', type=str, default='./configs/ca4mi_mp.yml')
    parser.add_argument('--phase', type=int, choices=sorted(PHASES), default=None,
                        help='Apply cumulative phase preset (0 = single-prototype reference)')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='Override any config value, applied after --phase')
    parser.add_argument('--n_runs', type=int)
    parser.add_argument('--n_subjects', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--checkpoint', type=str)
    flags = parser.parse_args()

    args = OmegaConf.load(flags.config)
    if flags.phase is not None:
        args = OmegaConf.merge(args, phase_settings(flags.phase))
        args.phase = flags.phase
    args = OmegaConf.merge(args, OmegaConf.from_dotlist(flags.set))
    for key in ['n_runs', 'n_subjects', 'device', 'checkpoint']:
        if getattr(flags, key) is not None:
            args[key] = getattr(flags, key)
    if flags.phase is not None:
        args.checkpoint = os.path.join(args.checkpoint, f"phase{flags.phase}")

    print("\n" + "=" * 80)
    print("CA4MI-MP CONFIGURATION")
    print("=" * 80)
    for key in ['phase', 'n_modes', 'modes_per_subject', 'max_prototypes', 'memory_selection',
                'class_balanced_memory', 'adaptive_proto', 'proto_distance', 'proto_contrastive_reg',
                'eval_with_current_shared', 'n_subjects', 'n_classes', 'device', 'checkpoint']:
        print(f"{key:25s}: {args.get(key)}")
    print("=" * 80)

    trainer = EEGTrainerMP(args)
    ACC, BWT, F1, AVG_Inference_Time = trainer.run_full_experiment()
    utils.print_time()
    return {'accuracy': ACC, 'backward_transfer': BWT, 'f1_score': F1, 'inference_time': AVG_Inference_Time}


if __name__ == '__main__':
    main()
