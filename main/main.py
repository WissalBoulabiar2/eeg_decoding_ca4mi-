"""
Main script for running EEG continual learning experiments
Continual Adaptation for Motor Imagery EEG Decoding (CA4MI)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import time
from omegaconf import OmegaConf
from trainer import EEGTrainer
import utils




def main():
    """Main function to parse arguments and run the experiment"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Continual Adaptation for Motor Imagery EEG Decoding (CA4MI)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/ca4mi.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--approach',
        type=str,
        choices=['ca4mi', 'ewc', 'der', 'finetuning', 'er', 'mudvi', 'cger'],
        help='Override approach specified in config file'
    )
    parser.add_argument(
        '--n_runs',
        type=int,
        help='Override number of runs specified in config file'
    )
    parser.add_argument(
        '--n_subjects',
        type=int,
        help='Override number of subjects specified in config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Override device specified in config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Override checkpoint directory specified in config file'
    )

    flags = parser.parse_args()

    # Load configuration
    try:
        args = OmegaConf.load(flags.config)
        print(f"Loaded configuration from: {flags.config}")
    except Exception as e:
        print(f"Error loading configuration file {flags.config}: {e}")
        return

    # Override config values with command line arguments if provided
    if flags.approach is not None:
        args.approach = flags.approach
        print(f"Override approach: {args.approach}")

    if flags.n_runs is not None:
        args.n_runs = flags.n_runs
        print(f"Override n_runs: {args.n_runs}")

    if flags.n_subjects is not None:
        args.n_subjects = flags.n_subjects
        print(f"Override n_subjects: {args.n_subjects}")

    if flags.device is not None:
        args.device = flags.device
        print(f"Override device: {args.device}")

    if flags.checkpoint is not None:
        args.checkpoint = flags.checkpoint
        print(f"Override checkpoint: {args.checkpoint}")

    # Print experiment configuration
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Approach:        {args.approach}")
    print(f"Number of runs:  {args.n_runs}")
    print(f"Number of subjects: {args.n_subjects}")
    print(f"Device:          {args.device}")
    print(f"Checkpoint dir:  {args.checkpoint}")
    print(f"Config file:     {flags.config}")
    print("=" * 80)

    # Validate required arguments
    required_args = ['approach', 'n_runs', 'n_subjects', 'device']
    missing_args = [arg for arg in required_args if not hasattr(args, arg) or getattr(args, arg) is None]

    if missing_args:
        print(f"Error: Missing required arguments: {missing_args}")
        print("Please check your configuration file or provide them via command line.")
        return

    # Initialize and run trainer
    try:
        trainer = EEGTrainer(args)
        print(f"\nInitialized EEGTrainer for approach: {args.approach}")

        # Run the complete experiment
        ACC, BWT, F1, AVG_Inference_Time = trainer.run_full_experiment()

        utils.print_time()

        # Return results for potential further processing
        return {
            'accuracy': ACC,
            'backward_transfer': BWT,
            'f1_score': F1,
            'inference_time': AVG_Inference_Time
        }

    except Exception as e:
        print(f"Error during experiment execution: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()