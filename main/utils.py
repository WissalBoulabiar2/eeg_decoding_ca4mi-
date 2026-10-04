import os
import numpy as np
from copy import deepcopy
import pickle
import time
import uuid
from subprocess import call
import torch

########################################################################################################################
# File and Checkpoint Management
########################################################################################################################
def copy_project_files_to_destination(args):
    current_working_directory = os.getcwd()
    destination_diretory = os.path.join(args.checkpoint, 'code') + '/'
    if not os.path.exists(destination_diretory):
        os.mkdir(destination_diretory)

    def get_full_folder_path(folder_name):
        return os.path.join(current_working_directory, folder_name)

    folders = [get_full_folder_path(item) for item in
               ['dataloaders', 'networks', 'methods', 'configs', 'main', 'continualTrainer.py']]
    for folder in folders:
        call('cp -rf {} {}'.format(folder, destination_diretory), shell=True)

def create_checkpoint_structure(args):
    import time
    timestamp = int(time.time())
    uid = f"{timestamp}_{uuid.uuid4().hex[:8]}"

    if args.checkpoint is None:
        base_dir = '../checkpoints'
        os.makedirs(base_dir, exist_ok=True)
        args.checkpoint = os.path.join(base_dir, uid)
    else:
        os.makedirs(args.checkpoint, exist_ok=True)
        args.checkpoint = os.path.join(args.checkpoint, uid)

    os.makedirs(args.checkpoint, exist_ok=True)
########################################################################################################################
# Utility Functions
########################################################################################################################
def print_time():
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Finished at: ", dt_string)


def report_tr_baseline(res, e, sbatch, clock0, clock1):
    # Training performance
    print(
        '| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}%|'.format(
            e + 1,
            1000 * sbatch * (clock1 - clock0) / res['size'],
            1000 * sbatch * (time.time() - clock1) / res['size'],
            res['loss_t'], res['acc_t']), end='')

def report_val_baseline(res):
    # Validation performance
    print(
        ' Valid loss={:.6f}, acc={:5.2f}% |'.format(
            res['loss_t'], res['acc_t']), end='')
########################################################################################################################
# Logging Functions
########################################################################################################################
def print_log_acc_bwt_f1_inference_time(acc, lss, f1_mat,inf_time_mat, output_path, run_id):
    """
    Print and log:
      - Accuracy matrix and average ACC
      - Backward transfer (BWT)
      - Macro-F1 matrix and average macro-F1
      - Per-class F1 for the final subject
      - Kappa matrix and average Kappa
    """
    # --- Accuracy matrix ---
    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end=',')
        for j in range(acc.shape[1]):
            print(f'{acc[i, j]:5.4f}% ', end=',')
        print()

    avg_acc = np.mean(acc[-1, :])
    print(f'ACC: {avg_acc:5.4f}%')
    print()

    gem_bwt = np.sum(acc[-1, :] - np.diag(acc)) / (acc.shape[1] - 1)
    print(f'BWT: {gem_bwt:5.2f}%')
    print('*' * 100)
    print('Done!')
    print()

    # --- Macro-F1 matrix ---
    print('F1 (macro) =')
    for i in range(f1_mat.shape[0]):
        print('\t', end=',')
        for j in range(f1_mat.shape[1]):
            print(f'{f1_mat[i, j]:5.4f}% ', end=',')
        print()

    avg_f1 = np.mean(f1_mat[-1, :])
    print(f'F1 (macro avg): {avg_f1:5.4f}%')
    print()

    # --- Assemble and save logs ---
    logs = {
        'name': output_path,
        'acc': acc,
        'loss': lss,
        'f1_macro': f1_mat,
        'inf_time_mat': inf_time_mat,
        'gem_bwt': gem_bwt,
        'ucb_bwt': (acc[-1, :] - np.diag(acc)).mean(),
        'rii': np.diag(acc),
        'rij': acc[-1],
        'f1_avg': avg_f1
    }

    if output_path.endswith('.txt'):
        log_dir = os.path.dirname(output_path)
    else:
        log_dir = output_path

    os.makedirs(log_dir, exist_ok=True)

    path = os.path.join(log_dir, 'logs_run_id_{}.p'.format(run_id))
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    print("Log file saved in ", path)

    return avg_acc, gem_bwt, avg_f1, inf_time_mat


def print_log_acc_bwt_f1(task_class, acc, lss, f1, inf_time_mat, output_path, run_id):
    """
    Enhanced function to print and log accuracy, F1-score
    BWT is only calculated for accuracy (following original implementation).
    """
    print('*' * 100)
    print('=== ACCURACY RESULTS ===')
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i, j]), end=',')
        print()

    avg_acc = np.mean(acc[acc.shape[0] - 1, :])
    print('ACC: {:5.4f}%'.format(avg_acc))
    print()

    # BWT calculated based on GEM paper - ONLY for accuracy
    if len(acc[-1]) > 1:
        gem_bwt = sum(acc[-1] - np.diag(acc)) / (len(acc[-1]) - 1)
        ucb_bwt = (acc[-1] - np.diag(acc)).mean()
        print('BWT: {:5.2f}%'.format(gem_bwt))
    else:
        gem_bwt = 0.0
        ucb_bwt = 0.0
        print('BWT: N/A (only one task)')
    print()

    print('=== F1-SCORE RESULTS ===')
    print('F1-scores =')
    for i in range(f1.shape[0]):
        print('\t', end=',')
        for j in range(f1.shape[1]):
            print('{:5.4f}% '.format(f1[i, j]), end=',')
        print()

    avg_f1 = np.mean(f1[f1.shape[0] - 1, :])
    print('F1: {:5.4f}%'.format(avg_f1))
    print()


    print('=== SUMMARY ===')
    print('Average Accuracy:  {:5.4f}%'.format(avg_acc))
    print('Average F1-score:  {:5.4f}%'.format(avg_f1))
    if len(acc[-1]) > 1:
        print('BWT (Accuracy):    {:5.2f}%'.format(gem_bwt))
    else:
        print('BWT (Accuracy):    N/A (only one task)')

    print('*' * 100)
    print('Done!')

    # Save comprehensive logs
    logs = {}
    logs['name'] = output_path
    logs['task_class'] = task_class
    logs['acc'] = acc
    logs['loss'] = lss
    logs['f1'] = f1
    logs['avg_acc'] = avg_acc
    logs['gem_bwt'] = gem_bwt
    logs['ucb_bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]
    logs['avg_f1'] = avg_f1
    logs['rii_f1'] = np.diag(f1)
    logs['rij_f1'] = f1[-1]
    logs['inf_time_mat']: inf_time_mat
    path = os.path.join(output_path, 'logs_run_id_{}.p'.format(run_id))
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    print("Log file saved in ", path)

    return {
        'avg_acc': avg_acc, 'bwt': gem_bwt,
        'avg_f1': avg_f1, 'inf_time_mat': inf_time_mat
    }

def print_running_acc_bwt_f1(acc, f1, task_num):
    """
    Enhanced function to print running metrics.
    BWT is only calculated for accuracy (following original approach).
    """
    print()
    acc = acc[:task_num + 1, :task_num + 1]
    f1 = f1[:task_num + 1, :task_num + 1]

    avg_acc = np.mean(acc[acc.shape[0] - 1, :])
    avg_f1 = np.mean(f1[f1.shape[0] - 1, :])

    print('ACC: {:5.4f}% | F1: {:5.4f}% '.format(avg_acc, avg_f1))

    if len(acc[-1]) > 1:
        gem_bwt = sum(acc[-1] - np.diag(acc)) / (len(acc[-1]) - 1)
        print('BWT: {:5.2f}%'.format(gem_bwt))
    else:
        print('BWT: N/A (only one task)')
    print()


def get_model(model):
    return deepcopy(model.state_dict())

def get_model_path(checkpoint_dir, model_type, sub_id):
    """Generate model file path for given model type and subject ID"""
    filename = f'{model_type}_{sub_id}.pth.tar'
    return os.path.join(checkpoint_dir, filename)

def load_model_checkpoint(checkpoint_dir, model_type, sub_id):
    """Load model checkpoint from disk with error handling"""
    model_path = get_model_path(checkpoint_dir, model_type, sub_id)
    checkpoint = torch.load(model_path, weights_only=True)
    return checkpoint

def load_current_models(network_class, args, checkpoint_dir, sub_id):
    """Load a complete model from checkpoint for current subject"""
    print(f"Loading checkpoint for subject {sub_id + 1}...")

    # Create new network instance
    net = network_class(args)

    # Load model checkpoint
    checkpoint = load_model_checkpoint(checkpoint_dir, 'model', sub_id)

    # Load state dict and move to device
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.to(args.device)

    # print(f"Model loaded successfully for subject {sub_id + 1}")
    return net


def load_model_with_shared_update(network_class, args, current_model, checkpoint_dir, sub_id):
    """update the current shared module"""
    print(f"Loading updated model ...")

    # Create new network instance
    net = network_class(args)

    # Load previous model checkpoint
    checkpoint = load_model_checkpoint(checkpoint_dir, 'model', sub_id)
    net.load_state_dict(checkpoint['model_state_dict'])

    # Update shared module with current state
    current_shared_module = deepcopy(current_model.shared.state_dict())
    net.shared.load_state_dict(current_shared_module)

    # Move to device
    encoder = net.to(args.device)

    return encoder







