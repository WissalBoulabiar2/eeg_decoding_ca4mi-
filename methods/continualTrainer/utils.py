import os
from copy import deepcopy
import torch

def get_model_path(checkpoint_dir, model_type, sub_id):
    """Generate model file path for given model type and subject ID"""
    filename = f'{model_type}_{sub_id}.pth.tar'
    return os.path.join(checkpoint_dir, filename)
def save_single_model(model_state_dict, checkpoint_dir, model_type, sub_id):
    """Save a single model's state dictionary to disk"""
    model_copy = deepcopy(model_state_dict)
    model_path = get_model_path(checkpoint_dir, model_type, sub_id)
    torch.save({'model_state_dict': model_copy}, model_path)

def save_models(model, discriminator, checkpoint_dir, sub_id):
    """Save both discriminator and main model for given subject"""
    print(f"Saving all models for subject {sub_id + 1}...")

    # Save discriminator model
    save_single_model(
        discriminator.state_dict(), checkpoint_dir, 'discriminator', sub_id)

    # Save main model
    save_single_model(
        model.state_dict(), checkpoint_dir, 'model', sub_id)

    print(f"All models saved successfully for subject {sub_id + 1}")

def loader_size(data_loader):
    return data_loader.dataset.__len__()