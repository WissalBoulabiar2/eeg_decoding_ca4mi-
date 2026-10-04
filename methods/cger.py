import sys, time, os
from main import utils

print(sys.path)
import copy
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CGER:
    def __init__(self, model, args, network):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.network = network
        self.nepochs = args.nepochs
        self.sbatch = args.batch_size
        self.lr = args.lr
        self.device = args.device
        self.checkpoint = args.checkpoint
        self.e_lr = args.e_lr
        self.task_cls = [[s, args.n_classes] for s in range(args.n_subjects)]
        self.lr_patience = args.lr_patience
        self.lr_factor = args.lr_factor
        self.lr_min = args.lr_min
        self.optimizer_e = self.get_e_optimizer(0)
        self.patience = args.patience
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        self.lamda = 1  # Weight for prototype loss same as the original paper


    def get_prototype_samples(self, loader, task_id):
        """
        Extracts prototype samples from the data loader by computing the mean feature vector for each class.
        """
        self.model.eval()
        all_fea = []
        all_output = []
        all_labels = []

        with torch.no_grad():
            for data, target,_,_ in loader:
                data = data[:, np.newaxis, :, :]
                x = data.to(self.device)
                y = target.to(self.device)
                try:
                    features = self.model.FeatureExtractor(x)
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    if hasattr(self.model.FeatureExtractor.model, 'block'):
                        features = self.model.FeatureExtractor.model.block(x)
                        features = features.view(features.size(0), -1)
                    outputs = self.model(x)
                except Exception as e:
                    features = outputs = self.model(x)
                all_fea.append(features)
                all_output.append(outputs)
                all_labels.append(y)

        all_fea = torch.cat(all_fea)
        all_output = torch.cat(all_output)
        all_labels = torch.cat(all_labels)

        all_output = nn.Softmax(dim=1)(all_output)

        class_prototypes = []
        num_classes = all_output.size(1)

        for i in range(num_classes):
            class_mask = all_labels == i
            if class_mask.sum() > 0:
                class_fea = all_fea[class_mask]
                class_proto = class_fea.mean(dim=0)
                class_prototypes.append(class_proto)
            else:
                print(f"No samples found for class {i}, skipping prototype computation.")

        if len(class_prototypes) > 0:
            prototypes = torch.stack(class_prototypes).to(self.device)
            print(f"Memory updated by adding {len(prototypes)} prototype samples for task {task_id}")
            return prototypes
        else:
            print(f"No prototypes could be computed for task {task_id}")
            return None


    def compute_prototype_loss(self, features, targets, prototypes):
        """
        Computes the prototype loss using cosine similarity with more explicit computation.
        """
        pro_loss = torch.tensor(0.0).to(self.device)

        if prototypes is not None and len(prototypes) > 0:
            features = features.view(features.size(0), -1)
            prototypes = prototypes.view(prototypes.size(0), -1)

            if prototypes.size(1) != features.size(1):
                prototypes = torch.nn.Linear(prototypes.size(1), features.size(1)).to(self.device)(prototypes)

            normalized_features = F.normalize(features, p=2, dim=1)
            normalized_prototypes = F.normalize(prototypes, p=2, dim=1)
            similarities = torch.matmul(normalized_features, normalized_prototypes.t())
            batch_size = features.size(0)
            prototype_mask = torch.zeros_like(similarities, dtype=torch.bool)

            for i in range(batch_size):
                target = targets[i].item()
                if target < len(prototypes):
                    prototype_mask[i, target] = True

            correct_similarities = similarities[prototype_mask]

            if correct_similarities.numel() > 0:
                pro_loss = 1 - correct_similarities.mean()
            else:
                print("No valid similarities found for prototype loss.")

        return pro_loss

    def train(self, dataset, task_id, *args, **kwargs):
        prototypes = kwargs.get('proto_', None)

        best_loss = np.inf
        best_model = deepcopy(self.model.state_dict())
        patience_counter = 0
        e_lr = self.e_lr

        if prototypes is not None:
            print(f"Received {len(prototypes)} prototypes for task {task_id}")

        current_task_prototypes = self.get_prototype_samples(dataset['train'], task_id)

        for e in range(self.nepochs):
            clock0 = time.time()
            combined_prototypes = (prototypes or []) + (
                [current_task_prototypes] if current_task_prototypes is not None else [])

            self.train_epoch(dataset['train'], task_id, combined_prototypes)

            clock1 = time.time()
            train_res = self.eval_(dataset['train'], task_id)
            utils.report_tr_baseline(train_res, e, self.sbatch, clock0, clock1)

            valid_res = self.eval_(dataset['valid'], task_id)
            utils.report_val_baseline(valid_res)

            if valid_res['loss_t'] < best_loss:
                best_loss = valid_res['loss_t']
                best_model = deepcopy(self.model.state_dict())
                patience_counter = 0
                print(' *', end='')
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f'\nEarly stopping at epoch {e + 1} due to no improvement in validation loss.')
                break

            print()

        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(task_id)

        return current_task_prototypes

    def train_epoch(self, train_loader, task_id, prototypes=None):
        self.model.train()
        # for data, target in train_loader:
        for data, target, _ ,_ in train_loader:
            data = data[:, np.newaxis, :, :]
            x = data.to(device=self.device, dtype=torch.float32)
            y = target.to(device=self.device, dtype=torch.long)


            self.optimizer_e.zero_grad()
            self.model.zero_grad()

            output = self.model(x)

            features= self.model.get_encoded_ftrs(x)

            classification_loss = self.ce_loss(output, y)

            cg_loss = torch.tensor(0).to(self.device)
            if prototypes is not None and len(prototypes) > 0:
                combined_prototypes = torch.cat(prototypes)
                cg_loss = self.compute_prototype_loss(features, y, combined_prototypes)

            total_loss = classification_loss + self.lamda * cg_loss

            total_loss.backward()
            self.optimizer_e.step()

        return

    def eval_(self, data_loader, task_id):
        self.model.eval()
        total_loss = 0
        correct = 0
        num = 0
        res = {}
        with torch.no_grad():
            # for data, target in data_loader:
            for data, target, _, _ in data_loader:
                data = data[:, np.newaxis, :, :]
                x = data.to(self.device)
                y = target.to(self.device)

                output = self.model(x)
                loss = self.ce_loss(output, y)
                total_loss += loss.item()

                pred = output.argmax(dim=1)
                correct += pred.eq(y).sum().item()

                num += x.size(0)

        res['loss_t'] = total_loss / num
        res['acc_t'] = 100. * correct / num
        res["size"] = self.loader_size(data_loader)
        return res


    def evaluate(self, test_loader, task_id, model=None):
        """Evaluate model performance with proper error handling"""
        from sklearn.metrics import f1_score, cohen_kappa_score
        import numpy as np

        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        batch_count = 0
        total_time = 0.0
        num_samples = 0

        # Check if test_loader is empty
        if len(test_loader) == 0:
            print(f"Warning: Test loader for subject {task_id + 1} is empty!")
            return {
                'accuracy': 0.0,
                'loss': 0.0,
                'f1_score': 0.0

            }

        with torch.no_grad():
            # for inputs, labels in test_loader:
            for inputs, labels, _ ,_ in test_loader:
                try:
                    inputs = inputs.to(self.device)

                    if len(inputs.shape) == 3:
                        inputs = inputs[:, np.newaxis, :, :]
                    elif len(inputs.shape) == 4 and inputs.shape[1] != 1:
                        pass

                    labels = labels.to(self.device)
                    start_time = time.time()

                    outputs = model(inputs)
                    if isinstance(outputs, dict):
                        if 'logits' in outputs:
                            logits = outputs['logits']
                        elif 'output' in outputs:
                            logits = outputs['output']
                        else:
                            logits = list(outputs.values())[0]
                    else:
                        logits = outputs

                    loss = self.criterion(logits, labels)
                    total_loss += loss.item()
                    end_time = time.time()
                    batch_count += 1

                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    total_time += (end_time - start_time)
                    num_samples += inputs.shape[0]
                    inf_time_mat = total_time / num_samples

                except Exception as e:
                    print(f"Warning: Error processing batch in subject {task_id + 1}: {e}")
                    print(f"Input shape: {inputs.shape if 'inputs' in locals() else 'Unknown'}")
                    print(f"Output type: {type(outputs) if 'outputs' in locals() else 'Unknown'}")
                    if 'outputs' in locals():
                        if isinstance(outputs, dict):
                            print(f"Output keys: {outputs.keys()}")
                        else:
                            print(f"Output shape: {outputs.shape}")
                    continue

        if total == 0 or batch_count == 0:
            print(f"Warning: No data processed for subject {task_id + 1}!")
            return {
                'accuracy': 0.0,
                'loss': 0.0,
                'f1_score': 0.0
            }

        accuracy = 100 * correct / total
        avg_loss = total_loss / batch_count

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        try:
            if len(np.unique(all_labels)) > 1:
                f1 = f1_score(all_labels, all_predictions, average='macro') * 100
                kappa = cohen_kappa_score(all_labels, all_predictions) * 100
            else:
                print(f"Warning: Only one class found in subject {task_id + 1}, setting F1 and Kappa to 0")
                f1 = 0.0
                kappa = 0.0
        except Exception as e:
            print(f"Warning: Error calculating F1/Kappa for subject {task_id + 1}: {e}")
            f1 = 0.0
            kappa = 0.0

        # self.model.train()

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'f1_score': f1,
            'inf_time_mat': inf_time_mat
        }

    def save_model(self, task_id):
        print("Saving all models for task {} ...".format(task_id + 1))
        model = utils.get_model(self.model)
        torch.save({'model_state_dict': model,
                    }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))

    def load_model(self, task_id):
        model = self.network.Net(self.args)
        checkpoint = torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))
        model.load_state_dict(checkpoint['model_state_dict'])
        current_feature_extractor = deepcopy(self.model.FeatureExtractor.state_dict())
        model.FeatureExtractor.load_state_dict(current_feature_extractor)
        model = model.to(self.args.device)
        return model

    def loader_size(self, data_loader):
        return data_loader.dataset.__len__()

    def get_e_optimizer(self, task_id, e_lr=None):
        """
        Get optimizer for the model

        Args:
            task_id: Current task ID
            e_lr: Learning rate (optional)

        Returns:
            Optimizer for the model
        """
        if e_lr is None:
            e_lr = self.e_lr
        optimizer_e = torch.optim.Adam(
            self.model.parameters(),
            lr=e_lr,
            weight_decay=self.args.wd
        )

        return optimizer_e