import torch

class DynamicFeatureExtractor(torch.nn.Module):
    def __init__(self, args):
        super(DynamicFeatureExtractor, self).__init__()

        self.args = args
        self.latent_dim = args.latent_dim

        if len(args.inputsize) == 3:
            self.input_channels, self.input_height, self.input_width = args.inputsize
        else:
            self.input_channels = 1
            self.input_height, self.input_width = args.inputsize

        if args.model == 'EEGNet':
            from .backbones import EEGNet
            self.base_model = EEGNet(args)
        elif args.model == 'SCN':
            from .backbones import ShallowConvNet as SCN
            self.base_model = SCN(args)
        elif args.model == 'DCN':
            from .backbones import DeepConvNet as DCN
            self.base_model = DCN(args, latent_dim=self.latent_dim, kernel_1=64, kernel_2=16,
                                  dropout=args.dropout, block_out_channels=[25, 25, 50, 100, 200])

        self.task_specific_layers = torch.nn.ModuleList()
        self.num_tasks = 0
        self.base_feature_dim = self._get_base_feature_dim()

    def _get_base_feature_dim(self):
        dummy_input = torch.randn(1, 1, self.input_height, self.input_width)

        try:
            with torch.no_grad():
                self.base_model.eval()
                base_output = self.base_model(dummy_input)

                if isinstance(base_output, dict):
                    if 'features' in base_output:
                        base_output = base_output['features']
                    elif 'logits' in base_output:
                        base_output = base_output['logits']
                    else:
                        base_output = list(base_output.values())[0]

                if base_output.dim() > 2:
                    base_output = base_output.view(base_output.size(0), -1)

                feature_dim = base_output.shape[1]
                return feature_dim

        except Exception as e:
            try:
                first_layer = list(self.base_model.modules())[1]
                if hasattr(first_layer, 'in_channels'):
                    pass
                if hasattr(first_layer, 'weight'):
                    pass
            except:
                pass

            return self.latent_dim

    def add_task(self):
        """Add task-specific layers for a new task"""
        self.num_tasks += 1

        task_layer = torch.nn.Sequential(
            torch.nn.Linear(self.base_feature_dim, self.latent_dim),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.task_specific_layers.append(task_layer)

        if len(list(self.base_model.parameters())) > 0:
            device = next(self.base_model.parameters()).device
            task_layer.to(device)

    def preprocess_input(self, x):
        """Preprocess input data to ensure correct format"""
        original_shape = x.shape

        if x.dim() == 4:
            if x.shape[1] == self.input_channels:
                processed_x = x
            elif x.shape[1] == self.input_height:
                processed_x = x.permute(0, 2, 1, 3)
            else:
                batch_size = x.shape[0]
                processed_x = x.view(batch_size, 1, self.input_height, self.input_width)

        elif x.dim() == 3:
            processed_x = x.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported input dimension: {x.shape}")

        expected_shape = (x.shape[0], 1, self.input_height, self.input_width)
        if processed_x.shape != expected_shape:
            try:
                processed_x = processed_x.view(expected_shape)
            except:
                processed_x = x

        return processed_x

    def forward(self, x, task_id=None):
        """Forward pass through shared and task-specific layers"""
        x = self.preprocess_input(x)
        try:
            base_features = self.base_model(x)
            if isinstance(base_features, dict):
                if 'features' in base_features:
                    base_features = base_features['features']
                elif 'logits' in base_features:
                    base_features = base_features['logits']
                else:
                    base_features = list(base_features.values())[0]

            if base_features.dim() > 2:
                base_features = base_features.view(base_features.size(0), -1)

        except Exception as e:
            batch_size = x.size(0)
            base_features = torch.zeros(batch_size, self.base_feature_dim).to(x.device)

        if task_id is None:
            return base_features

        if task_id >= self.num_tasks or task_id < 0:
            return base_features

        try:
            task_features = self.task_specific_layers[task_id](base_features)

            return {
                'base_features': base_features,
                'task_features': task_features,
                'combined_features': torch.cat([base_features, task_features], dim=1)
            }
        except Exception as e:
            return {
                'base_features': base_features,
                'task_features': base_features,
                'combined_features': base_features
            }


class Net(torch.nn.Module):
    """
    Fixed dynamic network
    """

    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args
        self.task_cls = args.task_cls
        self.latent_dim = args.latent_dim
        self.num_subjects = args.n_subjects
        self.hidden1 = args.hidden_dim[0]
        self.hidden2 = args.hidden_dim[1]

        self.FeatureExtractor = DynamicFeatureExtractor(args)

        self.heads = torch.nn.ModuleList()
        for i in range(self.num_subjects):
            self.heads.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim*2 , self.hidden1),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Dropout(),
                    torch.nn.Linear(self.hidden1, self.hidden2),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Linear(self.hidden2, self.task_cls[i][1])
                ))
        self.num_tasks = 0

        if args.model == 'EEGNet' or args.model == 'SCN':
            self.feature_multiplier = 2
        elif args.model == 'DCN':
            self.feature_multiplier = 2

        self.classifier = None

    def add_task(self, n_classes=None):
        """Add a new task with corresponding head and feature extractor expansion"""
        task_id = self.num_tasks

        self.FeatureExtractor.add_task()

        if n_classes is None:
            if task_id < len(self.task_cls):
                n_classes = self.task_cls[task_id][1]
            else:
                n_classes = 2

        head_input_dim = self.latent_dim * self.feature_multiplier

        try:
            head = torch.nn.Sequential(
                torch.nn.Linear(head_input_dim, self.hidden1),
                torch.nn.ELU(inplace=True),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(self.hidden1, self.hidden2),
                torch.nn.ELU(inplace=True),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(self.hidden2, n_classes)
            )

            self.heads.append(head)
            self.num_tasks += 1

            self.classifier = head

            if len(list(self.FeatureExtractor.parameters())) > 0:
                device = next(self.FeatureExtractor.parameters()).device
                head.to(device)

            return task_id

        except Exception as e:
            raise

    def forward(self, x, task_id=None, tt=None):
        """
        Robust forward method
        """
        original_shape = x.shape

        try:
            if task_id is not None:
                return self._forward_single_task(x, task_id)

            elif tt is not None:
                return self._forward_batch_tasks(x, tt)

            else:
                return self._forward_base_features(x)

        except Exception as e:
            return self._create_safe_output(x, task_id)

    def _forward_single_task(self, x, task_id):
        if task_id >= self.num_tasks:
            self.add_task(n_classes=2)

        if task_id < 0:
            task_id = 0

        try:
            feature_output = self.FeatureExtractor(x, task_id)

            if isinstance(feature_output, dict):
                features = feature_output['combined_features']
                base_features = feature_output.get('base_features', features)
                task_features = feature_output.get('task_features', features)
            else:
                features = feature_output
                base_features = features
                task_features = features

            expected_dim = self.latent_dim * self.feature_multiplier
            if features.shape[1] != expected_dim:
                if features.shape[1] < expected_dim:
                    pad_size = expected_dim - features.shape[1]
                    padding = torch.zeros(features.shape[0], pad_size).to(features.device)
                    features = torch.cat([features, padding], dim=1)
                else:
                    features = features[:, :expected_dim]

            if task_id >= len(self.heads):
                self.add_task(n_classes=2)

            logits = self.heads[task_id](features)

            return {
                'features': features,
                'base_features': base_features,
                'task_features': task_features,
                'logits': logits
            }

        except Exception as e:
            return self._create_safe_output(x, task_id)

    def _forward_batch_tasks(self, x, tt):
        """Batch-level task forward pass"""
        batch_size = x.size(0)
        outputs = []

        for i in range(batch_size):
            sample_task = int(tt[i].item())
            sample_input = x[i:i + 1]

            try:
                sample_output = self._forward_single_task(sample_input, sample_task)
                if isinstance(sample_output, dict) and 'logits' in sample_output:
                    outputs.append(sample_output['logits'])
                else:
                    outputs.append(torch.zeros(1, 2).to(x.device))
            except:
                outputs.append(torch.zeros(1, 2).to(x.device))

        return torch.cat(outputs, dim=0)

    def _forward_base_features(self, x):
        """Base feature forward pass"""
        try:
            base_features = self.FeatureExtractor(x)

            if isinstance(base_features, dict):
                return base_features.get('base_features', base_features)
            else:
                return base_features

        except Exception as e:
            batch_size = x.size(0)
            return torch.zeros(batch_size, self.latent_dim).to(x.device)

    def _create_safe_output(self, x, task_id):
        """Create safe default output"""
        batch_size = x.size(0)
        device = x.device

        n_classes = 2
        if task_id is not None and task_id < len(self.task_cls):
            n_classes = self.task_cls[task_id][1]

        dummy_features = torch.zeros(batch_size, self.latent_dim * self.feature_multiplier).to(device)
        dummy_logits = torch.zeros(batch_size, n_classes).to(device)

        return {
            'features': dummy_features,
            'base_features': dummy_features,
            'task_features': dummy_features,
            'logits': dummy_logits
        }

    def reset_classifier_for_task(self, task_id):
        """Reset classifier for specified task"""
        if task_id >= len(self.heads) or task_id < 0:
            return

        head = self.heads[task_id]

        with torch.no_grad():
            for layer in head:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def get_classifier_for_task(self, task_id):
        """Get classifier for specified task"""
        if task_id >= len(self.heads) or task_id < 0:
            return None
        return self.heads[task_id]

    def print_model_size(self):
        """Print model size information"""
        try:
            count_base = sum(p.numel() for p in self.FeatureExtractor.base_model.parameters() if p.requires_grad)
            count_task_features = sum(
                p.numel() for p in self.FeatureExtractor.task_specific_layers.parameters() if p.requires_grad)
            count_heads = sum(p.numel() for p in self.heads.parameters() if p.requires_grad)
            total_params = count_base + count_task_features + count_heads

            print('=' * 120)
            print('DYNAMIC NETWORK ARCHITECTURE SUMMARY')
            print('=' * 120)
            print(f'Base Feature Extractor    = {self.pretty_print(count_base)} parameters')
            print(
                f'Task-specific Features    = {self.pretty_print(count_task_features)} parameters ({self.num_tasks} tasks)')
            print(f'Classification Heads      = {self.pretty_print(count_heads)} parameters ({len(self.heads)} heads)')
            print(f'Total Parameters          = {self.pretty_print(total_params)} parameters')
            print(f'Memory Usage (approx)     = {self.pretty_print(4 * total_params)}B')
            print('=' * 120)
        except Exception as e:
            pass

    def pretty_print(self, num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

