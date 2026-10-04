import torch
class Discriminator(torch.nn.Module):
    def __init__(self, args, sub_id):
        super(Discriminator, self).__init__()

        latent_dim = args.latent_dim
        hidden1 = args.hidden_dim[0]
        hidden2 = args.hidden_dim[1]

        # If orthogonal constraint is enabled, use adversarial training with GRL
        if args.orth == 'yes':
            self.dis = torch.nn.Sequential(
                GradientReversal(args.lam),  # Reverses gradient during backpropagation
                torch.nn.Linear(latent_dim, hidden1),
                torch.nn.ELU(),
                torch.nn.Linear(hidden1, hidden2),
                torch.nn.Linear(hidden2, sub_id + 2)  # Output dimension includes subject classes
            )
        else:
            # Standard discriminator without GRL
            self.dis = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, hidden1),
                torch.nn.ELU(),
                torch.nn.Linear(hidden1, hidden2),
                torch.nn.Linear(hidden2, sub_id + 2)
            )

    def forward(self, z, labels, sub_id):
        # Forward pass of the discriminator
        return self.dis(z)


class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26

    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    """
    Gradient Reversal Layer as introduced in domain adversarial neural networks.
    Multiplies the gradient by `-lambda_` during the backward pass, allowing
    adversarial learning.

    Args:
        lambda_ (float): Scaling factor for the reversed gradient.

    Methods:
        forward(x): Applies the gradient reversal operation.

    Returns:
        Tensor: Input tensor `x` unchanged in the forward pass.
    """

    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)



class OrthLoss(torch.nn.Module):
    """
    Computes orthogonal loss to encourage feature separation between two
    feature matrices `D1` and `D2` by minimizing the dot product of their
    normalized representations, as described in "Domain Separation Networks"
    (https://arxiv.org/abs/1608.06019).

    Args:
        D1 (Tensor): First feature matrix of shape (batch_size, feature_dim).
        D2 (Tensor): Second feature matrix of shape (batch_size, feature_dim).

    Returns:
        Tensor: Scalar orthogonality loss.
    """

    def __init__(self):
        super(OrthLoss, self).__init__()

    def forward(self, D1, D2):
        # Flatten feature maps
        D1 = D1.view(D1.size(0), -1)
        D2 = D2.view(D2.size(0), -1)

        # Row-wise L2 normalization
        D1_norm = D1 / (torch.norm(D1, p=2, dim=1, keepdim=True).expand_as(D1) + 1e-6)
        D2_norm = D2 / (torch.norm(D2, p=2, dim=1, keepdim=True).expand_as(D2) + 1e-6)

        # Compute dot-product matrix and square it
        dot_product = D1_norm.mm(D2_norm.t()).pow(2)

        # Return average squared dot product as orthogonality loss
        return torch.mean(dot_product)
