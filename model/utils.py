# https://github.com/haaami01/google-research/blob/be6428bcb2886f4a4dda64aadafaf234b47aac5b/ciw_label_noise/utils.py#L45

import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# estimate mutual information between discrete random variables
def mutual_information(x, y):
    # x = torch.Tensor(x)
    # y = torch.Tensor(y)

    # Calculate the joint probability distribution
    joint_distribution = Counter(zip(x.tolist(), y.tolist()))
    total_samples = len(x)

    mutual_info = 0.0

    for (x_val, y_val), count in joint_distribution.items():
        p_x_y = count / total_samples
        p_x = (x == x_val).sum() / total_samples
        p_y = (y == y_val).sum() / total_samples

        if p_x_y > 0:
            mutual_info += p_x_y * (np.log(p_x_y) - np.log(p_x) - np.log(p_y))

    return mutual_info


def maybe_one_hot(labels, depth):
    """Convert categorical labels to one-hot, if needed.

    Args:
        labels: A `Tensor` containing labels.
        depth: An integer specifying the depth of one-hot represention (number of
        classes).

    Returns:
        One-hot labels.
    """
    if len(labels.shape) > 1:
        return labels
    else:
        return torch.one_hot(labels, depth=depth)


def get_smoothed_labels(labels, preds, smoothing_weights):
    """Smoothen the labels."""
    smoothing_weights = smoothing_weights.reshape(-1, 1)
    return labels * smoothing_weights + preds * (1. - smoothing_weights)


def mixup(images,
          labels,
          num_classes,
          mixup_alpha,
          mixing_weights=None,
          mixing_probs=None):
    """Mixup with mixing weights and probabilities.

    Args:
        images: A `Tensor` containing batch of images.
        labels: A `Tensor` containing batch of labels.
        num_classes: Number of classes.
        mixup_alpha: Parameter of Beta distribution for sampling mixing ratio
        (applicable for regular mixup).
        mixing_weights: A `Tensor` of size [batch_size] specifying mixing weights.
        mixing_probs: A `Tensor` of size [batch_size] specifying probabilities for
        sampling images for imixing.

    Returns:
        Minibatch of mixed up images and labels.
    """

    images = images.numpy()
    labels = maybe_one_hot(labels, num_classes).numpy()
    num_examples = images.shape[0]
    mixing_ratios_im = np.random.beta(
        mixup_alpha, mixup_alpha, size=(num_examples, 1, 1, 1))
    mixing_ratios_lab = np.reshape(mixing_ratios_im, [num_examples, 1])
    if mixing_probs is None:
        mixing_indices = np.random.permutation(num_examples)
    else:
        mixing_probs = np.round(mixing_probs, 5)
        mixing_probs = mixing_probs / np.sum(mixing_probs)
        mixing_indices = np.random.choice(
            num_examples, size=num_examples, replace=True, p=mixing_probs)
    if mixing_weights is not None:
        mixing_ratios_im = mixing_weights / (
            mixing_weights + mixing_weights[mixing_indices])
        mixing_ratios_im = np.reshape(mixing_ratios_im, [-1, 1, 1, 1])
        # mix labels in same proportions
        mixing_ratios_lab = np.reshape(mixing_ratios_im, [num_examples, 1])
    images = (
        images * mixing_ratios_im + images[mixing_indices] *
        (1. - mixing_ratios_im))
    labels = (
        labels * mixing_ratios_lab + labels[mixing_indices] *
        (1. - mixing_ratios_lab))
    return images, labels

def Conjugate_Hall(divtype='KL'):
    """
    Given the type of f-divergence, return the conjugate function of the corresponding convex function f.

    inputs:
        -t  input of the conjugate function;
        -divtype: name of the f-divergence; defalut: KL divergence
    """
    if divtype == 'KL':
        # KL divergence
        # t in R
        return lambda t:(t-1).exp()
    elif divtype == 'RKL':
        # Reserve KL divergence
        # t < 0
        return lambda t:-1-torch.log(-t)
    elif divtype == 'Pearson':
        # Pearson divergence
        # t in R
        return lambda t:1/4 * t**2 + t
    elif divtype == 'SquaredHel':
        # Squared Hellinger
        # t < 1
        return lambda t:t/(1-t)
    elif divtype == 'JS':
        # JS divergence
        # t < log 2
        return lambda t:-torch.log(2-t.exp())

def ActiF_Hall(divtype='KL'):
    """
    Given the type of f-divergence, return the conjugate function of the corresponding output activaion function gf.

    inputs:
        -t  input of the activation function;
        -divtype: name of the f-divergence; defalut: KL divergence
    """
    if divtype == 'KL':
        # KL divergence
        return lambda t: t
    elif divtype == 'RKL':
        # Reserve KL divergence
        return lambda t:-(-t).exp()
    elif divtype == 'Pearson':
        # Pearson divergence
        return lambda t:t
    elif divtype == 'SquaredHel':
        # Squared Hellinger
        return lambda t:1-(-t).exp()
    elif divtype == 'JS':
        # JS divergence
        return lambda t:np.log(2)-torch.log(1+(-t).exp())

def Mapping_Hall(divtype='KL'):
    if divtype == 'KL':
        # KL divergence
        return lambda t: (t-1).exp()
    elif divtype == 'RKL':
        # Reserve KL divergence
        return lambda t: -1/t
    elif divtype == 'Pearson':
        # Pearson divergence
        return lambda t: t/2 + 1
    elif divtype == 'SquaredHel':
        # Squared Hellinger
        return lambda t: 1/(1-t)**2
    elif divtype == 'JS':
        # JS divergence
        return lambda t: t.exp()/(2-t.exp())

def Conf_gf_Hall(divtype='KL'):
    """
    Given the type of f-divergence, return the f*(gf(t)), where f* is the conjugate function.

    inputs:
        -t  input of the activation function;
        -divtype: name of the f-divergence; defalut: KL divergence
    """
    if divtype == 'KL':
        # KL divergence
        return lambda t: (t-1).exp()
    elif divtype == 'RKL':
        # Reserve KL divergence
        return lambda t: t-1
    elif divtype == 'Pearson':
        # Pearson divergence
        return lambda t: 1/(4*t**2) + t
    elif divtype == 'SquaredHel':
        # Squared Hellinger
        return lambda t: t.exp()-1
    elif divtype == 'JS':
        # JS divergence
        return lambda t: -torch.log(2-2/(1+(-t).exp()))

def Devf_gf_Hall(divtype='KL'):
    """
    Given the type of f-divergence, return the (f*)'(gf(t)).

    inputs:
        -t  input of the activation function;
        -divtype: name of the f-divergence; defalut: KL divergence
    """
    if divtype == 'KL':
        # KL divergence
        return lambda t: (t-1).exp()
    elif divtype == 'RKL':
        # Reserve KL divergence
        return lambda t: t.exp()
    elif divtype == 'Pearson':
        # Pearson divergence
        return lambda t: t/2 + 1
    elif divtype == 'SquaredHel':
        # Squared Hellinger
        return lambda t: (2*t).exp()
    elif divtype == 'JS':
        # JS divergence
        return lambda t: t.exp()


class Discriminator_CIFAR(nn.Module):
    def __init__(self, hidden_size=60):
        super(Discriminator_CIFAR, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 4, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(hidden_size * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(hidden_size * 8, 1)
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(hidden_size * 8, hidden_size * 8),
            nn.Linear(hidden_size * 8, 10),
            nn.Softmax()
        )
        self.latent_layer = nn.Sequential(
            nn.Linear(hidden_size * 8, 2)
        )

    def forward(self, x):
        # return self.main(x).view(-1, 1)
        out = self.conv_blocks(x)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)
        return validity, label, latent_code

# discriminator for infoGAN
class Discriminator(nn.Module):
    def __init__(self, channels=1, n_classes=10, img_size=32, code_dim=10):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code

def kernel_loss(D, kernel, f, x, y, lamb=1.0, stats='E', l1_type='DV'):
    y_size = y.shape[0]
    x_size = x.shape[0]
    KXY = kernel(x.reshape(x_size, -1), y.reshape(y_size, -1))
    KXX = kernel(x.reshape(x_size, -1), x.reshape(x_size, -1))
    KYY = kernel(y.reshape(y_size, -1), y.reshape(y_size, -1))
    alpha_log = D(y.reshape(y_size, -1)).reshape(-1, 1)

    alpha = alpha_log.exp()

    if stats == 'E':
        omega_star = ((alpha.T.mm(KYY).mm(alpha))).squeeze()/(y_size**2) + KXX.mean() - 2*(KXY.mm(alpha)).mean()/y_size
    elif stats == 'U':
        # U statistic of MMD
        mask_y =  torch.ones([y_size]).diag().expand_as(KYY).bool().cuda()
        mask_x = torch.ones([x_size]).diag().expand_as(KXX).bool().cuda()
        KXX[mask_x] = 0
        KYY[mask_y] = 0
        omega_star = ((alpha.T.mm(KYY).mm(alpha))).squeeze()*y_size/(y_size-1) + KXX.sum()/(x_size*(x_size-1)) - 2*(KXY.mm(alpha)).mean()
    # print(alpha_log.exp().mean())
    if l1_type == 'DV':
        return -alpha_log.mean()+omega_star/(2*lamb)
    elif l1_type == 'NWJ':
        return f(alpha).mean()+omega_star/(2*lamb)
    elif l1_type == 'MMD':
        return omega_star