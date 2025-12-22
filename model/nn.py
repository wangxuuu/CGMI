import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd
from functools import partial
from scipy import integrate


def resample(data, batch_size, replace=False, return_idx=False):
    # Resample the given data sample and return the resampled data or index
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    if return_idx:
        return index
    else:
        return data[index]


class DNN_Net(nn.Module):
    def __init__(self, input_size=2, hidden_layers=2, hidden_size=100, sigma=0.02):
        super().__init__()
        self.hidden_layers = hidden_layers
        # self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size))
        for i in range(hidden_layers-1):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size,1))
        for i in range(hidden_layers+1):
            nn.init.normal_(self.fc[i].weight, std=sigma)
            nn.init.constant_(self.fc[i].bias, 0)

    def forward(self, input):
        output = input
        for i in range(self.hidden_layers):
            output = F.elu(self.fc[i](output))
            # output = F.elu(self.batchnorm(self.fc[i](output)))
        output = self.fc[self.hidden_layers](output)
        return output


class Linear_discriminator(nn.Module):
    '''
    Fullly connect neural network for MNIST
    '''
    def __init__(self, img_shape):
        super(Linear_discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape))*2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        return self.model(img)

# DV loss function for mutual information estimation
def DV_loss(D, x, y, y_ref=None, reg=False, alpha=1.0, eps=0):
    '''
    D: discriminator
    x: data x
    y: data y
    '''
    # reshuffle y and concatenate with x
    joint_xy = torch.cat((x, y), dim=1)
    # if y_ref is None:
    #     marginal_xy = torch.cat((x, y[torch.randperm(x.size()[0])]), dim=1)
    # else:
    #     marginal_xy = torch.cat((x, y_ref[torch.randperm(x.size()[0])]), dim=1)
    if y_ref is None:
        x_tile = x.unsqueeze(0).repeat((y.shape[0], 1, 1, 1, 1))
        y_tile = y.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))
        marginal_xy = torch.cat((x_tile, y_tile), dim=-3).view(x.shape[0]*y.shape[0], -1)
    else:
        x_tile = x.unsqueeze(0).repeat((y_ref.shape[0], 1, 1, 1, 1))
        y_tile = y_ref.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))
        marginal_xy = torch.cat((x_tile, y_tile), dim=-3).view(x.shape[0]*y_ref.shape[0], -1)

    joint_xy = joint_xy + eps*torch.randn_like(joint_xy)
    marginal_xy = marginal_xy + eps*torch.randn_like(marginal_xy)

    if reg: # add regularization term
        return -D(joint_xy.view(joint_xy.size(0), -1)).mean() + torch.logsumexp(D(marginal_xy.view(marginal_xy.size(0), -1)), dim=0).mean() - np.log(marginal_xy.shape[0]) + alpha*(torch.logsumexp(D(marginal_xy.view(marginal_xy.size(0), -1)), dim=0).mean() - np.log(marginal_xy.shape[0]))**2
    else:
        return - D(joint_xy.view(joint_xy.size(0), -1)).mean() + torch.logsumexp(D(marginal_xy.view(marginal_xy.size(0), -1)), dim=0).mean() - np.log(marginal_xy.shape[0])

# NWJ loss function for mutual information estimation
def NWJ_loss(D, x, y, y_ref=None, reg=False, eps=0):
    '''
    D: discriminator
    x: data x
    y: data y
    '''

    joint_xy = torch.cat((x, y), dim=1)
    # marginal_xy = torch.cat((x, y[torch.randperm(x.size()[0])]), dim=1)
    # reshuffle y and concatenate with x
    if y_ref is None:
        x_tile = x.unsqueeze(0).repeat((y.shape[0], 1, 1, 1, 1))
        y_tile = y.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))
        marginal_xy = torch.cat((x_tile, y_tile), dim=-3).view(x.shape[0]*y.shape[0], -1)
    else:
        x_tile = x.unsqueeze(0).repeat((y_ref.shape[0], 1, 1, 1, 1))
        y_tile = y_ref.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))
        marginal_xy = torch.cat((x_tile, y_tile), dim=-3).view(x.shape[0]*y_ref.shape[0], -1)

    return - D(joint_xy.view(joint_xy.size(0), -1)).mean() + (D(marginal_xy.view(marginal_xy.size(0), -1))).exp().mean() - 1


# InfoNCE loss function for mutual information estimation
def NCE_loss(D, x, y, y_ref=None, reg=False, alpha=1.0, eps=0):
    """
    x: data x
    y: data y
    """
    if y_ref is None:
        x_tile = x.unsqueeze(0).repeat((y.shape[0], 1, 1, 1, 1))
        y_tile = y.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))
    else:
        x_tile = x.unsqueeze(0).repeat((y_ref.shape[0], 1, 1, 1, 1))
        y_tile = y_ref.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))

    T0 = D(torch.cat([x, y], dim=1).view(x.shape[0], -1))
    if y_ref is None:
        T1 = D(torch.cat([x_tile, y_tile], dim=-3).view(x.shape[0], y.shape[0], -1))
    else:
        T1 = D(torch.cat([x_tile, y_tile], dim=-3).view(x.shape[0], y_ref.shape[0], -1))
    if reg:
        return - T0.mean() + T1.logsumexp(dim=0).mean() - np.log(T1.shape[0]) + alpha*(T1.logsumexp(dim=0) - np.log(T1.shape[0])).pow(2).mean()
    else:
        return - T0.mean() + T1.logsumexp(dim=0).mean() - np.log(T1.shape[0])

# mi estimate assuming that D approximates the optimal log density ratio
def mi_estimate(D, xy):
    return D(xy.view(xy.size(0), -1)).mean()

# PCM loss for mutual information estimation
def PCM_loss(D, x, y, y_ref=None, alpha=1.0, reg=False):
    """
    alpha: the ratio of reference samples / data samples
    """
    # make the sample size be integer
    ref_samplesize = int(x.size()[0]*alpha)
    if y_ref is None:
        xy_margin = torch.cat((x[torch.randperm(ref_samplesize)], y[torch.randperm(ref_samplesize)]), dim=1)
    else:
        xy_margin = torch.cat((x[torch.randperm(ref_samplesize)], y_ref[torch.randperm(ref_samplesize)]), dim=1)

    return - F.logsigmoid(D(torch.cat((x, y), dim=1).view(x.size()[0], -1))).mean() - alpha * F.logsigmoid(-D(xy_margin.view(ref_samplesize, -1))).mean()

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
            marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
            channels: The number of channels for feature maps of each resolution.
            embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                nn.Linear(embed_dim, embed_dim))

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(2, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 2, 3, stride=1)

        self.time_layer = nn.Linear(2*28*28, 1)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, z=None):
        # Obtain the Gaussian random feature embedding for t
        t = t.squeeze() # (n, 1) to (n)
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        # reshape the output to let it be a vector
        return h.reshape(x.shape[0],-1), self.time_layer(h.view(h.size(0), -1))

def sample_sequence_on_the_fly(px, qx, t):
    t = t.to(px.device)
    t = t.unsqueeze(dim=1).unsqueeze(dim=2) # (n, 1, 1, 1) to match the size of px and qx
    # note: t is functioning as \alpha(t) here
    return torch.sqrt(1 - t**2) * px + (t * qx)

def joint_score_estimation(sde, scorenet, px, qx, eps=1e-5, likelihood_weighting=False):
    """Return the loss for training the time data score function

    Args:
        sde: The stochastic differential equation that generates the data.
        scorenet: The score-based model.
        px: The data of px.
        qx: The data of qx.
        eps: a small float number.
        likelihood_weighting: Whether to use likelihood weighting to estimate
        the loss.
    """
    device = px.device
    n = px.size(0)
    t = torch.rand(n, 1) * (1-eps)
    # xt is the linear combination of px and qx: xt = sqrt(1-t^2) * px + t * qx
    xt = sample_sequence_on_the_fly(px, qx, t)

    # device things
    px = px.to(device)  # noise
    qx = qx.to(device)  # data
    xt = xt.to(device)  # interp
    t = t.to(device)
    t = t.detach()

    # compute the score of the interpolated data
    xt.requires_grad_()
    vectors = torch.randn_like(xt, device=device).reshape(xt.size(0), -1)
    score_x, score_t = scorenet(xt, t)

    grad1 = torch.cat([score_x, score_t], dim=-1)
    # grad1 = score_x
    gradv = torch.sum(score_x * vectors)
    grad2 = autograd.grad(gradv, xt, create_graph=True)[0].reshape(xt.size(0), -1)

    if not likelihood_weighting:
        weighting_fn = lambda t: torch.ones_like(t)
    else:
        weighting_fn = lambda t: sde.marginal_prob(torch.zeros_like(t), t)[1] ** 2

    def grad_weighting_fn(t):
        with torch.enable_grad():
            t.requires_grad_()
            return autograd.grad(weighting_fn(t).sum(), t)[0]

    # boundary conditions
    t0 = torch.zeros((len(px), 1)).to(px.device) + eps
    t1 = torch.ones((len(qx), 1)).to(qx.device) - eps

    # the appropriate weighting functions
    lambda_t = weighting_fn(t)
    lambda_t0 = weighting_fn(t0)
    lambda_t1 = weighting_fn(t1)
    if not likelihood_weighting:
        lambda_dt = 0.
    else:
        lambda_dt = grad_weighting_fn(t)

    # SSM loss (technically has the s(x,t)**2 term in there too)
    ssm_loss1 = (torch.sum(grad1 * grad1, dim=-1) / 2.).view(
        lambda_t.size()) * lambda_t
    ssm_loss2 = torch.sum(vectors * grad2, dim=-1).view(
        lambda_t.size()) * lambda_t
    ssm_loss = ssm_loss1 + ssm_loss2

    term1 = (scorenet(px, t0)[-1]) * lambda_t0  # T=0 is noise
    term2 = (scorenet(qx, t1)[-1]) * lambda_t1  # T=1 is data

    # need to differentiate score wrt t
    with torch.enable_grad():
        t.requires_grad_(True)
        xt_score_dt = \
        autograd.grad(scorenet(xt, t)[-1].sum(), t, create_graph=True)[0]
    term3 = (xt_score_dt) * lambda_t
    term4 = score_t * lambda_dt

    time_loss = term1 - term2 + term3 + term4
    loss = ssm_loss + time_loss

    # 1-d so we can just take the mean rather than summing
    return loss.mean()

def get_density_ratio_fn(rtol=1e-6, atol=1e-6, method='RK45', eps=1e-5):
    """Create a function to compute the density ratios of a given point.
    """

    def ratio_fn(score_model, x, score_type):
        with torch.no_grad():
            def ode_func(t, y, x, score_model):
                score_model.eval()
                t = (torch.ones(x.size(0)) * t).to(x.device).view(-1, 1)
                x = x.to(x.device)

                if score_type == 'joint':
                    rx = score_model(x, t)[-1]
                else:
                    rx = score_model(x, t)
                    # rx = np.reshape(rx.detach().cpu().numpy(), -1)
                rx = np.reshape(rx.detach().cpu().numpy(), -1)
                return rx

        # now just a function of t
        p_get_rx = partial(ode_func, x=x, score_model=score_model)
        # TODO: flipped (1, eps) for toy datasets
        solution = integrate.solve_ivp(p_get_rx, (eps, 1.),
                                        np.zeros((x.shape[0],)),
                                        method=method, rtol=rtol, atol=atol)
        nfe = solution.nfev
        density_ratio = solution.y[:, -1]
        # print("ratio computation took {} function evaluations.".format(nfe))

        return density_ratio, nfe

    return ratio_fn