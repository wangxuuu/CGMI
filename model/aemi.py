import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch.nn.functional as F
class AECross(nn.Module):
    
    def __init__(self, x_dim, y_dim, hidden_size=60, alpha=1, lam=1):
        """

        Paired AE models with cross predictive regularization
        
        """
        
        super(AECross, self).__init__()
        
        # choosing hidden layer sizes
        # Lx = int(2**np.floor(np.log2(x_dim)))
        # Ly = int(2**np.floor(np.log2(y_dim)))
        
        Lx, Ly = 1024, 1024
        
        if x_dim > 2048:
            Lx = int(2**np.floor(np.log2(x_dim)))
        if y_dim > 2048:
            Ly = int(2**np.floor(np.log2(y_dim)))

        self.x_encoder = nn.Sequential(nn.Linear(1, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Sequential(nn.Linear(hidden_size, 3))
            )
        
        self.y_encoder = nn.Sequential(nn.Linear(1, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Sequential(nn.Linear(hidden_size, 3))
            )
        
        self.xx_decoder = nn.Sequential(nn.Linear(3, hidden_size),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1)
                                )
                                    #   nn.LeakyReLU(negative_slope=0.2))
        
        self.yy_decoder = nn.Sequential(nn.Linear(3, hidden_size),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1)
                                )
                                    #   nn.LeakyReLU(negative_slope=0.2))
        
        self.yx_decoder = nn.Sequential(nn.Linear(3, hidden_size),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1)
                                )
                                    #   nn.LeakyReLU(negative_slope=0.2))
        
        self.xy_decoder = nn.Sequential(nn.Linear(3, hidden_size),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1)
                                )
                                    #   nn.LeakyReLU(negative_slope=0.2))

        self.alpha = alpha
        self.lam = lam

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        for net in [self.x_encoder, self.y_encoder, self.xx_decoder, self.yy_decoder, 
                    self.xy_decoder, self.yx_decoder]:
            net.apply(init_weights)
        
    def encode(self, x_samples, y_samples):
        Zx = self.x_encoder(x_samples)
        Zy = self.y_encoder(y_samples)
        return Zx, Zy

    def cross_decode(self, Zx, Zy):
        y_hat = self.xy_decoder(Zx)
        x_hat = self.yx_decoder(Zy)
        return x_hat, y_hat
    
    def decode(self, Zx, Zy):
        x_hat = self.xx_decoder(Zx)
        y_hat = self.yy_decoder(Zy)
        return x_hat, y_hat
    
    def rec_loss(self, hat, samples):
        return torch.nn.functional.mse_loss(hat, samples, reduction='mean')

    def learning_loss(self, x_samples, y_samples):
        
        Zx, Zy = self.encode(x_samples, y_samples)
        Xh, Yh = self.decode(Zx, Zy)
        cXh, cYh = self.cross_decode(Zx, Zy)
        
        auto_loss = self.rec_loss(Xh, x_samples) + self.rec_loss(Yh, y_samples)
        cross_loss = self.rec_loss(cXh, x_samples) + self.rec_loss(cYh, y_samples)
        # print(auto_loss, cross_loss)
        
        return self.lam*cross_loss + self.alpha*auto_loss

# discriminator for infoGAN
class mnist_encoder(nn.Module):
    def __init__(self, channels=1, n_classes=10, img_size=32, code_dim=10):
        super(mnist_encoder, self).__init__()

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
        # self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        # self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        # validity = self.adv_layer(out)
        # label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        # return validity, label, latent_code
        return latent_code

class mnist_decoder(nn.Module):
    def __init__(self, code_dim=10):
        super(mnist_decoder, self).__init__()
        self.init_size = 32 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(code_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, code):
        # gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(code)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class AECrossMNIST(nn.Module):
    
    def __init__(self, code_dim=12, alpha=1, lam=1):
        """

        Paired AE models with cross predictive regularization
        
        """
        
        super(AECrossMNIST, self).__init__()
        
        # choosing hidden layer sizes
        # Lx = int(2**np.floor(np.log2(x_dim)))
        # Ly = int(2**np.floor(np.log2(y_dim)))

        self.x_encoder = mnist_encoder(channels=1, n_classes=10, img_size=32, code_dim=code_dim)
        
        self.y_encoder = mnist_encoder(channels=1, n_classes=10, img_size=32, code_dim=code_dim)
        
        self.xx_decoder = mnist_decoder(code_dim=code_dim)
        self.yy_decoder = mnist_decoder(code_dim=code_dim)
        self.yx_decoder = mnist_decoder(code_dim=code_dim)
        self.xy_decoder = mnist_decoder(code_dim=code_dim)

        self.alpha = alpha
        self.lam = lam

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        for net in [self.x_encoder, self.y_encoder, self.xx_decoder, self.yy_decoder, 
                    self.xy_decoder, self.yx_decoder]:
            net.apply(init_weights)
        
    def encode(self, x_samples, y_samples):
        Zx = self.x_encoder(x_samples)
        Zy = self.y_encoder(y_samples)
        return Zx, Zy

    def cross_decode(self, Zx, Zy):
        y_hat = self.xy_decoder(Zx)
        x_hat = self.yx_decoder(Zy)
        return x_hat, y_hat
    
    def decode(self, Zx, Zy):
        x_hat = self.xx_decoder(Zx)
        y_hat = self.yy_decoder(Zy)
        return x_hat, y_hat
    
    def rec_loss(self, hat, samples):
        return torch.nn.functional.mse_loss(hat, samples, reduction='mean')

    def learning_loss(self, x_samples, y_samples):
        
        Zx, Zy = self.encode(x_samples, y_samples)
        Xh, Yh = self.decode(Zx, Zy)
        cXh, cYh = self.cross_decode(Zx, Zy)
        
        auto_loss = self.rec_loss(Xh, x_samples) + self.rec_loss(Yh, y_samples)
        cross_loss = self.rec_loss(cXh, x_samples) + self.rec_loss(cYh, y_samples)
        # print(auto_loss, cross_loss)
        
        return self.lam*cross_loss + self.alpha*auto_loss


# discriminator for infoGAN
class cifar_encoder(nn.Module):
    def __init__(self, code_dim=10, hidden_size=60):
        super(cifar_encoder, self).__init__()

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
        self.latent_layer = nn.Sequential(nn.Linear(hidden_size * 8, code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        # validity = self.adv_layer(out)
        # label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        # return validity, label, latent_code
        return latent_code

class cifar_decoder(nn.Module):
    def __init__(self, code_dim=10):
        super(cifar_decoder, self).__init__()
        self.fc1 = nn.Linear(code_dim, 1024)
        self.fc2 = nn.Linear(1024, 4 * 4 * 256)
        self.convt1 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1)
        self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.convt3 = nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding = 1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(4 * 4 * 256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, code):
        # x = torch.cat((noise, labels, code), -1)
        x = F.relu(self.bn1(self.fc1(code)))
        x = F.relu(self.bn2(self.fc2(x))).view(-1, 256, 4, 4)

        x = F.relu(self.bn3(self.convt1(x)))
        x = F.relu(self.bn4(self.convt2(x)))
        x = self.convt3(x)
        return F.tanh(x)

class AECrossCIFAR(nn.Module):
    
    def __init__(self, code_dim=12, alpha=1, lam=1):
        """

        Paired AE models with cross predictive regularization
        
        """
        
        super(AECrossCIFAR, self).__init__()
        
        # choosing hidden layer sizes
        # Lx = int(2**np.floor(np.log2(x_dim)))
        # Ly = int(2**np.floor(np.log2(y_dim)))

        self.x_encoder = cifar_encoder(code_dim=code_dim)
        
        self.y_encoder = cifar_encoder(code_dim=code_dim)
        
        self.xx_decoder = cifar_decoder(code_dim=code_dim)
        self.yy_decoder = cifar_decoder(code_dim=code_dim)
        self.yx_decoder = cifar_decoder(code_dim=code_dim)
        self.xy_decoder = cifar_decoder(code_dim=code_dim)

        self.alpha = alpha
        self.lam = lam

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        for net in [self.x_encoder, self.y_encoder, self.xx_decoder, self.yy_decoder, 
                    self.xy_decoder, self.yx_decoder]:
            net.apply(init_weights)
        
    def encode(self, x_samples, y_samples):
        Zx = self.x_encoder(x_samples)
        Zy = self.y_encoder(y_samples)
        return Zx, Zy

    def cross_decode(self, Zx, Zy):
        y_hat = self.xy_decoder(Zx)
        x_hat = self.yx_decoder(Zy)
        return x_hat, y_hat
    
    def decode(self, Zx, Zy):
        x_hat = self.xx_decoder(Zx)
        y_hat = self.yy_decoder(Zy)
        return x_hat, y_hat
    
    def rec_loss(self, hat, samples):
        return torch.nn.functional.mse_loss(hat, samples, reduction='mean')

    def learning_loss(self, x_samples, y_samples):
        
        Zx, Zy = self.encode(x_samples, y_samples)
        Xh, Yh = self.decode(Zx, Zy)
        cXh, cYh = self.cross_decode(Zx, Zy)
        
        auto_loss = self.rec_loss(Xh, x_samples) + self.rec_loss(Yh, y_samples)
        cross_loss = self.rec_loss(cXh, x_samples) + self.rec_loss(cYh, y_samples)
        # print(auto_loss, cross_loss)
        
        return self.lam*cross_loss + self.alpha*auto_loss


def train(model, X_train, Y_train,
          batch_size=512, lr=0.0001, epochs=300, patience=30, 
          quiet=True):
    """
    training loop for LMI models

    :param model: LMI model

    :param X_train: train samples, shape (N_samples, N_features)
    :param Y_train: train samples, shape (N_samples, N_features)
    :param X_test: test samples, shape (N_samples, N_features)
    :param Y_test: test samples, shape (N_samples, N_features)

    :param batch_size: samples per batch, defaults to 512
    :param lr: learning rate for Adam optimizer, defaults to 1e-4
    :param epochs: max number of epochs, defaults to 300
    :param patience: epochs without val. loss decline before early stopping, 
                     defaults to 300
    :param quiet: suppress training progress display, defaults to True
    """
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-07)
    
    train_dataloader = DataLoader(list(zip(X_train, Y_train)), 
                                batch_size=batch_size, 
                                shuffle=True)

    
    with tqdm(range(epochs), unit='Epoch', disable=quiet) as tepoch:
        for epoch in tepoch:
            
            for i, (X, Y) in enumerate(train_dataloader):
                
                model.train() 
                model_loss = model.learning_loss(X, Y)

                optimizer.zero_grad()
                model_loss.backward()
                optimizer.step()
        


def ae(Xs, Ys,
       regularizer='models.AECross', 
       alpha=1, lam=1,
       N_dims=8, batch_size=512, lr=0.0001, epochs=300,
       validation_split=0.3, patience=30, quiet=True):
    """
    train paired AE model and embed data

    :param Xs:
    :param Ys:

    :param train_indices:
    :param test_indices:

    :param regularizer:
    :param alpha:
    :param lam:
    :param N_dims:
    
    :param batch_size:
    :param lr:
    :param epochs:
    :param validation_split:
    :param patience:
    :param quiet:
    """
    X_train = Xs
    Y_train = Ys
    
    if 128 < N_dims or 128 < N_dims:
        warnings.warn("Hidden layer smaller than latent dimension. Consider reducing N_dims")
        
    # assert X_train.shape[1] // 4 > 0, "Hidden layer with size 0. Consider tiling input."
    # assert Y_train.shape[1] // 4 > 0, "Hidden layer with size 0. Consider tiling input."
    
    model = eval(regularizer)(X_train.shape[1], Y_train.shape[1], N_dims, 
                              alpha=alpha, lam=lam).cuda()
    
    train(model, X_train, Y_train, 
          batch_size=batch_size, lr=lr, epochs=epochs, patience=patience,
          quiet=quiet)
    
    with torch.no_grad():
        model.eval()
        Zx, Zy = model.encode(Xs, Ys)
        
        Zx, Zy = Zx.cpu(), Zy.cpu()


        return Zx.cpu(), Zy.cpu(), model

def lmi(Xs, Ys, regularizer='models.AECross', 
         alpha=1, lam=1,
         N_dims=8, validation_split=0.5, estimate_on_val=True,
         batch_size=512, lr=0.0001, epochs=300, patience=30,
         quiet=True):
    """
    return pMIs, with NaNs for points not included in KSG estimate

    :param Xs:
    :param Ys:

    :param regularizer:
    :param alpha:
    :param lam:
    :param N_dims:
    
    :param batch_size:
    :param lr:
    :param epochs:
    :param validation_split:
    :param patience:
    :param quiet:
    """

    # Xs = torch.from_numpy(np.nan_to_num((Xs - Xs.mean(axis=0)) / Xs.std(axis=0))).float().cuda()
    # Ys = torch.from_numpy(np.nan_to_num((Ys - Ys.mean(axis=0)) / Ys.std(axis=0))).float().cuda()

    # Xs = torch.clip(Xs, min=-10, max=10)
    # Ys = torch.clip(Ys, min=-10, max=10)

    # assert len(Xs) == len(Ys), "X and Y must be same length!"
    
    # N_train = int(len(Xs) * (1 - validation_split))
    
    # indices = np.arange(len(Xs))
    # np.random.shuffle(indices)
    
    # train_indices = indices[:N_train]
    # test_indices = indices[N_train:]

    
    Zx, Zy, model = ae(Xs, Ys,
                regularizer=regularizer, N_dims=N_dims, batch_size=batch_size,
                patience=patience, epochs=epochs, 
                lr=lr, quiet=quiet,
                alpha=alpha, lam=lam)

    if torch.isnan(Zx).any() or torch.isnan(Zy).any():
        warnings.warn("NaNs in embedding! converted to 0s")

    Zx = torch.nan_to_num(Zx)
    Zy = torch.nan_to_num(Zy)
    
    estimate = 0

    if estimate_on_val:

        # make nan array
        estimate = np.zeros(len(Xs))
        estimate += np.NaN

        # fill val pMIs
        estimate[indices[N_train:]] = ksg.mi(Zx.cpu()[indices[N_train:]], 
        Zy.cpu()[indices[N_train:]])
    
    else:
        estimate += ksg.mi(Zx.cpu(), Zy.cpu())
    
    return estimate, (Zx.cpu(), Zy.cpu()), model