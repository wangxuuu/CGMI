import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from models.utils import *
from models.nn import *

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS_PER_STEP = 20  # Increase this for more accurate results
DATASET_NAME = 'MNIST' # Change to 'CIFAR10' to switch
ROWS_TO_TEST = [8, 16, 24, 32] 


# --- 1. Data Loading ---
def get_dataloader(name, batch_size):
    if name == 'MNIST':
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        input_dim = 32 * 32
        img_shape = (1, 32, 32)
        total_rows = 32
    elif name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        input_dim = 3 * 32 * 32
        img_shape = (3, 32, 32)
        total_rows = 32
    
    # Use a subset for faster demonstration (Optional)
    # dataset = Subset(dataset, range(5000)) 
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, input_dim, img_shape, total_rows

# --- 2. Masking Function ---
def mask_bottom_rows(images, t, img_shape):
    """
    Keeps top t rows, zeros out the rest.
    images: [B, C, H, W]
    """
    B, C, H, W = images.shape
    mask = torch.zeros_like(images)
    mask[:, :, :t, :] = 1.0
    return images * mask

# --- 3. The MINE Estimator Network (Statistics Network) ---
class JointCritic(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=512):
        super(JointCritic, self).__init__()
        # Taking X and Y concatenated
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        # Flatten inputs
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        input_pair = torch.cat([x, y], dim=1)
        return self.net(input_pair)

def code(encoder, data, type='dis'):
    """Obtain the codes from the pixel data.count

    Args:
        - encoder: the encoder of InfoGAN model Q(c|X)
        - data: pixel data

    Return:
        The codes/representation of data.
    """
    _, label, latent_code = encoder(data)
    dis_code = torch.argmax(label, dim=1)
    if type == 'dis':
        return F.one_hot(dis_code.long(), num_classes=10).float()
    elif type == 'con':
        return latent_code
    elif type == 'all':
        return torch.cat((latent_code, F.one_hot(dis_code.long(), num_classes=10).float()), dim=1)
    elif type == 'digit':
        return dis_code
    elif type == 'logit':
        return torch.cat((latent_code, label), dim=1)
    elif type == 'lmi':
        return torch.cat([dis_code.reshape(-1,1), latent_code], dim=1)
    
# --- 4. MINE Training Loop ---
def estimate_mi(loader, input_dim, img_shape, setting, t):
    """
    Trains the estimator for a specific setting and row count t.
    """
    # load the pretrained encoder
    if DATASET_NAME == 'MNIST':
        # pass # No encoder needed for MNIST in this example
        # move to GPU
        # net = DNN_Net(input_size=(code_dim+n_classes)*2)
        discriminator = Discriminator(img_size=32, n_classes=10, code_dim=2)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        discriminator.cuda()
        # load the discriminator
        discriminator.load_state_dict(torch.load(f'checkpoints/cont{2}dist10discriminator_199.pt'))
        
    elif DATASET_NAME == 'CIFAR10':
        # pass # No encoder needed for CIFAR10 in this example
        discriminator = Discriminator_CIFAR()
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        discriminator.cuda()
        # load the discriminator 
        discriminator.load_state_dict(torch.load(f'checkpoints/cifar_discriminator.pth'))
    # Adjust input dimensions based on setting
    if setting == 3: # Additivity: 2 images concatenated
        x_dim = input_dim * 2
        y_dim = input_dim * 2
        model = JointCritic(6, 6).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif setting == 2: # Data Processing: 2 identical images concatenated
        x_dim = input_dim * 2
        y_dim = input_dim * 2
        model = JointCritic(6, 6).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else: # Baseline
        x_dim = input_dim
        y_dim = input_dim
        model = JointCritic(3, 3).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    
    mi_est_history = []

    for epoch in range(EPOCHS_PER_STEP):
        epoch_mi = []
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(DEVICE)
            B = images.shape[0]

            # --- PREPARE DATA BASED ON SETTINGS (See Fig 3) ---
            
            # Setting 1: Baseline
            # X = Image, Y = Masked Image (top t rows)
            if setting == 1:
                X = images
                Y = mask_bottom_rows(images, t, img_shape)
                X = code(discriminator, X, type='lmi')
                Y = code(discriminator, Y, type='lmi')

            # Setting 2: Data Processing·
            # X = (Image, Same Image), Y = (Masked t, Masked t) 
            # (Simplification: using t1=t2=t for demonstration)
            elif setting == 2:
                X = torch.cat([code(discriminator, images, type='lmi'), code(discriminator, images, type='lmi')], dim=1) # Stack channels or width? Usually flattened concat.
                # Let's treat them as flattened vectors concatenated later, 
                # but here we keep tensor shape for masking logic.
                # Actually, simply concatenating along channel dim is easiest for processing, 
                # but for the MLP it will be flattened anyway.
                Y_single = mask_bottom_rows(images, t, img_shape)
                Y = torch.cat([code(discriminator, Y_single, type='lmi'), code(discriminator, Y_single, type='lmi')], dim=1) 

            # Setting 3: Additivity
            # X = (Img1, Img2), Y = (Masked Img1, Masked Img2)
            # We split the batch in half to get "independent" pairs
            elif setting == 3:
                if B % 2 != 0: continue # Skip odd batches
                half = B // 2
                img1 = images[:half]
                img2 = images[half:]
                X = torch.cat([code(discriminator, img1, type='lmi'), code(discriminator, img2, type='lmi')], dim=1)
                
                y1 = mask_bottom_rows(img1, t, img_shape)
                y2 = mask_bottom_rows(img2, t, img_shape)
                Y = torch.cat([code(discriminator, y1, type='lmi'), code(discriminator, y2, type='lmi')], dim=1)
                
                # X = code(discriminator, X, type='lmi')
                # Y = code(discriminator, Y, type='lmi')

            # --- MINE LOSS CALCULATION ---
            # Forward pass joint: T(x, y)
            joint_t = model(X, Y)
            
            # Forward pass marginal: T(x, y_shuffled)
            # Shuffle Y to break dependency
            Y_shuffle = Y[torch.randperm(Y.size(0))]
            marginal_t = model(X, Y_shuffle)

            # MINE Loss (Lower Bound maximization) = -(E[T] - log(E[e^T]))
            # loss = -(torch.mean(joint_t) - torch.log(torch.mean(torch.exp(marginal_t))))
            # use the logsunexp trick for numerical stability
            loss = -(torch.mean(joint_t) - torch.logsumexp(marginal_t, dim=0) + np.log(B))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # The MI estimate is the negative loss
            epoch_mi.append(-loss.item())

        avg_mi = np.mean(epoch_mi)
        # print(f"Setting {setting} | t={t} | Epoch {epoch+1} | MI: {avg_mi:.4f}")
        mi_est_history.append(avg_mi)

    # Return the average MI of the last few epochs for stability
    return np.mean(mi_est_history[-5:])

# --- 5. Main Experiment Execution ---

def run_experiment():
    print(f"Running Consistency Test on {DATASET_NAME}...")
    loader, input_dim, img_shape, total_rows = get_dataloader(DATASET_NAME, BATCH_SIZE)
    
    # --- Step A: Calculate I(X; X) for Normalization ---
    print(f"Estimating I(X; X) for Normalization (using t={total_rows})...")
    mi_self = estimate_mi(loader, input_dim, img_shape, setting=1, t=total_rows)
    print(f"Estimated I(X; X) = {mi_self:.4f}\n")
    
    results = {1: [], 2: [], 3: []}

    for t in ROWS_TO_TEST:
        print(f"\n--- Testing with {t} rows ---")
        
        # Setting 1: Baseline
        mi_1 = estimate_mi(loader, input_dim, img_shape, setting=1, t=t)
        results[1].append(mi_1)
        print(f"Setting 1 (Baseline) MI: {mi_1:.4f}")
        
        # Setting 2: Data Processing (Copies)
        # Theoretically: I((X,X); (Y,Y)) should contain same info as I(X;Y)
        mi_2 = estimate_mi(loader, input_dim, img_shape, setting=2, t=t)
        results[2].append(mi_2)
        print(f"Setting 2 (Same Img) MI: {mi_2:.4f}")

        # Setting 3: Additivity (Independent)
        # Theoretically: I((X1,X2); (Y1,Y2)) should be approx 2 * I(X;Y)
        mi_3 = estimate_mi(loader, input_dim, img_shape, setting=3, t=t)
        results[3].append(mi_3)
        print(f"Setting 3 (Diff Img) MI: {mi_3:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(ROWS_TO_TEST, results[1], 'o-', label='Setting 1: Baseline I(X; Y)')
    plt.plot(ROWS_TO_TEST, results[2], 's--', label='Setting 2: Same Image (Data Proc)')
    plt.plot(ROWS_TO_TEST, results[3], '^-.', label='Setting 3: Indep Images (Additivity)')
    
    plt.title(f'Self-Consistency Test on {DATASET_NAME} (Estimator: MINE)')
    plt.xlabel('Number of Rows (t)')
    plt.ylabel('Estimated Mutual Information (nats)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'consistency_result_{DATASET_NAME}.png')
    plt.show()

if __name__ == "__main__":
    run_experiment()