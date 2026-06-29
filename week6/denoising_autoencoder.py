import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Hyperparameters
batch_size = 128
epochs = 5
learning_rate = 1e-3
noise_factor = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loading
# We will use the provided mnist_png dataset
data_dir = r"c:\Users\Bhumi\Downloads\week6\archive\mnist_png"
train_dir = os.path.join(data_dir, "training")
test_dir = os.path.join(data_dir, "testing")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

print("Loading datasets...")
try:
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
except Exception as e:
    print(f"Error loading dataset from {data_dir}. Make sure the paths are correct.")
    print(e)
    exit()

# Define the Denoising Autoencoder Model
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # (32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, 7) # (64, 1, 1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), # (32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # (1, 28, 28)
            nn.Sigmoid() # Images are in range [0, 1]
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training on device: {device}")

# Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        
        # Add random Gaussian noise to the images
        noisy_imgs = img + noise_factor * torch.randn(*img.shape).to(device)
        noisy_imgs = torch.clamp(noisy_imgs, 0., 1.) # Clip to maintain [0,1] range
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, img) # We want the model to output the CLEAN image
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the model
model_path = "denoising_autoencoder.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Visualization / Evaluation
model.eval()
dataiter = iter(test_loader)
images, _ = next(dataiter)

# Create noisy images
noisy_images = images + noise_factor * torch.randn(*images.shape)
noisy_images = torch.clamp(noisy_images, 0., 1.)

# Move to device and get outputs
noisy_images = noisy_images.to(device)
with torch.no_grad():
    outputs = model(noisy_images)

# Plot the results
images = images.cpu().numpy()
noisy_images = noisy_images.cpu().numpy()
outputs = outputs.cpu().numpy()

fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20, 6))

for i in range(10):
    # Original images
    axes[0, i].imshow(images[i].squeeze(), cmap='gray')
    axes[0, i].set_title("Original")
    axes[0, i].get_xaxis().set_visible(False)
    axes[0, i].get_yaxis().set_visible(False)
    
    # Noisy images
    axes[1, i].imshow(noisy_images[i].squeeze(), cmap='gray')
    axes[1, i].set_title("Noisy")
    axes[1, i].get_xaxis().set_visible(False)
    axes[1, i].get_yaxis().set_visible(False)
    
    # Denoised images
    axes[2, i].imshow(outputs[i].squeeze(), cmap='gray')
    axes[2, i].set_title("Denoised")
    axes[2, i].get_xaxis().set_visible(False)
    axes[2, i].get_yaxis().set_visible(False)

plt.tight_layout()
plot_path = "denoising_results.png"
plt.savefig(plot_path)
print(f"Results saved to {plot_path}")
# plt.show()
