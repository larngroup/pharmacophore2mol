import torch
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

os.chdir(os.path.join(os.path.dirname(__file__), "."))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])



class RotationMNIST(datasets.MNIST):
    def __getitem__(self, index):
        # img, _ = super().__getitem__(index)
        if self.train:
            img, _ = super().__getitem__(0)
        else:
            img, _ = super().__getitem__(8) #they are both 5

        # Randomly rotate the image
        angle = torch.randint(0, 360, (1,)).item()
        img = transforms.functional.rotate(img, angle, fill=-1.0)
        # img = transforms.functional.center_crop(img, (19, 19))
        target = torch.tensor([torch.cos(torch.tensor(angle * 3.14 / 180)), torch.sin(torch.tensor(angle * 3.14 / 180))])
        return img, target
    
    # def __len__(self):
    #     return 1000
    

train_dataset = RotationMNIST(root='./saves', train=True, transform=transform)
test_dataset = RotationMNIST(root='./saves', train=False, transform=transform)


# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Print dataset shapes
print(f"Training data size: {len(train_dataset)}, Test data size: {len(test_dataset)}")

# Example of accessing a batch
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")

# # Make a grid of images and display them
# def show_image_grid(images):
#     # Denormalize images
#     images = images * 0.5 + 0.5  # Convert back to [0, 1]
#     grid = utils.make_grid(images, nrow=8)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(grid.permute(1, 2, 0))  # Convert from CHW to HWC
#     plt.axis('off')
#     plt.show()

# show_image_grid(images)


model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Flatten(),
    torch.nn.Linear(64 * 7 * 7, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(128, 2)
)

model = model.to(device)


def normalize_output(x, y, eps=1e-8):
    norm = torch.sqrt(x**2 + y**2 + eps)
    return x / norm, y / norm

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training loop
num_epochs = 100


# Training loop
for epoch in range(num_epochs):
    loop = tqdm(train_loader, leave=True)
    loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
    loop.set_postfix(loss=0)
    for i, (images, labels) in enumerate(loop):
        # Forward pass
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    #Validation
    with torch.no_grad():
        model.eval()
        val_loss = 0
        mae = 0
        val_loop = tqdm(test_loader, leave=True)
        val_loop.set_description(f"Validation [{epoch+1}/{num_epochs}]")
        for images, labels in val_loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item()

            normalized_outputs = normalize_output(outputs[:, 0], outputs[:, 1])
            pred_angles = torch.atan2(normalized_outputs[1], normalized_outputs[0]) * 180 / 3.14
            real_angles = torch.atan2(labels[:, 1], labels[:, 0]) * 180 / 3.14
            mae += torch.mean(torch.abs(pred_angles - real_angles))

            

        val_loss /= len(test_loader)
        mae /= len(test_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation MAE: {mae:.4f}")
        model.train()
