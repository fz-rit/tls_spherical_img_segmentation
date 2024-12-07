import segmentation_models_pytorch as smp
import torch
from prepare_dataset import load_data

config_file = 'params/paths_zmachine.json'
train_loader = load_data(config_file)

# Example: a UNet with ResNet34 encoder
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=5  # 
)

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = smp.losses.DiceLoss(mode='multiclass')

model.train()
for epoch in range(10):
    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
