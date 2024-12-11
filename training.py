import segmentation_models_pytorch as smp
import torch
from prepare_dataset import load_data
import torch.nn as nn
from pathlib import Path
import json

class JointLoss(nn.Module):
    def __init__(self, first_loss, second_loss, first_weight=0.5, second_weight=0.5):
        super(JointLoss, self).__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.first_weight = first_weight
        self.second_weight = second_weight

    def forward(self, y_pred, y_true):
        loss1 = self.first_loss(y_pred, y_true)
        loss2 = self.second_loss(y_pred, y_true)
        return self.first_weight * loss1 + self.second_weight * loss2

def train_unet(config, save_model=False):
    _, train_loader, _ = load_data(config)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6, # Intensity-Range-Zvalue-nx-ny-nz 
        classes=5  
    )

    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = smp.losses.DiceLoss(mode='multiclass')
    loss_fn = JointLoss(
        smp.losses.DiceLoss(mode='multiclass'),
        torch.nn.CrossEntropyLoss(),
        first_weight=0.5,
        second_weight=0.5
    )
    epoch_num = 200
    model.train()
    for epoch in range(epoch_num):
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    if save_model:
        model_dir = Path(config['root_dir']) / config['model_dir']
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
            print(f"----Created directory {model_dir}----")
        save_model_path = model_dir / f'model_epoch{epoch_num:03d}.pth'
        torch.save(model.state_dict(), save_model_path)
        print(f"----Model saved at {save_model_path}----")

    return model

if __name__ == "__main__":
    config_file = 'params/paths_zmachine.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    model = train_unet(config, save_model=True)


