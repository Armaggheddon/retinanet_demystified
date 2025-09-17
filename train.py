from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from retinanet import RetinaNet, RaccoonDataset, collate_fn


#############################################
# Setup
#############################################

# Expects the raccoon_dataset directory to be in the same directory
# as this script.
THIS_PATH = Path(__file__).parent

PLOT_PATH = THIS_PATH / "plots"
PLOT_PATH.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = RaccoonDataset(THIS_PATH / "raccoon_dataset")
train_test_split = int(0.8 * len(dataset))
train_dataset, dataset_test = torch.utils.data.random_split(
    dataset, [train_test_split, len(dataset) - train_test_split],
    generator=torch.Generator().manual_seed(42)
)

train_dataloader = DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(
    dataset_test, batch_size=2, shuffle=False, collate_fn=collate_fn)


BACKBONE_NAME = "resnet18"

model = RetinaNet(
    backbone_name=BACKBONE_NAME, 
    pretrained_backbone=True,
    num_classes=len(dataset.class2idx))
model = model.to(DEVICE)

#############################################
# Hyperparameters
#############################################
# ### 4.1 Inference and Training
# RetinaNet is trained with stochastic gradient descent (SGD).
# ...
# Unless otherwise specified, all models are trained for 90k iterations with an 
# initial learning rate of 0.01, which is then divided by 10 at 60k and again 
# at 80k iterations.
# ...
# Weight decay of 0.0001 and momentum of 0.9 are used.
# ###

num_epochs = 20
learning_rate = 1e-2
num_classes = len(dataset.class2idx)

# optimizer = torch.optim.SGD(
#     model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer=optimizer, milestones=[60000, 80000], gamma=0.1)

# Using Adam optimizer for faster convergence in this small dataset
# as opposed to SGD used in the original paper for large datasets (e.g. COCO).
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#############################################
# Training/eval loop
#############################################

metrics = {
    "training": {
        "loss": [],
        "class_loss": [],
        "box_loss": [],
    },
    "evaluation": {
        "loss": [],
        "class_loss": [],
        "box_loss": [],
    }
}

def train_one_epoch(model, dataloader, device, optimizer, epoch_idx, total_epochs, log_every_n=10):
    model.train()
    epoch_loss = 0.0
    epoch_class_loss = 0.0
    epoch_box_loss = 0.0

    loss_metric = []
    class_loss_metric = []
    box_loss_metric = []

    for batch_idx, (images, targets) in enumerate(dataloader):
        images_on_device = images.to(device)
        targets_on_device = [
            {
                'boxes': t['boxes'].to(device), 
                'labels': t['labels'].to(device),
            } 
            for idx, t in enumerate(targets)
        ]

        optimizer.zero_grad()
        losses_dict = model(images_on_device, targets=targets_on_device)
        total_loss = losses_dict["loss"]
        class_loss = losses_dict["class_loss"]
        box_loss = losses_dict["box_loss"]

        total_loss.backward()

        # To keep the code simple to follow and understand, the convolutions
        # do not use any form of normalization (BatchNorm, GroupNorm, etc).
        # In practice, when training a model without normalization layers,
        # it is often beneficial to clip the gradients before performing
        # the optimization step.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_class_loss += class_loss.item()
        epoch_box_loss += box_loss.item()

        loss_metric.append(total_loss.item())
        class_loss_metric.append(class_loss.item())
        box_loss_metric.append(box_loss.item())

        if (batch_idx + 1) % log_every_n == 0:
            print(f"\r\tStep [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss: {total_loss.item():.4f}, "
                  f"Cls Loss: {class_loss.item():.4f}, "
                  f"Box Loss: {box_loss.item():.4f}", end="")
            
    print()

    metrics["training"]["loss"].append(loss_metric)
    metrics["training"]["class_loss"].append(class_loss_metric)
    metrics["training"]["box_loss"].append(box_loss_metric)

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    avg_epoch_class_loss = epoch_class_loss / len(train_dataloader)
    avg_epoch_box_loss = epoch_box_loss / len(train_dataloader)

    print(
        f"Epoch [{epoch_idx+1}/{total_epochs}], "
        f"Avg Loss: {avg_epoch_loss:.4f}, "
        f"Avg Cls Loss: {avg_epoch_class_loss:.4f}, "
        f"Avg Box Loss: {avg_epoch_box_loss:.4f}")

def evaluate(model, dataloader, device):
    model.eval()
    epoch_loss = 0.0
    epoch_class_loss = 0.0
    epoch_box_loss = 0.0

    loss_metric = []
    class_loss_metric = []
    box_loss_metric = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images_on_device = images.to(device)
            targets_on_device = [
                {
                    'boxes': t['boxes'].to(device), 
                    'labels': t['labels'].to(device),
                } 
                for idx, t in enumerate(targets)
            ]

            # Get losses
            losses_dict = model(images_on_device, targets=targets_on_device)
            total_loss = losses_dict["loss"]
            class_loss = losses_dict["class_loss"]
            box_loss = losses_dict["box_loss"]

            epoch_loss += total_loss.item()
            epoch_class_loss += class_loss.item()
            epoch_box_loss += box_loss.item()

            loss_metric.append(total_loss.item())
            class_loss_metric.append(class_loss.item())
            box_loss_metric.append(box_loss.item())

    metrics["evaluation"]["loss"].append(loss_metric)
    metrics["evaluation"]["class_loss"].append(class_loss_metric)
    metrics["evaluation"]["box_loss"].append(box_loss_metric)

    avg_epoch_loss = epoch_loss / len(dataloader)
    avg_epoch_class_loss = epoch_class_loss / len(dataloader)
    avg_epoch_box_loss = epoch_box_loss / len(dataloader)

    print(
        f"Eval: Avg Loss: {avg_epoch_loss:.4f}, "
        f"Avg Cls Loss: {avg_epoch_class_loss:.4f}, "
        f"Avg Box Loss: {avg_epoch_box_loss:.4f}"
    )


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
print("Starting training...")

for epoch in range(num_epochs):
    train_one_epoch(model, train_dataloader, DEVICE, optimizer, epoch, num_epochs)
    evaluate(model, test_dataloader, DEVICE)

    # lr_scheduler.step()
    
print("Training complete.")


#############################################
# Save the trained model
#############################################
MODEL_SAVENAME = f"retinanet_raccoon_rn"
if BACKBONE_NAME[-1] == "1" or BACKBONE_NAME[-1] == "2":
    # backbone is resnet101 or resnet152
    MODEL_SAVENAME += BACKBONE_NAME[-3:]
else:
    # backbone is resnet18, resnet34, or resnet50
    MODEL_SAVENAME += BACKBONE_NAME[-2:]

torch.save(model.state_dict(), THIS_PATH / f"{MODEL_SAVENAME}.pth")

#############################################
# Plot training metrics
#############################################
def plot_training_metrics(metrics: dict):
    steps_per_epoch = [len(l) for l in metrics["loss"]]
    # offset by one epochs to match metric collection
    epochs_to_steps = [
        steps_per_epoch[0] + sum(steps_per_epoch[:i]) 
        for i in range(len(steps_per_epoch))
    ]
    total_steps = sum(steps_per_epoch)
    steps = range(1, total_steps + 1)

    # Plot loss per step (clip to 1.5 for visibility), and average loss per epoch
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    all_loss = [min(loss, 1.5) for epoch_losses in metrics["loss"] for loss in epoch_losses]
    plt.plot(steps, all_loss, label="Loss per step", alpha=0.3)
    
    avg_epoch_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in metrics["loss"]]
    # scale epochs on x-axis to match steps
    plt.plot(epochs_to_steps, avg_epoch_loss, label="Avg Loss per epoch", marker='o')
    plt.xlabel("Steps / Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()   
    plt.grid()
    # Plot class loss per step, and average class loss per epoch
    plt.subplot(1, 3, 2)
    all_class_loss = [min(loss, 1.5) for epoch_losses in metrics["class_loss"] for loss in epoch_losses]
    plt.plot(steps, all_class_loss, label="Class Loss per step", alpha=0.3)
    avg_epoch_class_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in metrics["class_loss"]]
    plt.plot(epochs_to_steps, avg_epoch_class_loss, label="Avg Class Loss per epoch", marker='o')
    plt.xlabel("Steps / Epochs")
    plt.ylabel("Class Loss")
    plt.title("Training Class Loss")
    plt.legend()   
    plt.grid()
    # Plot box loss per step, and average box loss per epoch
    plt.subplot(1, 3, 3)
    all_box_loss = [min(loss, 1.5) for epoch_losses in metrics["box_loss"] for loss in epoch_losses]
    plt.plot(steps, all_box_loss, label="Box Loss per step", alpha=0.3)
    avg_epoch_box_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in metrics["box_loss"]]
    plt.plot(epochs_to_steps, avg_epoch_box_loss, label="Avg Box Loss per epoch", marker='o')
    plt.xlabel("Steps / Epochs")
    plt.ylabel("Box Loss")
    plt.title("Training Box Loss")
    plt.legend()
    plt.grid()

    plt.savefig(PLOT_PATH / "training_metrics.png")

plot_training_metrics(metrics["training"])

#############################################
# Plot evaluation metrics
#############################################
def plot_eval_metrics(metrics: dict):
    steps_per_epoch = [len(l) for l in metrics["loss"]]
    # offset by one epochs to match metric collection
    epochs_to_steps = [
        steps_per_epoch[0] + sum(steps_per_epoch[:i]) 
        for i in range(len(steps_per_epoch))
    ]
    total_steps = sum(steps_per_epoch)
    steps = range(1, total_steps + 1)

    # Plot loss per step (clip to 1.5 for visibility), and average loss per epoch
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    all_loss = [min(loss, 1.5) for epoch_losses in metrics["loss"] for loss in epoch_losses]
    plt.plot(steps, all_loss, label="Loss per step", alpha=0.3)
    
    avg_epoch_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in metrics["loss"]]
    # scale epochs on x-axis to match steps
    plt.plot(epochs_to_steps, avg_epoch_loss, label="Avg Loss per epoch", marker='o')
    plt.xlabel("Steps / Epochs")
    plt.ylabel("Loss")
    plt.title("Evaluation Loss")
    plt.legend()   
    plt.grid()
    # Plot class loss per step, and average class loss per epoch
    plt.subplot(1, 3, 2)
    all_class_loss = [min(loss, 1.5) for epoch_losses in metrics["class_loss"] for loss in epoch_losses]
    plt.plot(steps, all_class_loss, label="Class Loss per step", alpha=0.3)
    avg_epoch_class_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in metrics["class_loss"]]
    plt.plot(epochs_to_steps, avg_epoch_class_loss, label="Avg Class Loss per epoch", marker='o')
    plt.xlabel("Steps / Epochs")
    plt.ylabel("Class Loss")
    plt.title("Evaluation Class Loss")
    plt.legend()   
    plt.grid()
    # Plot box loss per step, and average box loss per epoch
    plt.subplot(1, 3, 3)
    all_box_loss = [min(loss, 1.5) for epoch_losses in metrics["box_loss"] for loss in epoch_losses]
    plt.plot(steps, all_box_loss, label="Box Loss per step", alpha=0.3)
    avg_epoch_box_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in metrics["box_loss"]]
    plt.plot(epochs_to_steps, avg_epoch_box_loss, label="Avg Box Loss per epoch", marker='o')
    plt.xlabel("Steps / Epochs")
    plt.ylabel("Box Loss")
    plt.title("Evaluation Box Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(PLOT_PATH / "evaluation_loss_metrics.png")

plot_eval_metrics(metrics["evaluation"])

#############################################
# Plot average train/eval loss per epoch
#############################################
def plot_avg_train_eval_loss(train_metrics: dict, eval_metrics: dict):
    epochs = range(1, len(train_metrics["loss"]) + 1)

    avg_train_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in train_metrics["loss"]]
    avg_eval_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in eval_metrics["loss"]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_train_loss, label="Avg Training Loss", marker='o')
    plt.plot(epochs, avg_eval_loss, label="Avg Evaluation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Average Training and Evaluation Loss per Epoch")
    plt.legend()   
    plt.grid()
    plt.savefig(PLOT_PATH / "train_eval_avg_loss.png")

plot_avg_train_eval_loss(metrics["training"], metrics["evaluation"])
