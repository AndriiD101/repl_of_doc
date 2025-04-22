import torch

# --- GPU Check ---
if torch.cuda.is_available():
    print("✅ GPU is available:", torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print("❌ GPU not available. Using CPU.")
    device = torch.device('cpu')

import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# ------------------------------------------------------------------------------
# 1. Define the transformation that will be applied to each sample.
# ------------------------------------------------------------------------------
class CocoTransform:
    def __call__(self, image, target):
        # Ensure the image is a PIL Image.
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        # Convert PIL image to a tensor.
        image = F.to_tensor(image)
        return image, target

# ------------------------------------------------------------------------------
# 2. Helper function to create the dataset.
# ------------------------------------------------------------------------------
def get_coco_dataset(img_dir, ann_file):
    """
    img_dir: Directory with images.
    ann_file: JSON file with COCO-format annotations.
    """
    return CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=CocoTransform()
    )

# ------------------------------------------------------------------------------
# 3. Set up datasets and DataLoaders.
#    Update the paths below to match your dataset organization.
# ------------------------------------------------------------------------------
train_dataset = get_coco_dataset(
    img_dir="/content/drive/MyDrive/replic_of_doc/fasterrcnn_dataset/train",
    ann_file="/content/drive/MyDrive/replic_of_doc/fasterrcnn_dataset/annotations/instances_train.json"
)

val_dataset = get_coco_dataset(
    img_dir="/content/drive/MyDrive/replic_of_doc/fasterrcnn_dataset/val",
    ann_file="/content/drive/MyDrive/replic_of_doc/fasterrcnn_dataset/annotations/instances_val.json"
)

def collate_fn(batch):
    # Converts a list of tuples to a tuple of lists.
    return tuple(zip(*batch))

# Use the desired batch size.
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# ------------------------------------------------------------------------------
# 4. Define the model.
# ------------------------------------------------------------------------------
def get_model(num_classes):
    """
    Loads a pre-trained Faster R-CNN model and replaces its head
    to match the number of classes in the dataset.
    
    num_classes: Number of classes (including background).
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Update the number of classes (including background)
num_classes = 4
model = get_model(num_classes)

# ------------------------------------------------------------------------------
# 5. Move model to device, and set up the optimizer and learning rate scheduler.
# ------------------------------------------------------------------------------
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999))
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ------------------------------------------------------------------------------
# 6. Define the training loop for one epoch.
# ------------------------------------------------------------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        # Move images to the device.
        images = [img.to(device) for img in images]

        processed_targets = []
        valid_images = []

        # Convert each COCO bbox ([x, y, w, h]) to [x_min, y_min, x_max, y_max].
        for i, target in enumerate(targets):
            boxes = []
            labels = []
            for obj in target:
                bbox = obj["bbox"]
                x, y, w, h = bbox
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(obj["category_id"])
            if boxes:
                processed_target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                }
                processed_targets.append(processed_target)
                valid_images.append(images[i])

        if not processed_targets:
            continue

        loss_dict = model(valid_images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch}] Training Loss: {losses.item():.4f}")

# ------------------------------------------------------------------------------
# 7. Define the evaluation function.
#    NOTE: To compute losses, the model must be in training mode.
#    We'll temporarily set model to train() and disable gradients.
# ------------------------------------------------------------------------------
def evaluate(model, data_loader, device):
    # Force the model to training mode so it returns loss dictionaries.
    model.train()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]

            processed_targets = []
            valid_images = []

            for i, target in enumerate(targets):
                boxes = []
                labels = []
                for obj in target:
                    bbox = obj["bbox"]
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        boxes.append([x, y, x + w, y + h])
                        labels.append(obj["category_id"])
                if boxes:
                    processed_target = {
                        "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                        "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                    }
                    processed_targets.append(processed_target)
                    valid_images.append(images[i])
            if not processed_targets:
                continue
            loss_dict = model(valid_images, processed_targets)
            # Now loss_dict is a dictionary; we can sum the loss values.
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else float('inf')
    return avg_loss

# ------------------------------------------------------------------------------
# 8. Main training loop: Save the best model (based on validation loss) only.
# ------------------------------------------------------------------------------
def main():
    num_epochs = 100
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}] Validation Loss: {val_loss:.4f}")

        # Save the model only if validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Best model updated at epoch {epoch+1} with Validation Loss: {val_loss:.4f}")

    print("\nTraining complete.")
    print("Best validation loss:", best_val_loss)

if __name__ == "__main__":
    main()
