import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
from PIL import Image

# --- GPU Check ---
if torch.cuda.is_available():
    print("✅ GPU is available:", torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print("❌ GPU not available. Using CPU.")
    device = torch.device('cpu')

# --- Configurations ---
DATA_DIR = "maskrcnn_ready_dataset"
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_ANNOTATIONS_PATH = os.path.join(DATA_DIR, "train", "annotations", "instances_train.json")
VAL_IMAGE_DIR = os.path.join(DATA_DIR, "val", "images")
VAL_ANNOTATIONS_PATH = os.path.join(DATA_DIR, "val", "annotations", "instances_val.json")
NUM_CLASSES = 4  # background + 3 classes
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "maskrcnn_rice_disease.pth"

# --- Transform ---
class CocoTransform:
    def __init__(self):
        self.transforms = T.ToTensor()

    def __call__(self, image, target):
        image = self.transforms(image)
        return image, target

# --- Fixed Dataset Wrapper with Safe Annotation Handling ---
class CocoDetectionMaskWrapper(CocoDetection):
    def __getitem__(self, idx):
        original_idx = idx
        attempts = 0
        max_attempts = len(self.ids)  # Prevent infinite loop

        while attempts < max_attempts:
            try:
                img, _ = super().__getitem__(idx)
                img_id = self.ids[idx]
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)

                # If no valid annotations, move to next
                if not anns:
                    idx = (idx + 1) % len(self.ids)
                    attempts += 1
                    continue

                # Process annotations
                boxes = []
                labels = []
                masks = []

                for ann in anns:
                    # Extra check: Skip invalid bbox
                    if 'bbox' not in ann or ann['bbox'] is None:
                        continue
                    x_min, y_min, width, height = ann['bbox']
                    if width <= 0 or height <= 0:
                        continue  # Skip invalid boxes
                    x_max = x_min + width
                    y_max = y_min + height
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(ann['category_id'])

                    mask = self.coco.annToMask(ann)
                    masks.append(mask)

                # If after filtering there are no boxes
                if len(boxes) == 0:
                    idx = (idx + 1) % len(self.ids)
                    attempts += 1
                    continue

                # Convert to tensors
                masks = torch.as_tensor(masks, dtype=torch.uint8) if len(masks) else torch.zeros((0, img.height, img.width), dtype=torch.uint8)

                target = {
                    'boxes': torch.tensor(boxes, dtype=torch.float32),
                    'labels': torch.tensor(labels, dtype=torch.int64),
                    'masks': masks,
                    'image_id': torch.tensor([img_id]),
                    'area': torch.tensor([ann["area"] for ann in anns], dtype=torch.float32),
                    'iscrowd': torch.tensor([ann.get("iscrowd", 0) for ann in anns], dtype=torch.int64)
                }

                img, target = CocoTransform()(img, target)
                return img, target

            except Exception as e:
                print(f"Error loading index {idx}: {str(e)}")
                idx = (idx + 1) % len(self.ids)
                attempts += 1

        # Instead of crashing the whole training, return dummy sample
        print(f"⚠️ WARNING: No valid annotations found after {max_attempts} attempts. Returning dummy sample.")
        dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32)
        dummy_target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'masks': torch.zeros((0, 224, 224), dtype=torch.uint8),
            'image_id': torch.tensor([-1]),
            'area': torch.tensor([], dtype=torch.float32),
            'iscrowd': torch.tensor([], dtype=torch.int64)
        }
        return dummy_image, dummy_target

# --- Dataset and DataLoader ---
train_dataset = CocoDetectionMaskWrapper(TRAIN_IMAGE_DIR, TRAIN_ANNOTATIONS_PATH)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)),
    num_workers=2
)

# --- Model Setup ---
weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
model = maskrcnn_resnet50_fpn(weights=weights)

# Modify classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, NUM_CLASSES)

# Modify mask head
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    in_features_mask, hidden_layer, NUM_CLASSES)

model.to(device)

# --- Optimizer Setup ---
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
print("🚀 Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Skip dummy samples
        images, targets = zip(*[(img, tgt) for img, tgt in zip(images, targets) if tgt['boxes'].numel() > 0])

        if len(images) == 0:
            continue  # Skip if batch becomes empty

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {losses.item():.4f}")

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {epoch_loss/len(train_loader):.4f}")

# --- Save Model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Training complete! Model saved to {MODEL_SAVE_PATH}")
