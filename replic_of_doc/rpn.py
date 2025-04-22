import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T

# === 1. RPN MODEL ===
class RPN(nn.Module):
    def __init__(self, in_channels=3, mid_channels=256, num_anchors=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_regs = self.bbox_pred(x)
        return logits, bbox_regs

# === 2. ANCHOR GENERATOR ===
def generate_anchors(feature_size, ratios=[0.5, 1, 2], scales=[64, 128, 256], stride=16):
    anchors = []
    for y in range(feature_size):
        for x in range(feature_size):
            cx, cy = x * stride, y * stride
            for scale in scales:
                for ratio in ratios:
                    w = math.sqrt(scale * scale / ratio)
                    h = w * ratio
                    anchors.append([
                        cx - w / 2, cy - h / 2,
                        cx + w / 2, cy + h / 2
                    ])
    return torch.tensor(anchors)

# === 3. IoU CALCULATION ===
def compute_iou(boxes1, boxes2):
    N = boxes1.size(0)
    M = boxes2.size(0)
    boxes1 = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2 = boxes2.unsqueeze(0).expand(N, M, 4)
    inter_x1 = torch.max(boxes1[:, :, 0], boxes2[:, :, 0])
    inter_y1 = torch.max(boxes1[:, :, 1], boxes2[:, :, 1])
    inter_x2 = torch.min(boxes1[:, :, 2], boxes2[:, :, 2])
    inter_y2 = torch.min(boxes1[:, :, 3], boxes2[:, :, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
    area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
    union = area1 + area2 - inter_area + 1e-6
    return inter_area / union

# === 4. DELTA ENCODING ===
def bbox_transform(anchors, gt_boxes):
    anchors_w = anchors[:, 2] - anchors[:, 0]
    anchors_h = anchors[:, 3] - anchors[:, 1]
    anchors_cx = anchors[:, 0] + 0.5 * anchors_w
    anchors_cy = anchors[:, 1] + 0.5 * anchors_h
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_h
    dx = (gt_cx - anchors_cx) / anchors_w
    dy = (gt_cy - anchors_cy) / anchors_h
    dw = torch.log(gt_w / anchors_w)
    dh = torch.log(gt_h / anchors_h)
    return torch.stack([dx, dy, dw, dh], dim=1)

# === 5. CUSTOM DATASET ===
class CustomBoxDataset(Dataset):
    def __init__(self, root_dir):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.image_filenames = sorted([
            fname for fname in os.listdir(self.image_dir)
            if fname.endswith(".jpg") or fname.endswith(".png")
        ])
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        width, height = 224, 224
        label_path = os.path.join(self.label_dir, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
        boxes = []
        if not os.path.exists(label_path):
            print(f"[DEBUG] Label file missing: {label_path}")
        else:
            print(f"[DEBUG] Reading {label_path}")
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"[DEBUG] EMPTY label file: {label_path}")
                for line in lines:
                    print(f"[DEBUG] line: {line.strip()}")
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        _, x_c, y_c, w, h = parts
                        x_c *= width
                        y_c *= height
                        w *= width
                        h *= height
                        xmin = x_c - w / 2
                        ymin = y_c - h / 2
                        xmax = x_c + w / 2
                        ymax = y_c + h / 2
                        boxes.append([xmin, ymin, xmax, ymax])
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        print(f"[DEBUG] Final parsed boxes shape: {boxes.shape}")
        return image, boxes


# === 6. LOSS FUNCTION ===
def rpn_loss(objectness, pred_deltas, labels, bbox_targets):
    cls_loss = F.binary_cross_entropy_with_logits(objectness, labels)
    reg_loss = F.smooth_l1_loss(pred_deltas, bbox_targets)
    return cls_loss + reg_loss

# === 7. TRAIN LOOP ===
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RPN().to(device)
    dataset = CustomBoxDataset("/content/drive/MyDrive/replic_of_doc/dataset")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    stride = 16
    feature_map_size = 224 // stride
    anchors = generate_anchors(feature_map_size, scales=[16, 32, 64]).to(device)  # smaller anchors

    best_loss = float('inf')
    for epoch in range(100):
        total_loss = 0.0
        total_correct = 0
        total_preds = 0
        model.train()
        for images, gt_boxes in loader:
            images = images.to(device)
            gt_boxes = [b.to(device) for b in gt_boxes]
            optimizer.zero_grad()
            logits, bbox_regs = model(images)
            B, A, H, W = logits.shape
            objectness = logits.permute(0, 2, 3, 1).reshape(B, -1)
            bbox_regs = bbox_regs.permute(0, 2, 3, 1).reshape(B, -1, 4)
            losses = []
            for b in range(B):
                if gt_boxes[b].numel() == 0:
                    print(f"[WARN] No boxes for image {b}, skipping")
                    continue

                ious = compute_iou(anchors, gt_boxes[b])
                iou_max, iou_idx = ious.max(dim=1)
                labels = (iou_max > 0.5).float()
                matched_gt = gt_boxes[b][iou_idx]
                bbox_targets = bbox_transform(anchors, matched_gt)

                pos_count = (labels > 0).sum().item()
                print(f"Sample {b}: mean IoU = {iou_max.mean():.4f}, pos anchors = {pos_count}")

                if pos_count == 0:
                    continue

                loss = rpn_loss(objectness[b], bbox_regs[b], labels, bbox_targets)
                losses.append(loss)

                preds = torch.sigmoid(objectness[b]) > 0.5
                total_correct += (preds == labels).sum().item()
                total_preds += labels.numel()

            if losses:
                total = sum(losses) / len(losses)
                total.backward()
                optimizer.step()
                total_loss += total.item()

        accuracy = total_correct / total_preds if total_preds > 0 else 0
        print(f"Epoch {epoch+1}/100 - Loss: {total_loss:.4f} - Accuracy: {accuracy:.4f}")

        # Save best model
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), "best_rpn.pth")
            print("Saved best model with loss:", best_loss)


if __name__ == "__main__":
    train()
