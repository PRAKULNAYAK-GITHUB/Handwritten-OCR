import os, random, string, numpy as np, pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import editdistance
import warnings
import cv2

torch.backends.cudnn.benchmark = True
from torch.amp import autocast, GradScaler
scaler = GradScaler()

# ------------------ Config ------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/handwriting-recognitionocr"
WORK_DIR = "/kaggle/working/ocr_model"
os.makedirs(WORK_DIR, exist_ok=True)

MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
DISTORTION_MODEL_PATH = os.path.join(WORK_DIR, "best_distortion_model.pth")

# Image config
IMG_HEIGHT = 64
MAX_WIDTH = 256

# Training config
batch_size = 64
num_epochs = 20
distortion_epochs = 10  # Fine-tuning epochs with distortions
learning_rate = 3e-4
distortion_lr = 1e-4  # Lower LR for fine-tuning

# Pretrained options
USE_PRETRAINED_RESNET = False
LOCAL_RESNET_WEIGHTS = "/kaggle/input/resnet34-b627a593/resnet34-b627a593.pth"

# Distortion config
DISTORTION_PROBABILITY = 0.7  # Probability of applying distortions during fine-tuning

# ------------------ Charset ------------------
characters = "-" + " " + string.ascii_letters + string.digits
characters = ''.join(sorted(set(characters)))
char_to_idx = {c: i for i, c in enumerate(characters)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
num_classes = len(characters)

print(f"Vocabulary size: {num_classes} characters")
print(f"Characters sample: {characters[:50]}")

def normalize_text(t):
    return ''.join(c for c in str(t) if c in char_to_idx)

def text_to_indices(text):
    return [char_to_idx[c] for c in normalize_text(text)]

def indices_to_text(indices):
    chars, prev = [], None
    blank = char_to_idx['-']
    for idx in indices:
        if idx not in idx_to_char:
            prev = idx
            continue
        if idx != blank and idx != prev:
            chars.append(idx_to_char[idx])
        prev = idx
    return ''.join(chars)

# ------------------ Distortion Augmentation Functions ------------------
class DistortionAugmentation:
    """Advanced distortion techniques for handwriting robustness"""
    
    @staticmethod
    def apply_rotation(img, max_angle=15):
        """Random rotation within ±max_angle degrees"""
        angle = random.uniform(-max_angle, max_angle)
        return img.rotate(angle, fillcolor=255, expand=False)
    
    @staticmethod
    def apply_skew(img, max_skew=0.3):
        """Apply perspective skew (affine transform)"""
        w, h = img.size
        img_np = np.array(img)
        
        # Random skew parameters
        shear_x = random.uniform(-max_skew, max_skew)
        shear_y = random.uniform(-max_skew * 0.3, max_skew * 0.3)
        
        M = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
        skewed = cv2.warpAffine(img_np, M, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=255)
        return Image.fromarray(skewed)
    
    @staticmethod
    def apply_stretch(img, max_stretch=0.15):
        """Random horizontal/vertical stretch"""
        w, h = img.size
        scale_x = random.uniform(1 - max_stretch, 1 + max_stretch)
        scale_y = random.uniform(1 - max_stretch * 0.5, 1 + max_stretch * 0.5)
        new_w = max(16, int(w * scale_x))
        new_h = max(16, int(h * scale_y))
        return img.resize((new_w, new_h), Image.BILINEAR)
    
    @staticmethod
    def apply_noise(img, intensity=0.02):
        """Add Gaussian noise"""
        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(0, intensity * 255, img_np.shape)
        noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    @staticmethod
    def apply_blur(img, max_radius=1.5):
        """Apply Gaussian blur"""
        radius = random.uniform(0.3, max_radius)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    @staticmethod
    def apply_brightness(img, factor_range=(0.7, 1.3)):
        """Adjust brightness"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    @staticmethod
    def apply_contrast(img, factor_range=(0.8, 1.3)):
        """Adjust contrast"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    @staticmethod
    def apply_elastic_distortion(img, alpha=20, sigma=5):
        """Elastic deformation for handwriting variation"""
        img_np = np.array(img)
        h, w = img_np.shape
        
        # Create random displacement fields
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = (y + dy).astype(np.float32), (x + dx).astype(np.float32)
        
        distorted = cv2.remap(img_np, indices[1], indices[0], 
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)
        return Image.fromarray(distorted)
    
    @staticmethod
    def apply_random_distortions(img, probability=0.7):
        """Apply random combination of distortions"""
        if random.random() > probability:
            return img
        
        # Choose 2-4 random distortions
        num_distortions = random.randint(2, 4)
        distortion_funcs = [
            lambda x: DistortionAugmentation.apply_rotation(x, max_angle=12),
            lambda x: DistortionAugmentation.apply_skew(x, max_skew=0.25),
            lambda x: DistortionAugmentation.apply_stretch(x, max_stretch=0.12),
            lambda x: DistortionAugmentation.apply_noise(x, intensity=0.015),
            lambda x: DistortionAugmentation.apply_blur(x, max_radius=1.2),
            lambda x: DistortionAugmentation.apply_brightness(x),
            lambda x: DistortionAugmentation.apply_contrast(x),
            lambda x: DistortionAugmentation.apply_elastic_distortion(x, alpha=15, sigma=4),
        ]
        
        selected = random.sample(distortion_funcs, num_distortions)
        
        for func in selected:
            try:
                img = func(img)
            except Exception as e:
                warnings.warn(f"Distortion failed: {e}")
                continue
        
        return img

# ------------------ Enhanced Dataset ------------------
class HandwritingDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None, max_width=256, 
                 use_distortions=False, distortion_prob=0.7):
        self.df = pd.read_csv(csv_path).iloc[:, :2].copy()
        self.df.columns = ['filename', 'text']
        self.df['text'] = self.df['text'].fillna('').astype(str)
        self.df = self.df[self.df['text'].str.len() > 0].reset_index(drop=True)
       
        self.images_dir = images_dir
        self.transform = transform
        self.max_width = max_width
        self.use_distortions = use_distortions
        self.distortion_prob = distortion_prob
        self.augmenter = DistortionAugmentation()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['filename'])
        label = normalize_text(row['text'].strip())
       
        img_path = os.path.join(self.images_dir, img_name)
       
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            warnings.warn(f"Image load failed for {img_path}: {e}")
            image = Image.new('L', (self.max_width, IMG_HEIGHT), color=255)
            label = ""
        
        # Apply distortions BEFORE resizing (on original image)
        if self.use_distortions and len(label) > 0:
            image = self.augmenter.apply_random_distortions(image, self.distortion_prob)
        
        # Resize preserving aspect ratio
        w, h = image.size
        aspect = w / h if h != 0 else 1.0
        new_w = int(IMG_HEIGHT * aspect)
        new_w = max(16, min(new_w, self.max_width))
        image = image.resize((new_w, IMG_HEIGHT), Image.BILINEAR)
       
        # Convert to tensor
        tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        c, hh, ww = tensor.shape
       
        # Pad to max_width
        if ww < self.max_width:
            pad = torch.zeros((c, hh, self.max_width - ww))
            tensor = torch.cat([tensor, pad], 2)
        elif ww > self.max_width:
            tensor = tensor[:, :, :self.max_width]
       
        return tensor, label, img_name

# ------------------ CRNN Model ------------------
class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, use_pretrained=False, local_weights_path=None):
        super().__init__()
       
        if use_pretrained:
            try:
                resnet = models.resnet34(weights=None)
                if local_weights_path and os.path.exists(local_weights_path):
                    state = torch.load(local_weights_path, map_location='cpu')
                    resnet.load_state_dict(state)
                    print("Loaded ResNet34 weights from local file.")
                else:
                    warnings.warn("Requested pretrained ResNet but local weights not found.")
            except Exception as e:
                warnings.warn(f"Failed to load local pretrained ResNet34 ({e}).")
                resnet = models.resnet34(weights=None)
        else:
            resnet = models.resnet34(weights=None)
       
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        try:
            with torch.no_grad():
                if hasattr(resnet, 'conv1'):
                    w = resnet.conv1.weight
                    if w.shape[1] == 3:
                        self.conv1.weight = nn.Parameter(w.mean(dim=1, keepdim=True))
                    elif w.shape[1] == 1:
                        self.conv1.weight = nn.Parameter(w)
                    else:
                        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        except Exception:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
       
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
       
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
       
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
       
        x = F.adaptive_avg_pool2d(x, (1, None))
        x = x.squeeze(2)
        x = x.permute(0, 2, 1).contiguous()
       
        x, _ = self.rnn(x)
        x = self.fc(x)
       
        return F.log_softmax(x, dim=2)

# ------------------ Training Functions ------------------
def make_ctc_targets(texts, device):
    labels_list = [torch.tensor(text_to_indices(t), dtype=torch.long) for t in texts if len(t) > 0]
    if len(labels_list) == 0:
        return torch.tensor([], dtype=torch.long, device=device), torch.tensor([0], dtype=torch.long, device=device)
    label_lengths = torch.tensor([len(l) for l in labels_list], dtype=torch.long, device=device)
    labels = torch.cat(labels_list).to(device)
    return labels, label_lengths

criterion = nn.CTCLoss(blank=char_to_idx['-'], zero_infinity=True)

def train_epoch(model, loader, optimizer, scheduler, device, epoch, stage="Normal"):
    model.train()
    total_loss = 0.0
    num_batches = 0
   
    pbar = tqdm(loader, desc=f"{stage} Epoch {epoch:02d}")
    for imgs, texts, _ in pbar:
        valid_indices = [i for i, t in enumerate(texts) if len(t) > 0]
        if len(valid_indices) == 0:
            continue
        imgs = imgs[valid_indices].to(device)
        texts = [texts[i] for i in valid_indices]
       
        labels, label_lengths = make_ctc_targets(texts, device)
        if labels.numel() == 0:
            continue
       
        optimizer.zero_grad()
       
        log_probs = model(imgs)
        log_probs_ctc = log_probs.permute(1, 0, 2)
        input_lengths = torch.full((log_probs.size(0),), log_probs.size(1), dtype=torch.long, device=device)
       
        loss = criterion(log_probs_ctc, labels, input_lengths, label_lengths)
       
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            try:
                scheduler.step()
            except Exception:
                pass
           
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
   
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_cer = 0.0
    total_correct = 0
    count = 0
   
    for imgs, texts, _ in tqdm(loader, desc="Validating", leave=False):
        imgs = imgs.to(device)
        log_probs = model(imgs)
        preds = log_probs.argmax(dim=2).cpu().numpy()
       
        for pred, truth in zip(preds, texts):
            if len(truth) == 0:
                continue
            pred_text = indices_to_text(pred)
            truth_norm = normalize_text(truth)
           
            if pred_text == truth_norm:
                total_correct += 1
           
            cer = editdistance.eval(pred_text, truth_norm) / max(len(truth_norm), 1)
            total_cer += cer
            count += 1
   
    avg_cer = total_cer / max(count, 1)
    accuracy = total_correct / max(count, 1)
   
    return avg_cer, accuracy

# ------------------ Data Loading ------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_csv = os.path.join(DATA_ROOT, "CSV", "written_name_train.csv")
val_csv = os.path.join(DATA_ROOT, "CSV", "written_name_validation.csv")
train_images = os.path.join(DATA_ROOT, "train_v2", "train")
val_images = os.path.join(DATA_ROOT, "validation_v2", "validation")

print("\n" + "="*70)
print("📂 Loading datasets...")
print(f"   Train CSV: {os.path.exists(train_csv)}")
print(f"   Val CSV: {os.path.exists(val_csv)}")

# Normal training dataset (no distortions)
train_ds = HandwritingDataset(train_csv, train_images, transform, MAX_WIDTH, 
                               use_distortions=False)
val_ds = HandwritingDataset(val_csv, val_images, transform, MAX_WIDTH, 
                             use_distortions=False)

# Distortion-aware dataset for fine-tuning
train_ds_distorted = HandwritingDataset(train_csv, train_images, transform, MAX_WIDTH,
                                         use_distortions=True, 
                                         distortion_prob=DISTORTION_PROBABILITY)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                        num_workers=2, pin_memory=True)
train_loader_distorted = DataLoader(train_ds_distorted, batch_size=batch_size, 
                                   shuffle=True, num_workers=2, pin_memory=True)

print(f"✅ Train samples: {len(train_ds)}")
print(f"✅ Val samples: {len(val_ds)}")
print("="*70 + "\n")

# ------------------ Stage 1: Normal Training ------------------
print("="*70)
print("🚀 STAGE 1: NORMAL TRAINING")
print("="*70)

model = CRNN(num_classes, use_pretrained=USE_PRETRAINED_RESNET, 
             local_weights_path=LOCAL_RESNET_WEIGHTS).to(device)

if os.path.exists(MODEL_PATH):
    print("✅ Loading saved base model...\n")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("🔧 Training base model...\n")
   
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate * 3, epochs=num_epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.1
    )
   
    best_accuracy = 0.0
    patience, patience_counter = 5, 0
   
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'─'*70}")
        print(f"📅 Epoch {epoch}/{num_epochs}")
        print(f"{'─'*70}")
       
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_cer, val_acc = validate(model, val_loader, device)
       
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val CER: {val_cer*100:.2f}%")
        print(f"   Val Accuracy: {val_acc*100:.2f}%")
       
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"   ✅ Best model saved! Accuracy: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⚠️  Early stopping triggered")
                break
   
    print(f"\n{'='*70}")
    print(f"✅ Stage 1 complete! Best accuracy: {best_accuracy*100:.2f}%")
    print(f"{'='*70}\n")
   
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# ------------------ Stage 2: Distortion-Aware Fine-Tuning ------------------
print("\n" + "="*70)
print("🎯 STAGE 2: DISTORTION-AWARE FINE-TUNING")
print("="*70)
print(f"   Distortion probability: {DISTORTION_PROBABILITY*100:.0f}%")
print(f"   Fine-tuning epochs: {distortion_epochs}")
print(f"   Learning rate: {distortion_lr}")
print("="*70 + "\n")

if os.path.exists(DISTORTION_MODEL_PATH):
    print("✅ Loading saved distortion-aware model...\n")
    model.load_state_dict(torch.load(DISTORTION_MODEL_PATH, map_location=device))
else:
    print("🔧 Fine-tuning with distortions...\n")
   
    # Lower learning rate for fine-tuning
    optimizer_ft = optim.AdamW(model.parameters(), lr=distortion_lr, weight_decay=1e-5)
    steps_per_epoch = max(1, len(train_loader_distorted))
    scheduler_ft = optim.lr_scheduler.OneCycleLR(
        optimizer_ft, max_lr=distortion_lr * 2, epochs=distortion_epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.2
    )
   
    best_distortion_accuracy = 0.0
    patience_ft, patience_counter_ft = 4, 0
   
    for epoch in range(1, distortion_epochs + 1):
        print(f"\n{'─'*70}")
        print(f"📅 Fine-tune Epoch {epoch}/{distortion_epochs}")
        print(f"{'─'*70}")
       
        train_loss = train_epoch(model, train_loader_distorted, optimizer_ft, 
                                scheduler_ft, device, epoch, stage="Distortion")
        val_cer, val_acc = validate(model, val_loader, device)
       
        print(f"\n📊 Results:")
        print(f"   Train Loss (distorted): {train_loss:.4f}")
        print(f"   Val CER: {val_cer*100:.2f}%")
        print(f"   Val Accuracy: {val_acc*100:.2f}%")
       
        if val_acc > best_distortion_accuracy:
            best_distortion_accuracy = val_acc
            patience_counter_ft = 0
            torch.save(model.state_dict(), DISTORTION_MODEL_PATH)
            print(f"   ✅ Best distortion-aware model saved! Accuracy: {val_acc*100:.2f}%")
        else:
            patience_counter_ft += 1
            if patience_counter_ft >= patience_ft:
                print(f"   ⚠️  Early stopping triggered")
                break
   
    print(f"\n{'='*70}")
    print(f"✅ Stage 2 complete! Best accuracy: {best_distortion_accuracy*100:.2f}%")
    print(f"{'='*70}\n")
   
    if os.path.exists(DISTORTION_MODEL_PATH):
        model.load_state_dict(torch.load(DISTORTION_MODEL_PATH, map_location=device))

# ------------------ Prediction Function ------------------
@torch.no_grad()
def predict_image(image_path, model, device, show_details=True):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
   
    if show_details:
        print(f"\n{'─'*70}")
        print(f"📸 Processing: {os.path.basename(image_path)}")
   
    image = Image.open(image_path).convert('L')
    orig_w, orig_h = image.size
    if show_details:
        print(f"   Original size: {orig_w}x{orig_h}")
   
    aspect = orig_w / orig_h if orig_h != 0 else 1.0
    new_w = int(IMG_HEIGHT * aspect)
    new_w = max(16, min(new_w, MAX_WIDTH))
    image = image.resize((new_w, IMG_HEIGHT), Image.BILINEAR)
    if show_details:
        print(f"   Resized to: {new_w}x{IMG_HEIGHT}")
   
    tensor = transform(image).unsqueeze(0).to(device)
   
    if tensor.shape[3] < MAX_WIDTH:
        pad = torch.zeros((1, 1, IMG_HEIGHT, MAX_WIDTH - tensor.shape[3]), device=device)
        tensor = torch.cat([tensor, pad], 3)
   
    model.eval()
    log_probs = model(tensor)
    pred = log_probs.argmax(dim=2).squeeze(0).cpu().numpy()
    result = indices_to_text(pred)
    confidence = log_probs.exp().max(dim=2)[0].mean().item()
   
    if show_details:
        print(f"   ✅ Prediction: '{result}'")
        print(f"   📊 Confidence: {confidence*100:.2f}%")
        print(f"{'─'*70}")
   
    return {
        'text': result,
        'confidence': confidence,
        'original_size': (orig_w, orig_h),
        'processed_size': (new_w, IMG_HEIGHT)
    }

# ------------------ Robustness Test ------------------
def test_robustness(model, val_loader, device, num_samples=50):
    """Test model on distorted validation samples"""
    print("\n" + "="*70)
    print("🧪 ROBUSTNESS TEST (Distorted Validation Samples)")
    print("="*70)
   
    model.eval()
    correct, total = 0, 0
    augmenter = DistortionAugmentation()
   
    sample_indices = random.sample(range(len(val_ds)), min(num_samples, len(val_ds)))
   
    for idx in tqdm(sample_indices, desc="Testing robustness"):
        _, true_label, img_name = val_ds[idx]
        img_path = os.path.join(val_images, img_name)
       
        try:
            image = Image.open(img_path).convert('L')
            # Apply heavy distortions
            image = augmenter.apply_random_distortions(image, probability=1.0)
           
            # Save temporarily and predict
            temp_path = "/tmp/test_distorted.png"
            image.save(temp_path)
            result = predict_image(temp_path, model, device, show_details=False)
           
            if result and result['text'].lower() == true_label.lower():
                correct += 1
            total += 1
        except Exception as e:
            warnings.warn(f"Robustness test failed for {img_name}: {e}")
   
    accuracy = correct / max(total, 1) * 100
    print(f"\n📊 Robustness Test Results:")
    print(f"   Correct: {correct}/{total}")
    print(f"   Accuracy on Distorted Images: {accuracy:.2f}%")
    print("="*70 + "\n")

# Run robustness test
test_robustness(model, val_loader, device, num_samples=50)

# ------------------ Interactive Menu ------------------
print("\n" + "="*70)
print("✅ DISTORTION-AWARE MODEL READY!")
print("="*70)
print("📌 Robust to rotation, skew, noise, blur, and elastic distortions")
print("📌 Two-stage training: normal → distortion-aware fine-tuning")
print("="*70 + "\n")

while True:
    print("\n" + "="*70)
    print("🔍 OCR TESTING MENU")
    print("="*70)
    print("1. Test single image")
    print("2. Test on validation sample")
    print("3. Test with artificial distortions")
    print("4. Run robustness benchmark")
    print("5. Compare base vs distortion-aware model")
    print("6. Exit")
    print("="*70)
   
    try:
        choice = input("Enter choice (1-6): ").strip()
    except Exception:
        print("Non-interactive environment detected — exiting.")
        break
   
    if choice == "1":
        path = input("\n📂 Enter image path: ").strip()
        result = predict_image(path, model, device)
        if result:
            print(f"\n✨ RESULT: '{result['text']}'")
   
    elif choice == "2":
        # Test on random validation sample
        sample_idx = random.randint(0, len(val_ds) - 1)
        _, true_label, img_name = val_ds[sample_idx]
        img_path = os.path.join(val_images, img_name)
       
        print(f"\n📝 Ground truth: '{true_label}'")
        result = predict_image(img_path, model, device)
       
        if result:
            match = "✅ CORRECT" if result['text'].lower() == true_label.lower() else "❌ WRONG"
            print(f"\n{match}")
   
    elif choice == "3":
        # Test with artificial distortions
        sample_idx = random.randint(0, len(val_ds) - 1)
        _, true_label, img_name = val_ds[sample_idx]
        img_path = os.path.join(val_images, img_name)
       
        print(f"\n📝 Ground truth: '{true_label}'")
        print("🎨 Applying random distortions...\n")
       
        try:
            image = Image.open(img_path).convert('L')
            augmenter = DistortionAugmentation()
            distorted = augmenter.apply_random_distortions(image, probability=1.0)
           
            # Save temporarily
            temp_path = "/tmp/test_distorted.png"
            distorted.save(temp_path)
           
            print("Original image:")
            result_orig = predict_image(img_path, model, device)
           
            print("\nDistorted image:")
            result_dist = predict_image(temp_path, model, device)
           
            if result_orig and result_dist:
                print(f"\n{'='*70}")
                print(f"📊 COMPARISON:")
                print(f"   Ground truth:  '{true_label}'")
                print(f"   Original:      '{result_orig['text']}' (conf: {result_orig['confidence']*100:.1f}%)")
                print(f"   Distorted:     '{result_dist['text']}' (conf: {result_dist['confidence']*100:.1f}%)")
                
                orig_correct = result_orig['text'].lower() == true_label.lower()
                dist_correct = result_dist['text'].lower() == true_label.lower()
                
                if orig_correct and dist_correct:
                    print(f"   ✅ Model is ROBUST to distortions!")
                elif orig_correct and not dist_correct:
                    print(f"   ⚠️  Model struggled with distortions")
                else:
                    print(f"   ❌ Base prediction was incorrect")
                print(f"{'='*70}")
       
        except Exception as e:
            print(f"❌ Error: {e}")
   
    elif choice == "4":
        # Run comprehensive robustness test
        num_samples = input("\n📊 Number of samples to test (default 50): ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 50
        test_robustness(model, val_loader, device, num_samples=num_samples)
   
    elif choice == "5":
        # Compare base model vs distortion-aware model
        if not os.path.exists(MODEL_PATH):
            print("\n❌ Base model not found. Train it first.")
            continue
       
        print("\n" + "="*70)
        print("⚖️  COMPARISON: Base vs Distortion-Aware Model")
        print("="*70)
       
        # Load base model
        base_model = CRNN(num_classes).to(device)
        base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
       
        # Test on normal validation set
        print("\n📋 Testing on normal validation set...")
        base_cer, base_acc = validate(base_model, val_loader, device)
        dist_cer, dist_acc = validate(model, val_loader, device)
       
        print(f"\n{'─'*70}")
        print(f"Normal Validation Results:")
        print(f"   Base Model:        CER={base_cer*100:.2f}%  Acc={base_acc*100:.2f}%")
        print(f"   Distortion-Aware:  CER={dist_cer*100:.2f}%  Acc={dist_acc*100:.2f}%")
        print(f"{'─'*70}")
       
        # Test on distorted samples
        print(f"\n📋 Testing on distorted samples (n=30)...")
        
        base_correct, dist_correct, total = 0, 0, 0
        augmenter = DistortionAugmentation()
        
        test_indices = random.sample(range(len(val_ds)), min(30, len(val_ds)))
        
        for idx in tqdm(test_indices, desc="Testing"):
            _, true_label, img_name = val_ds[idx]
            img_path = os.path.join(val_images, img_name)
           
            try:
                image = Image.open(img_path).convert('L')
                distorted = augmenter.apply_random_distortions(image, probability=1.0)
               
                temp_path = "/tmp/compare_distorted.png"
                distorted.save(temp_path)
               
                # Test base model
                result_base = predict_image(temp_path, base_model, device, show_details=False)
                if result_base and result_base['text'].lower() == true_label.lower():
                    base_correct += 1
               
                # Test distortion-aware model
                result_dist = predict_image(temp_path, model, device, show_details=False)
                if result_dist and result_dist['text'].lower() == true_label.lower():
                    dist_correct += 1
               
                total += 1
            except Exception:
                continue
       
        print(f"\n{'─'*70}")
        print(f"Distorted Validation Results:")
        print(f"   Base Model:        {base_correct}/{total} ({base_correct/max(total,1)*100:.1f}%)")
        print(f"   Distortion-Aware:  {dist_correct}/{total} ({dist_correct/max(total,1)*100:.1f}%)")
        print(f"   Improvement:       +{(dist_correct-base_correct)/max(total,1)*100:.1f}%")
        print(f"{'─'*70}")
       
        if dist_correct > base_correct:
            print(f"\n✅ Distortion-aware model is MORE ROBUST!")
        elif dist_correct == base_correct:
            print(f"\n➖ Both models performed similarly on distorted data")
        else:
            print(f"\n⚠️  Base model performed better (unexpected)")
       
        print(f"{'='*70}")
   
    elif choice == "6":
        print("\n👋 Goodbye!")
        break
   
    else:
        print("\n❌ Invalid choice. Please enter 1-6.")

print("\n" + "="*70)
print("✨ DISTORTION-AWARE OCR TRAINING COMPLETE")
print("="*70)
print(f"📁 Base model saved: {MODEL_PATH}")
print(f"📁 Distortion-aware model: {DISTORTION_MODEL_PATH}")
print("="*70)