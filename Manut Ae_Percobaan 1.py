import os, torch, timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
# [IMPROVISASI] Impor scheduler WarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from timm.data import Mixup
import time

from torch.cuda.amp import GradScaler
# [FIX DEPRECATION] Ganti autocast lama
from torch.amp import autocast

from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- [IMPROVISASI] Parameter untuk Warm Restarts ---
IMAGE_SIZE = 384
BATCH_SIZE = 4   
EPOCHS = 1  # <-- [IMPROVISASI] Naikkan ke 75 untuk 5 siklus (5x15)
NUM_CLASSES = 15
LR = 1e-3
FINETUNE_LR = 2e-5  # <-- [IMPROVISASI] Naikkan sedikit agar "tendangan" lebih kuat
UNFREEZE_EPOCH = 5
# --------------------------------------------------

# --- Path (Tidak Berubah) ---
TRAIN_IMG_DIR = "train/train_cropped" 
TEST_IMG_DIR = "test/test_cropped"

FOOD_CLASSES = [
    'Ayam Bakar','Ayam Betutu','Ayam Goreng','Ayam Pop','Bakso',
    'Coto Makassar','Gado Gado','Gudeg','Nasi Goreng','Pempek',
    'Rendang','Rawon','Sate Madura','Sate Padang','Soto'
]

# --- Dataset (Tidak Berubah) ---
class FoodDataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        self.img_dir, self.df, self.transform = img_dir, df, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['ID']+'.jpg')
        label = FOOD_CLASSES.index(row['label'])
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="white")
        except Exception as e:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="white")

        if self.transform:
            img = self.transform(img)
            
        return img, label

# --- Transformasi (Ukuran 384px, Tidak Berubah) ---
IMAGENET_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # <-- 384px
    RandAugment(num_ops=2, magnitude=9),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), 
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    IMAGENET_NORM
])

test_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # <-- 384px
    transforms.ToTensor(),
    IMAGENET_NORM
])

# --- Helper UnNormalize (Tidak Berubah) ---
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        tensor = tensor.clone() 
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

IMAGENET_UNNORM = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --- create_model (Disederhanakan, ResNeSt dihapus) ---
def create_model(name):
    model = timm.create_model(name, pretrained=True, num_classes=NUM_CLASSES)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if 'convnext' in name or 'swin' in name:
        print(f"Menggunakan head '{name}.head' ...")
        model.head = nn.Sequential(nn.Dropout(0.5), model.head)
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Model type {name} tidak dikenali untuk modifikasi classifier.")
    
    return model.to(device)


# --- EarlyStopping (Tidak Berubah) ---
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    def __call__(self, val_acc, model, path):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"==> EarlyStopping terpicu! Tidak ada peningkatan selama {self.patience} epoch.")
        else:
            self.best_score = score
            torch.save(model.state_dict(), path)
            self.counter = 0

# --- Mixup (Tidak Berubah) ---
mixup_fn = Mixup(
    mixup_alpha=0.4, cutmix_alpha=0.4, label_smoothing=0.1,
    num_classes=NUM_CLASSES, prob=1.0, switch_prob=0.5, mode='batch'
)

# --- [IMPROVISASI UTAMA] train_model dengan Warm Restarts ---
def train_model(model, train_loader, val_loader, model_name):
    crit = nn.CrossEntropyLoss()
    print(f"--- Tahap 1: Melatih Head (Epoch 0-{UNFREEZE_EPOCH-1}) ---")
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    # Scheduler Tahap 1: Tetap CosineAnnealingLR
    scheduler = CosineAnnealingLR(opt, T_max=UNFREEZE_EPOCH)
    
    scaler = GradScaler()
    best_model_path = f"{model_name}_best.pth"
    
    # Patience Tahap 1: 5 sudah cukup
    early_stopping = EarlyStopping(patience=5) 
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []} 
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        if epoch == UNFREEZE_EPOCH:
            print(f"\n--- Tahap 2: Unfreezing Backbone di Epoch {epoch} ---")
            print(f"Memuat state model terbaik dari Tahap 1: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
            
            print(f"Mengganti LR dari {LR} menjadi {FINETUNE_LR}")
            for param in model.parameters():
                param.requires_grad = True
            
            opt = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
            
            # --- [IMPROVISASI] ---
            # Ganti scheduler lama dengan Warm Restarts
            # T_0 = 15: Reset LR setiap 15 epoch.
            print(f"Menggunakan CosineAnnealingWarmRestarts. T_0=15, eta_min=1e-7")
            scheduler = CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=1, eta_min=1e-7)
            # ---------------------------
            
            # [IMPROVISASI] Beri kesabaran lebih untuk fine-tuning
            print("Me-reset EarlyStopping untuk tahap fine-tuning (Patience=15).")
            early_stopping = EarlyStopping(patience=15) # <-- Naikkan ke 15
            # ---------------------------

        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, labels_mixup = mixup_fn(imgs, labels)
            
            opt.zero_grad()
            # [FIX DEPRECATION] Gunakan torch.amp.autocast
            with torch.amp.autocast(device_type=device.type):
                out = model(imgs)
                loss = crit(out, labels_mixup) 
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, crit) 
        
        # Panggil scheduler.step() DI LUAR loop batch (sekali per epoch)
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        print(f"{model_name} | Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {opt.param_groups[0]['lr']:.1e} | " # Anda akan lihat LR ini 'melompat'
                f"Time: {elapsed:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        early_stopping(val_acc, model, best_model_path) 
        
        if early_stopping.early_stop:
            print("EarlyStopping terpicu. Menghentikan training.")
            break
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return {"best_score": early_stopping.best_score, "history": history}


# --- evaluate (Perbaikan deprecation warning) ---
def evaluate(model, loader, crit):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # [FIX DEPRECATION] Gunakan torch.amp.autocast
            with torch.amp.autocast(device_type=device.type):
                 out = model(imgs)
            loss = F.cross_entropy(out, labels) 
            running_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return running_loss / total, correct / total

# --- plot_history (Tidak Berubah) ---
def plot_history(history, model_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{model_name} Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Accuracy", color='orange')
    plt.title(f"{model_name} Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- plot_confusion_matrix (Perbaikan deprecation warning) ---
def plot_confusion_matrix(model, loader, model_name):
    print(f"--- Generating Confusion Matrix for {model_name} ---")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Calculating CM for {model_name}"):
            imgs = imgs.to(device)
            # [FIX DEPRECATION] Gunakan torch.amp.autocast
            with torch.amp.autocast(device_type=device.type):
                out = model(imgs)
            _, pred = torch.max(out, 1)
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=FOOD_CLASSES, yticklabels=FOOD_CLASSES)
    plt.title(f"{model_name} Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# --- plot_saliency_maps (Perbaikan deprecation warning) ---
def plot_saliency_maps(model, loader, unnorm_transform, model_name, num_images=5):
    print(f"--- Generating Saliency Maps for {model_name} ---")
    model.eval()
    try:
        imgs_norm, labels = next(iter(loader))
    except StopIteration:
        print("Validation loader is empty, cannot generate saliency maps.")
        return
    if len(imgs_norm) < num_images:
        num_images = len(imgs_norm)
    imgs_norm = imgs_norm[:num_images].to(device)
    labels = labels[:num_images].to(device)
    imgs_unnorm = [unnorm_transform(img.cpu()) for img in imgs_norm]
    imgs_norm.requires_grad = True
    model.zero_grad()
    
    # [FIX DEPRECATION] Gunakan torch.amp.autocast
    with torch.amp.autocast(device_type=device.type):
        out = model(imgs_norm)
        
    scores = out.gather(1, labels.view(-1, 1)).squeeze()
    scores.sum().backward()
    saliency = imgs_norm.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)
    saliency = saliency.cpu().numpy()
    plt.figure(figsize=(10, num_images * 3.5))
    plt.suptitle(f"{model_name} Saliency Maps", fontsize=18)
    for i in range(num_images):
        ax = plt.subplot(num_images, 2, i*2 + 1)
        img_display = imgs_unnorm[i].permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        ax.imshow(img_display)
        ax.set_title(f"Original: {FOOD_CLASSES[labels[i].item()]}")
        ax.axis('off')
        ax = plt.subplot(num_images, 2, i*2 + 2)
        ax.imshow(saliency[i], cmap='hot')
        ax.set_title("Saliency Map")
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

# --- predict_ensemble_test (Perbaikan deprecation warning) ---
def predict_ensemble_test(models, test_dir, base_transform, model_norms, out_csv="submission_ensemble.csv"):
    for model in models:
        model.eval()
    assert len(models) == len(model_norms), "Jumlah model dan normalisasi harus sama"
    preds, ids = [], []
    
    tta_base_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg','.png'))])
    
    with torch.no_grad():
        for f in tqdm(files, desc="Ensemble Predicting (Logit Avg + TTA)"):
            img_path = os.path.join(test_dir, f)
            
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error memuat {img_path}: {e}. Skipping.")
                continue

            img_t = tta_base_transform(img).unsqueeze(0).to(device)
            img_flipped_t = tta_base_transform(img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
            
            all_logits = []
            for i, model in enumerate(models):
                norm_transform = model_norms[i]
                normalized_img = norm_transform(img_t) 
                normalized_img_flipped = norm_transform(img_flipped_t)
                
                # [FIX DEPRECATION] Gunakan torch.amp.autocast
                with torch.amp.autocast(device_type=device.type):
                    logits_orig = model(normalized_img)
                    logits_flipped = model(normalized_img_flipped)
                
                avg_tta_logits = (logits_orig + logits_flipped) / 2.0
                all_logits.append(avg_tta_logits)
            
            avg_ensemble_logits = torch.stack(all_logits).mean(dim=0)
            _, pred = torch.max(avg_ensemble_logits, 1)
            
            preds.append(FOOD_CLASSES[pred.item()])
            file_id = os.path.splitext(f)[0]
            ids.append(int(file_id))

    submission_df = pd.DataFrame({"ID": ids, "label": preds})
    submission_df = submission_df.sort_values("ID")
    submission_df.to_csv(out_csv, index=False)
    print(f"Ensemble predictions (with TTA) saved to {out_csv}")


# --- [PERUBAHAN UTAMA] Fungsi main() ---
def main():
    labels_df = pd.read_csv("train/train_labels_completed3.csv").dropna(subset=["label"])
    labels_df = labels_df[labels_df['label'] != 'Kotak Putih']
    if labels_df.empty:
        print("Error: Tidak ada data tersisa setelah memfilter 'Kotak Putih'.")
        return
    label_counts = labels_df['label'].value_counts()
    if (label_counts < 2).any():
        print("Peringatan: Beberapa kelas hanya memiliki 1 sampel, stratifikasi dinonaktifkan.")
        train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)
    else:
        train_df, val_df = train_test_split(labels_df, test_size=0.2,
                                            stratify=labels_df["label"], random_state=42)

    # --- [PERUBAHAN] num_workers diatur ke 16 ---
    # INGAT: HANYA JALANKAN SEBAGAI FILE .PY DARI TERMINAL
    loader_ops = {'num_workers': 16, 'pin_memory': True}
    print(f"Menggunakan DataLoader options: {loader_ops}")
    # -----------------------------------------------

    # Buat SATU set Dataset & DataLoader (384px)
    train_ds = FoodDataset(TRAIN_IMG_DIR, train_df, train_tf)
    val_ds   = FoodDataset(TRAIN_IMG_DIR, val_df, test_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_ops)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_ops)

    # --- Tentukan 2 Model Anda ---
    model_1_name = "convnextv2_base.fcmae_ft_in22k_in1k" 
    model_2_name = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"

    # --- Model 1: ConvNeXt ---
    print(f"\n=== Training {model_1_name} (384px, Warm Restarts) ===")
    model_1 = create_model(model_1_name)
    result_1 = train_model(model_1, train_loader, val_loader, model_1_name)
    plot_history(result_1["history"], model_1_name)
    print(f"\n--- Memulai Visualisasi untuk {model_1_name} ---")
    model_1.load_state_dict(torch.load(f"{model_1_name}_best.pth")) 
    plot_confusion_matrix(model_1, val_loader, model_1_name)
    # plot_saliency_maps(model_1, val_loader, IMAGENET_UNNORM, model_1_name) # Opsional


    # --- Model 2: Swin Transformer (384px) ---
    print(f"\n=== Training {model_2_name} (384px, Warm Restarts) ===")
    model_2 = create_model(model_2_name)
    result_2 = train_model(model_2, train_loader, val_loader, model_2_name)
    plot_history(result_2["history"], model_2_name)
    print(f"\n--- Memulai Visualisasi untuk {model_2_name} ---")
    model_2.load_state_dict(torch.load(f"{model_2_name}_best.pth")) 
    plot_confusion_matrix(model_2, val_loader, model_2_name)
    # plot_saliency_maps(model_2, val_loader, IMAGENET_UNNORM, model_2_name) # Opsional

    # --- Laporan Akhir (Hanya 2 model) ---
    print("\n=== Training Selesai ===")
    print(f"{model_1_name} Best Val Acc: {result_1['best_score']:.4f}")
    print(f"{model_2_name} Best Val Acc: {result_2['best_score']:.4f}")

    # --- Ensemble Prediction (Hanya 2 model) ---
    print("\n=== Memulai Ensemble Prediction (Logit Averaging + TTA) ===")
    ensemble_models = [model_1, model_2] # <-- Hanya 2 model
    
    pred_base_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # <-- 384px
        transforms.ToTensor()
    ])
    
    pred_model_norms = [
        IMAGENET_NORM, # Untuk ConvNext
        IMAGENET_NORM  # Untuk Swin
    ]
    
    predict_ensemble_test(
        ensemble_models, 
        TEST_IMG_DIR, 
        pred_base_tf,
        pred_model_norms,
        out_csv="submission_ensemble_2models_WarmRestarts.csv" 
    )

# Pastikan ini ada di akhir file dan TIDAK ada kode lain di luar fungsi
if __name__ == "__main__":
    main()