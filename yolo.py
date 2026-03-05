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
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data import Mixup
import time

from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F

# -----------------------------------------------------------------
# [PERBAIKAN] YOLO DIHAPUS DARI SINI. 
# SEMUA LOGIKA YOLO PINDAH KE 'preprocess_images.py'
# -----------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 25
NUM_CLASSES = 15
LR = 1e-3
FINETUNE_LR = 1e-4 
UNFREEZE_EPOCH = 5

# --- [PERBAIKAN] Tentukan path ke data yang sudah di-crop ---
TRAIN_IMG_DIR = "train/train_cropped" 
TEST_IMG_DIR = "test/test_cropped"
# ------------------------------------------------------------


FOOD_CLASSES = [
    'Ayam Bakar','Ayam Betutu','Ayam Goreng','Ayam Pop','Bakso',
    'Coto Makassar','Gado Gado','Gudeg','Nasi Goreng','Pempek',
    'Rendang','Rawon','Sate Madura','Sate Padang','Soto'
]

# --- [PERBAIKAN] DATASET YANG JAUH LEBIH SEDERHANA & CEPAT ---
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
            print(f"Peringatan: File tidak ditemukan {img_path}. Menggunakan placeholder.")
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="white")
        except Exception as e:
            print(f"Error memuat {img_path}: {e}. Menggunakan placeholder.")
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="white")

        if self.transform:
            img = self.transform(img)
            
        # Kita tidak perlu lagi mengembalikan 'is_cropped'
        return img, label
# --- AKHIR PERBAIKAN ---

# --- Transformasi (Tidak Berubah, RandAugment penting untuk akurasi) ---
IMAGENET_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
convnext_train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    RandAugment(num_ops=2, magnitude=9), # <-- Augmentasi kuat
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), # <-- Tingkatkan sedikit
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    IMAGENET_NORM
])
convnext_test_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    IMAGENET_NORM
])
TF_NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
effnet_train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    RandAugment(num_ops=2, magnitude=9), # <-- Augmentasi kuat
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), # <-- Tingkatkan sedikit
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    TF_NORM
])
effnet_test_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    TF_NORM
])
pred_base_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
pred_model_norms = [IMAGENET_NORM, TF_NORM]

# --- (Sisa kode helper tidak berubah) ---
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
TF_UNNORM = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def create_model(name):
    model = timm.create_model(name, pretrained=True, num_classes=NUM_CLASSES)
    for param in model.parameters():
        param.requires_grad = False
    if 'convnext' in name:
        model.head = nn.Sequential(nn.Dropout(0.5), model.head)
        for param in model.head.parameters():
            param.requires_grad = True
    elif 'efficientnet' in name:
        model.classifier = nn.Sequential(nn.Dropout(0.5), model.classifier)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Model type {name} not recognized for classifier modification.")
    return model.to(device)

# --- [PERBAIKAN] Tambahkan print() pada EarlyStopping ---
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
                # --- [TAMBAHAN] Pesan yang Anda minta ---
                print(f"==> EarlyStopping terpicu! Tidak ada peningkatan selama {self.patience} epoch.")
                # -----------------------------------------
        else:
            self.best_score = score
            torch.save(model.state_dict(), path)
            self.counter = 0

mixup_fn = Mixup(
    mixup_alpha=0.4, cutmix_alpha=0.4, label_smoothing=0.1,
    num_classes=NUM_CLASSES, prob=1.0, switch_prob=0.5, mode='batch'
)

def train_model(model, train_loader, val_loader, model_name):
    crit = nn.CrossEntropyLoss()
    print(f"--- Tahap 1: Melatih Head (Epoch 0-{UNFREEZE_EPOCH-1}) ---")
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=UNFREEZE_EPOCH)
    
    early_stopping = EarlyStopping(patience=5) # Anda bisa naikkan patience ke 7 atau 10
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []} 
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        if epoch == UNFREEZE_EPOCH:
            print(f"\n--- Tahap 2: Unfreezing Backbone di Epoch {epoch} ---")
            print(f"Mengganti LR dari {LR} menjadi {FINETUNE_LR}")
            for param in model.parameters():
                param.requires_grad = True
            opt = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(opt, T_max=(EPOCHS - UNFREEZE_EPOCH))

        model.train()
        running_loss = 0.0
        # [PERBAIKAN] Ubah `_` menjadi `labels` karena `is_cropped` sudah dihapus
        for imgs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, labels_mixup = mixup_fn(imgs, labels) # Gunakan labels_mixup untuk loss
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels_mixup) # Gunakan label dari mixup
            loss.backward()
            opt.step()
            running_loss += loss.item() * imgs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, crit) # Fungsi evaluate disederhanakan
        scheduler.step()
        elapsed = time.time() - start_time
        
        print(f"{model_name} | Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              # [PERBAIKAN] Info crop dihapus, tidak relevan lagi
              f"LR: {opt.param_groups[0]['lr']:.1e} | "
              f"Time: {elapsed:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        early_stopping(val_acc, model, f"{model_name}_best.pth")
        
        if early_stopping.early_stop:
            # Pesan akan dicetak oleh class EarlyStopping
            break
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return {"best_acc": early_stopping.best_score, "history": history}

def evaluate(model, loader, crit):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    # total_cropped dihapus
    with torch.no_grad():
        # [PERBAIKAN] `cropped_flags` dihapus
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            # [PERBAIKAN] Gunakan label asli untuk evaluasi, bukan mixup
            loss = F.cross_entropy(out, labels) 
            running_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            # total_cropped += sum(cropped_flags) # Dihapus
    # [PERBAIKAN] Return value disederhanakan
    return running_loss / total, correct / total

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

def plot_confusion_matrix(model, loader, model_name):
    print(f"--- Generating Confusion Matrix for {model_name} ---")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        # [PERBAIKAN] `_` dihapus
        for imgs, labels in tqdm(loader, desc=f"Calculating CM for {model_name}"):
            imgs = imgs.to(device)
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

def plot_saliency_maps(model, loader, unnorm_transform, model_name, num_images=5):
    print(f"--- Generating Saliency Maps for {model_name} ---")
    model.eval()
    try:
        # [PERBAIKAN] `_` dihapus
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

# --- [PERBAIKAN] Fungsi ensemble yang jauh lebih sederhana ---
def predict_ensemble_test(models, test_dir, base_transform, model_norms, out_csv="submission_ensemble.csv"):
    for model in models:
        model.eval()
    assert len(models) == len(model_norms), "Jumlah model dan normalisasi harus sama"
    preds, ids = [], []
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg','.png'))])
    
    with torch.no_grad():
        for f in tqdm(files, desc="Ensemble Predicting (Logit Avg)"):
            img_path = os.path.join(test_dir, f)
            
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error memuat {img_path}: {e}. Skipping.")
                continue

            # [PERBAIKAN] Logika YOLO dihapus. Langsung transform.
            img_t = base_transform(img).unsqueeze(0).to(device)
            
            all_logits = []
            for i, model in enumerate(models):
                norm_transform = model_norms[i]
                # Terapkan normalisasi spesifik model
                # Perhatian: `base_transform` sudah ToTensor(), jadi kita hanya perlu norm.
                # Kita perlu sedikit modifikasi di sini.
                
                # Mari perbaiki alur transformasi
                # `base_transform` (pred_base_tf) HANYA Resize + ToTensor
                # `model_norms` adalah transforms.Normalize
                
                normalized_img = norm_transform(img_t) # Terapkan normalisasi
                
                logits = model(normalized_img)
                all_logits.append(logits)
            
            avg_logits = torch.stack(all_logits).mean(dim=0)
            _, pred = torch.max(avg_logits, 1)
            
            preds.append(FOOD_CLASSES[pred.item()])
            file_id = os.path.splitext(f)[0]
            ids.append(int(file_id))

    submission_df = pd.DataFrame({"ID": ids, "label": preds})
    submission_df = submission_df.sort_values("ID")
    submission_df.to_csv(out_csv, index=False)
    print(f"Ensemble predictions saved to {out_csv}")
# --- AKHIR PERBAIKAN ---


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

    # --- [PERBAIKAN] Tentukan OPSI LOADER ---
    # Sesuaikan num_workers dengan jumlah core CPU Anda (misal: 4, 8)
    loader_ops = {'num_workers': 16, 'pin_memory': True}
    print(f"Menggunakan DataLoader options: {loader_ops}")
    # -------------------------------------------

    # --- Model 1: ConvNeXt ---
    # [PERBAIKAN] Gunakan TRAIN_IMG_DIR
    convnext_train_ds = FoodDataset(TRAIN_IMG_DIR, train_df, convnext_train_tf)
    convnext_val_ds   = FoodDataset(TRAIN_IMG_DIR, val_df, convnext_test_tf)
    convnext_train_loader = DataLoader(convnext_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_ops)
    convnext_val_loader   = DataLoader(convnext_val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_ops)

    model_1_name = "convnextv2_base.fcmae_ft_in22k_in1k"
    print(f"\n=== Training {model_1_name} ===")
    model_1 = create_model(model_1_name)
    result_1 = train_model(model_1, convnext_train_loader, convnext_val_loader, model_1_name)
    plot_history(result_1["history"], model_1_name)

    print(f"\n--- Memulai Visualisasi untuk {model_1_name} ---")
    model_1.load_state_dict(torch.load(f"{model_1_name}_best.pth")) # Hapus weights_only=True
    plot_confusion_matrix(model_1, convnext_val_loader, model_1_name)
    plot_saliency_maps(model_1, convnext_val_loader, IMAGENET_UNNORM, model_1_name)


    # --- Model 2: EfficientNetV2 (TF) ---
    # [PERBAIKAN] Gunakan TRAIN_IMG_DIR
    effnet_train_ds = FoodDataset(TRAIN_IMG_DIR, train_df, effnet_train_tf)
    effnet_val_ds   = FoodDataset(TRAIN_IMG_DIR, val_df, effnet_test_tf)
    effnet_train_loader = DataLoader(effnet_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_ops)
    effnet_val_loader   = DataLoader(effnet_val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_ops)

    model_2_name = "tf_efficientnetv2_l.in21k_ft_in1k"
    print(f"\n=== Training {model_2_name} ===")
    model_2 = create_model(model_2_name)
    result_2 = train_model(model_2, effnet_train_loader, effnet_val_loader, model_2_name)
    plot_history(result_2["history"], model_2_name)

    print(f"\n--- Memulai Visualisasi untuk {model_2_name} ---")
    model_2.load_state_dict(torch.load(f"{model_2_name}_best.pth")) # Hapus weights_only=True
    plot_confusion_matrix(model_2, effnet_val_loader, model_2_name)
    plot_saliency_maps(model_2, effnet_val_loader, TF_UNNORM, model_2_name)


    # --- Laporan Akhir ---
    print("\n=== Training Selesai ===")
    print(f"{model_1_name} Best Val Acc: {result_1['best_acc']:.4f}")
    print(f"{model_2_name} Best Val Acc: {result_2['best_acc']:.4f}")

    # --- Ensemble Prediction ---
    print("\n=== Memulai Ensemble Prediction (Logit Averaging) ===")
    ensemble_models = [model_1, model_2]
    
    # [PERBAIKAN] Perbaiki alur transformasi untuk prediksi
    # `pred_base_tf` (Resize + ToTensor)
    # `pred_model_norms` (List berisi [IMAGENET_NORM, TF_NORM])
    
    # [PERBAIKAN] Pastikan `pred_base_tf` TIDAK menormalkan
    pred_base_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    pred_model_norms = [
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    
    predict_ensemble_test(
        ensemble_models, 
        TEST_IMG_DIR, # [PERBAIKAN] Gunakan folder test yang sudah di-crop
        pred_base_tf,
        pred_model_norms,
        out_csv="submission_ensemble(pre-cropped-convnext+effnetv2_L-logit_avg).csv" # Nama file baru
    )