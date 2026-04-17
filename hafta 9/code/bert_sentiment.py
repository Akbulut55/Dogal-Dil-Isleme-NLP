import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# ============================================================
# 1) VERİ SETİNİ YÜKLE
# ============================================================
print("=" * 60)
print("1) ROTTEN TOMATOES VERİ SETİ YÜKLENİYOR...")
print("=" * 60)

dataset = load_dataset("rotten_tomatoes")

print(f"\nTrain     : {len(dataset['train'])} örnek")
print(f"Validation: {len(dataset['validation'])} örnek")
print(f"Test      : {len(dataset['test'])} örnek")

from collections import Counter
print(f"\nTrain label dağılımı: {Counter(dataset['train']['label'])}")
print("  0 = Negatif, 1 = Pozitif")

# ============================================================
# 2) TRAIN SETİNDEN 5 ÖRNEK GÖSTER
# ============================================================
print("\n" + "=" * 60)
print("2) TRAIN SETİNDEN 5 ÖRNEK TEXT ve LABEL")
print("=" * 60)

label_map = {0: "NEGATİF 😠", 1: "POZİTİF 😊"}
for i in range(5):
    text = dataset['train'][i]['text']
    label = dataset['train'][i]['label']
    print(f"\n  [{i+1}] Label: {label} ({label_map[label]})")
    print(f"      Text : \"{text}\"")

# ============================================================
# 3) BERT TOKENİZER & MODEL (FROZEN)
# ============================================================
print("\n" + "=" * 60)
print("3) BERT MODELİ YÜKLENİYOR (ÖZELLİK ÇIKARICI - FROZEN)...")
print("=" * 60)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model = bert_model.to(device)
bert_model.eval()  # Eğitim yapmayacağız, inference modunda

# BERT parametrelerini dondur
for param in bert_model.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in bert_model.parameters())
print(f"BERT toplam parametre: {total_params:,}")
print("Tüm parametreler DONDURULDU (requires_grad=False)")

# ============================================================
# 4) BERT İLE EMBEDDİNG ÇIKARMA FONKSİYONU
# ============================================================
def extract_bert_embeddings(texts, tokenizer, model, batch_size=64, max_length=128):
    """
    Verilen text listesi için BERT [CLS] token embeddinglerini çıkarır.
    [CLS] token: Cümlenin genel anlamsal temsilini taşır (768 boyutlu vektör)
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # BERT forward pass (gradient hesaplamaya gerek yok)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # [CLS] token embedding'i al (ilk token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        all_embeddings.append(cls_embeddings.cpu())
        
        if (i // batch_size) % 20 == 0:
            print(f"  İşlenen: {min(i + batch_size, len(texts))}/{len(texts)}")
    
    return torch.cat(all_embeddings, dim=0)

# ============================================================
# 5) TÜM SPLİTLER İÇİN EMBEDDİNG ÇIKAR
# ============================================================
print("\n" + "=" * 60)
print("5) BERT EMBEDDİNGLERİ ÇIKARILIYOR...")
print("=" * 60)

print("\n--- Train seti ---")
train_embeddings = extract_bert_embeddings(dataset['train']['text'], tokenizer, bert_model)
train_labels = torch.tensor(dataset['train']['label'], dtype=torch.long)

print("\n--- Validation seti ---")
val_embeddings = extract_bert_embeddings(dataset['validation']['text'], tokenizer, bert_model)
val_labels = torch.tensor(dataset['validation']['label'], dtype=torch.long)

print("\n--- Test seti ---")
test_embeddings = extract_bert_embeddings(dataset['test']['text'], tokenizer, bert_model)
test_labels = torch.tensor(dataset['test']['label'], dtype=torch.long)

print(f"\nEmbedding boyutları:")
print(f"  Train : {train_embeddings.shape}")   # (8530, 768)
print(f"  Val   : {val_embeddings.shape}")      # (1066, 768)
print(f"  Test  : {test_embeddings.shape}")     # (1066, 768)

# ============================================================
# 6) DATALOADER OLUŞTUR
# ============================================================
BATCH_SIZE = 64

train_dataset = TensorDataset(train_embeddings, train_labels)
val_dataset = TensorDataset(val_embeddings, val_labels)
test_dataset = TensorDataset(test_embeddings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 7) 3 KATMANLI MLP MODELİ (PyTorch)
# ============================================================
print("\n" + "=" * 60)
print("7) MLP MODELİ TANIMLANIYOR (3 Katman, 2 Output Nöronu)")
print("=" * 60)

class SentimentMLP(nn.Module):
    """
    3 Katmanlı MLP:
      Katman 1: 768 → 256  (BERT embedding boyutu → gizli katman)
      Katman 2: 256 → 64   (gizli katman → gizli katman)
      Katman 3: 64  → 2    (gizli katman → output: negatif/pozitif)
    """
    def __init__(self, input_dim=768, hidden1=256, hidden2=64, output_dim=2):
        super(SentimentMLP, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.layer3 = nn.Linear(hidden2, output_dim)  # 2 output nöronu
    
    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.layer3(x)  # logits (softmax CrossEntropyLoss içinde)
        return x

model = SentimentMLP().to(device)

# Model özeti
print(f"\nModel Mimarisi:")
print(model)
mlp_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nEğitilebilir MLP parametre sayısı: {mlp_params:,}")

# ============================================================
# 8) EĞİTİM AYARLARI
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

NUM_EPOCHS = 15

# ============================================================
# 9) EĞİTİM DÖNGÜSÜ
# ============================================================
print("\n" + "=" * 60)
print("9) MLP EĞİTİMİ BAŞLIYOR...")
print("=" * 60)

best_val_acc = 0.0
train_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    # --- TRAIN ---
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for embeddings_batch, labels_batch in train_loader:
        embeddings_batch = embeddings_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels_batch).sum().item()
        total += labels_batch.size(0)
    
    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # --- VALIDATION ---
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for embeddings_batch, labels_batch in val_loader:
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            outputs = model(embeddings_batch)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels_batch).sum().item()
            val_total += labels_batch.size(0)
    
    val_acc = val_correct / val_total
    val_accuracies.append(val_acc)
    
    # En iyi modeli kaydet
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
    
    scheduler.step()
    
    print(f"  Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}"
          f"{' ⭐ Best!' if val_acc >= best_val_acc else ''}")

print(f"\nEn iyi Validation Accuracy: {best_val_acc:.4f}")

# ============================================================
# 10) TEST SETİ DEĞERLENDİRME
# ============================================================
print("\n" + "=" * 60)
print("10) TEST SETİ DEĞERLENDİRMESİ (En İyi Model)")
print("=" * 60)

model.load_state_dict(best_model_state)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for embeddings_batch, labels_batch in test_loader:
        embeddings_batch = embeddings_batch.to(device)
        outputs = model(embeddings_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = (all_preds == all_labels).mean()
print(f"\n🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

print(f"\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Negatif", "Pozitif"]))

print(f"Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(f"                Predicted")
print(f"              Neg    Pos")
print(f"  Actual Neg  {cm[0][0]:4d}   {cm[0][1]:4d}")
print(f"  Actual Pos  {cm[1][0]:4d}   {cm[1][1]:4d}")

# ============================================================
# 11) CANLI TAHMİN ÖRNEKLERİ
# ============================================================
print("\n" + "=" * 60)
print("11) CANLI TAHMİN ÖRNEKLERİ")
print("=" * 60)

sample_texts = [
    "This movie was absolutely fantastic, a true masterpiece!",
    "Terrible film, waste of time and money.",
    "An average movie, nothing special but not bad either.",
    "The acting was brilliant and the story was captivating.",
    "I fell asleep halfway through this boring disaster.",
]

for text in sample_texts:
    encoded = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        bert_output = bert_model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device)
        )
        cls_emb = bert_output.last_hidden_state[:, 0, :].to(device)
        logits = model(cls_emb)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    emoji = "😊" if pred == 1 else "😠"
    print(f"\n  Text: \"{text}\"")
    print(f"  Tahmin: {label_map[pred]} | Neg: {probs[0][0]:.4f} | Pos: {probs[0][1]:.4f}")