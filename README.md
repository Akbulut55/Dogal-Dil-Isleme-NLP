# 🚀 Doğal Dil İşleme (NLP) - Sıfırdan LLM Geliştirme (Building an LLM From Scratch)
**Samsun Üniversitesi - Yazılım Mühendisliği Bölümü** 🔴⚪

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Course](https://img.shields.io/badge/Ders-NLP-success)
![Status](https://img.shields.io/badge/Durum-Aktif-brightgreen)

Samsun Üniversitesi Yazılım Mühendisliği bölümü öğrencileri için hazırlanan bu GitHub deposu, **Doğal Dil İşleme (NLP)** dersinin ana kod havuzudur. Bu dönemki ana temamız: **Sıfırdan Büyük Dil Modeli (LLM) Geliştirmek!** 🧠💻

Bu repoda, devasa dil modellerinin (ChatGPT, Llama vb.) arkasında yatan **Transformer** mimarisinin tüm temel bileşenlerini adım adım kodlayacağız. Kara kutuları kırıyor, her bir nöronun ve matris çarpımının ne işe yaradığını PyTorch ile en temelden inşa ederek öğreniyoruz.



---

## 📚 Ders İçeriği ve Yol Haritası

Bu deponun içeriği, temel referans kaynağımız olan Sebastian Raschka'nın *Build a Large Language Model (From Scratch)* kitabına paralel olarak ilerlemektedir.

Aşağıdaki klasörler, haftalık ders işleyişimize göre isimlendirilmiştir:

### 🔹 Bölüm 1: LLM'leri Anlamak ve Temel Kavramlar
Büyük Dil Modellerine (LLM) giriş. GPT mimarisinin temelleri ve büyük verilerle çalışma prensipleri.

### 🔹 Bölüm 2: Metin Verisiyle Çalışmak (Data Processing & Embeddings)
Ham metni makinenin anlayabileceği formata dönüştürme.
* **Tokenizer Tasarımı:** Byte-Pair Encoding (BPE) mekanizması.
* **Vector Embeddings:** Kelimeleri vektör uzayına taşıma.
* **Positional Encoding:** Modele kelimelerin sırasını öğretme.

### 🔹 Bölüm 3: Dikkat Mekanizmalarını Kodlama (Attention Mechanisms)
Transformer mimarisinin kalbi olan Dikkat mekanizmaları.
* Basit Self-Attention (Öz-Dikkat) hesaplamaları.
* Masked Self-Attention (Gelecekteki kelimeleri gizleme).
* Multi-Head Attention (Çok Başlı Dikkat) bloğunun sıfırdan yazılması.



[Image of self attention mechanism]


### 🔹 Bölüm 4: Sıfırdan GPT Mimarisini İnşa Etmek
Öğrendiğimiz tüm parçaları birleştirerek çalışan bir model oluşturma.
* Transformer Bloklarının (LayerNorm, GELU vb.) kodlanması.
* GPT-2 benzeri bir ağ mimarisinin PyTorch ile birleştirilmesi.

### 🔹 Bölüm 5: Etiketsiz Veri ile Ön Eğitim (Pretraining)
Modelin dilin gramerini ve yapısını öğrenmesi için devasa metinler üzerinde eğitilmesi (Next-token prediction).

### 🔹 Bölüm 6 & 7: İnce Ayar (Fine-Tuning)
Eğitilen ham modeli belirli görevler için özelleştirme.
* Sınıflandırma görevleri için fine-tuning (Spam tespiti vb.).
* Talimat İzleme (Instruction Following) için fine-tuning.

---
