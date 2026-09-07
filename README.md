# 🚀 Doğal Dil İşleme (NLP) - Sıfırdan LLM Geliştirme (Building an LLM From Scratch)

**Samsun Üniversitesi - Yazılım Mühendisliği Bölümü** 🔴⚪

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch\&logoColor=white)
![Course](https://img.shields.io/badge/Ders-NLP-success)
![Status](https://img.shields.io/badge/Durum-Aktif-brightgreen)

Samsun Üniversitesi Yazılım Mühendisliği bölümü öğrencileri için hazırlanan bu GitHub deposu, **Doğal Dil İşleme (NLP)** dersinin ana kod havuzudur. Bu dönemki ana tema, **sıfırdan bir Büyük Dil Modeli (LLM) geliştirmektir**. 🧠💻

Bu repoda, büyük dil modellerinin (ChatGPT, Llama vb.) temelini oluşturan **Transformer** mimarisinin ana bileşenlerini adım adım kodlayarak öğreniyoruz. Amaç, bu sistemleri yalnızca kullanmak yerine; tokenization, embeddings, attention mekanizmaları ve model mimarisi gibi temel yapı taşlarını PyTorch ile sıfırdan uygulayarak çalışma prensiplerini anlamaktır.

---

## 📚 Ders İçeriği ve Yol Haritası

Bu deponun içeriği, temel referans kaynağımız olan Sebastian Raschka'nın *Build a Large Language Model (From Scratch)* kitabına paralel olarak ilerlemektedir.

Aşağıdaki bölümler, dersin haftalık işleyişine göre düzenlenmiştir.

### 🔹 Bölüm 1: LLM'leri Anlamak ve Temel Kavramlar

Büyük Dil Modellerine (LLM) giriş, GPT mimarisinin temelleri ve büyük ölçekli metin verileriyle çalışma prensipleri.

### 🔹 Bölüm 2: Metin Verisiyle Çalışmak (Data Processing & Embeddings)

Ham metni modelin işleyebileceği sayısal temsillere dönüştürme.

* **Tokenizer Tasarımı:** Byte-Pair Encoding (BPE) mekanizması
* **Vector Embeddings:** Tokenları vektör uzayında temsil etme
* **Positional Encoding:** Modele tokenların sıralı konum bilgisini kazandırma

### 🔹 Bölüm 3: Dikkat Mekanizmalarını Kodlama (Attention Mechanisms)

Transformer mimarisinin temelini oluşturan attention mekanizmalarının uygulanması.

* Basit Self-Attention (Öz-Dikkat) hesaplamaları
* Masked Self-Attention ile gelecekteki tokenların gizlenmesi
* Multi-Head Attention (Çok Başlı Dikkat) bloğunun sıfırdan uygulanması

### 🔹 Bölüm 4: Sıfırdan GPT Mimarisini İnşa Etmek

Önceki bölümlerde oluşturulan yapı taşlarını birleştirerek çalışan bir GPT benzeri model oluşturma.

* Transformer bloklarının (`LayerNorm`, `GELU` vb.) uygulanması
* GPT-2 benzeri bir model mimarisinin PyTorch ile oluşturulması

### 🔹 Bölüm 5: Etiketsiz Veri ile Ön Eğitim (Pretraining)

Modelin dil örüntülerini öğrenebilmesi için büyük metin veri kümeleri üzerinde **next-token prediction** yaklaşımıyla ön eğitim yapılması.

### 🔹 Bölüm 6 & 7: İnce Ayar (Fine-Tuning)

Önceden eğitilmiş modelin belirli görevler için özelleştirilmesi.

* Sınıflandırma görevleri için fine-tuning (ör. spam tespiti)
* Talimat izleme (Instruction Following) için fine-tuning

---
