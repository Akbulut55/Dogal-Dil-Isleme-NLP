
# from transformers import GPT2Tokenizer

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# text = "Tokenization is the first step in LLM training."
# tokens = tokenizer.tokenize(text)
# ids = tokenizer.convert_tokens_to_ids(tokens)

# print(f"Text: {text}")
# print(f"Tokens: {tokens}")
# print(f"Token IDs: {ids}")

# import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")
# text = "Tokenization is fascinating!"
# token_ids = tokenizer.encode(text) 
# print("Metin:", text) 
# print("Token ID'leri:", token_ids)

# import torch
# import torch.nn as nn

# print("--- 1. nn.Embedding Örneği (Kelime Vektörleştirme) ---")
# # Sözlüğümüzde 10 farklı kelime var (num_embeddings=10)
# # Her kelimeyi 3 boyutlu bir vektörle (embedding_dim=3) temsil edeceğiz.
# embedding_katmani = nn.Embedding(num_embeddings=4, embedding_dim=8)

# # GİRDİ: Kelimelerin sözlükteki indeksleri (TAM SAYI olmalı!)
# # Örneğin: 2="Merhaba", 5="Dünya"
# kelime_indeksleri = torch.tensor([2, 3])
# # ÇIKTI: 2. ve 5. satırdaki 3 boyutlu vektörleri doğrudan çeker.
# embedding_ciktisi = embedding_katmani(kelime_indeksleri)
# print("Girdi (İndeksler):\n", kelime_indeksleri)
# print("Çıktı (Vektörler):\n", embedding_ciktisi)

import tiktoken
# 1. GPT-4 ve GPT-3.5'in kullandığı standart kodlayıcıyı (encoding) yüklüyoruz
encoding = tiktoken.get_encoding("gpt2")
metin = "Tokenization is the first step in LLM training."
#metin = "akwirwier"
# 2. Metni Token ID'lerine dönüştürüyoruz (Encode işlemi)
token_id_listesi = encoding.encode(metin)
print('token list',token_id_listesi)
print(f"Orijinal Metin: '{metin}'\n")
print(f"{'Token ID':<12} | {'Token Metni'}")
print("-" * 30)
# 3. Her bir ID'yi tek tek metne geri çeviriyoruz (Decode işlemi)
for token_id in token_id_listesi:
    # decode_single_token_bytes byte döndürür, utf-8 ile string'e çeviriyoruz
    # errors='replace' parametresi, Türkçe karakterler parçalanırsa hata vermesini önler
    token_metni = encoding.decode_single_token_bytes(token_id).decode('utf-8', errors='replace')
    
    print(f"{token_id:<12} | '{token_metni}'")