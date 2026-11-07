GPT-2 small'ın hesabı: embedding'ler $50257\cdot768+1024\cdot768=39{,}383{,}808$; bloklar $12\times7{,}087{,}872=85{,}054{,}464$; son LN $1{,}536$. Toplam $124{,}439{,}808$ — HuggingFace'in raporladığı sayı. Parametrelerin yaklaşık üçte biri embedding.

Top-k ile temperature birlikte çalışır: önce kes, sonra ölçekle ve normalize et.
