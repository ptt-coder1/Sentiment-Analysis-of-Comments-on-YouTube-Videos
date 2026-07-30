# 🎯 Phân Tích Cảm Xúc Bình Luận YouTube Tiếng Việt

> Dự án phân tích cảm xúc (Sentiment Analysis) bình luận YouTube tiếng Việt sử dụng kết hợp các mô hình Machine Learning cổ điển (Random Forest, Logistic Regression, Naive Bayes) và các mô hình Transformer hiện đại (PhoBERT-base-v2, mDeBERTa-v3-base, CafeBERT).

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Pipeline xử lý](#-pipeline-xử-lý)
- [Cài đặt](#-cài-đặt)
- [Cấu hình API Key](#-cấu-hình-api-key)
- [Hướng dẫn chạy](#-hướng-dẫn-chạy)
- [Kết quả mô hình](#-kết-quả-mô-hình)
- [Lưu ý quan trọng](#-lưu-ý-quan-trọng)

---

## 🔍 Tổng quan

| Mục | Chi tiết |
|-----|----------|
| **Bài toán** | Phân loại cảm xúc 3 nhãn: Tích cực / Trung tính / Tiêu cực |
| **Nguồn dữ liệu** | Bình luận YouTube thu thập qua YouTube Data API v3 |
| **Ngôn ngữ** | Tiếng Việt (có xử lý teencode, emoji, stopword) |
| **Metric chính** | F1-Macro (phù hợp với dữ liệu mất cân bằng nhãn) |
| **Môi trường** | Google Colab (GPU T4/A100) |

---

## 📁 Cấu trúc dự án

```
.
├── notebooks/
│   └── phan_tich_cam_xuc_youtube.ipynb   # Notebook chính (đã xoá output)
├── data/
│   └── DMviettat_Goiy.csv                # Từ điển teencode (ViLexNorm)
├── requirements.txt                       # Danh sách thư viện
├── .env.example                           # Mẫu biến môi trường
├── .gitignore
└── LICENSE
```

> **Lưu ý:** Dữ liệu bình luận thô (CSV/Excel) **không được** đưa vào repo do kích thước lớn và điều khoản sử dụng YouTube API. Các file model lớn (`.pkl`, `.pt`) cũng được loại trừ.

---

## ⚙️ Pipeline xử lý

```
Thu thập DL          Tiền xử lý           Gán nhãn          Huấn luyện         Đánh giá
(YouTube API)  →  (Teencode/Emoji/  →  (ViSoBERT tự  →  (ML + Transformer) →  (F1-Macro,
                   Stopword/Lang)        động)                                  Confusion Matrix)
```

**7 bước chính trong notebook:**

1. **Thu thập dữ liệu** — Gọi YouTube Data API v3 (chiến lược 2 pha)
2. **Tiền xử lý** — Chuẩn hóa teencode, xử lý emoji, lọc ngôn ngữ nước ngoài, loại stopword
3. **Gán nhãn tự động** — Dùng ViSoBERT (`uitnlp/visobert`) để gán nhãn pseudo-label
4. **Mã hóa & Chia tập** — Label encoding, train/test split 60/40
5. **Huấn luyện ML** — Random Forest, Logistic Regression, Multinomial Naive Bayes
6. **Huấn luyện Transformer** — PhoBERT-base-v2, mDeBERTa-v3-base, CafeBERT
7. **Đánh giá & Phân tích** — So sánh F1-Macro, Confusion Matrix, xu hướng cảm xúc theo thời gian

---

## 🛠 Cài đặt

### Chạy trên Google Colab (khuyến nghị)

```python
# Chạy cell đầu tiên trong notebook để mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

Sau đó cài thư viện:

```bash
!pip install -r requirements.txt
```

### Chạy local (tùy chọn)

```bash
# Clone repo
git clone https://github.com/<username>/<ten-repo>.git
cd <ten-repo>

# Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Cài thư viện
pip install -r requirements.txt
```

---

## 🔑 Cấu hình API Key

Dự án cần 2 credentials:

| Credential | Dùng cho | Cách lấy |
|---|---|---|
| `YOUTUBE_API_KEY` | Thu thập bình luận YouTube | [Google Cloud Console](https://console.cloud.google.com/apis/credentials) |
| `HF_TOKEN` | Tải model PhoBERT / CafeBERT từ HuggingFace | [HuggingFace Settings](https://huggingface.co/settings/tokens) |

### Trên Google Colab (khuyến nghị — an toàn nhất)

1. Nhấn icon 🔑 **Secrets** ở sidebar trái Colab
2. Thêm 2 secret: `YOUTUBE_API_KEY` và `HF_TOKEN`
3. Notebook tự đọc bằng `google.colab.userdata.get('YOUTUBE_API_KEY')`

### Chạy local

```bash
cp .env.example .env
# Mở .env và điền giá trị thực vào
```

> ⚠️ **KHÔNG** commit file `.env` lên Git. File này đã được thêm vào `.gitignore`.

---

## ▶️ Hướng dẫn chạy

Mở `notebooks/phan_tich_cam_xuc_youtube.ipynb` và chạy tuần tự từ trên xuống.

Nếu chỉ muốn chạy từ bước huấn luyện (đã có dữ liệu sẵn):
- Bỏ qua **Mục 1** (thu thập) và **Mục 2-3** (tiền xử lý)
- Bắt đầu từ **Mục 4** (gán nhãn & chia tập)

---

## 📊 Kết quả mô hình

> Kết quả dưới đây là ví dụ tham khảo. Kết quả thực tế phụ thuộc vào dữ liệu thu thập được.

| Mô hình | Nhóm | F1-Macro (ước tính) |
|---|---|---|
| **PhoBERT-base-v2** | Transformer | ~0.82–0.87 |
| **CafeBERT** | Transformer | ~0.80–0.85 |
| **mDeBERTa-v3-base** | Transformer | ~0.78–0.83 |
| Random Forest | ML cổ điển | ~0.65–0.75 |
| Logistic Regression | ML cổ điển | ~0.62–0.72 |
| Multinomial Naive Bayes | ML cổ điển | ~0.58–0.68 |

Model tốt nhất được tự động lưu vào:
- `best_transformer_model/` — weights + tokenizer (dùng `AutoModel.from_pretrained()`)
- `best_ml_model.pkl` + `best_ml_vectorizer.pkl` — dùng `joblib.load()`

---

## ⚠️ Lưu ý quan trọng

1. **Dữ liệu thô không có trong repo** — Bạn cần tự thu thập bằng YouTube API hoặc cung cấp file CSV/Excel theo đúng schema.
2. **Model files không có trong repo** — Sau khi train, các file `.pkl` và thư mục `best_transformer_model/` được tạo ra locally. Nếu muốn chia sẻ model, dùng [HuggingFace Hub](https://huggingface.co/models) hoặc [Google Drive](https://drive.google.com).
3. **Chạy trên CPU sẽ rất chậm** — Bước huấn luyện Transformer (8 epoch × 3 model) cần GPU. Colab Free (T4) ước tính ~3–5 giờ.
4. **Quota YouTube API** — Mỗi project được 10,000 units/ngày. Thu thập lớn cần nhiều ngày hoặc nhiều API key.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
