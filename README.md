# age_detection
Dưới đây là nội dung README cho project của bạn trên GitHub, giúp người xem hiểu mục tiêu, cách chạy, và cấu trúc code của hệ thống **dự đoán tuổi từ ảnh chân dung có tính đến sắc tộc (race)**:

---

```markdown
## 🧠 Age Estimation from Facial Images with Race Awareness

This project aims to predict a person's age from a cropped face image, using a **deep learning model that considers race** as an auxiliary input. The dataset includes images in the format `[age]_[gender]_[race]_[datetime].jpg`.

---

## 📂 Dataset Format

All images are portraits, and have been preprocessed using **face detection and cropping** to focus only on the face. File naming convention:

```

\[age]*\[gender]*\[race]\_\[datetime].jpg

````

- `age`: integer from 0 to 100+
- `gender`: 0 (male), 1 (female)
- `race`:  
  - 0 = White  
  - 1 = Black  
  - 2 = Asian  
  - 3 = Indian  
  - 4 = Others

---

## 🚀 Features

- ✅ Automatically crop faces using MTCNN (facenet-pytorch)
- ✅ Extract labels directly from filename
- ✅ Train CNN model with image and race as input
- ✅ Supports both **age regression** and **age classification**
- ✅ Modular dataset loading and training pipeline

---

## 🛠️ Setup

```bash
# Clone repo
git clone https://github.com/yourusername/age-race-estimation.git
cd age-race-estimation

# Install requirements
pip install -r requirements.txt
````

> ⚠️ Requires: `torch`, `torchvision`, `PIL`, `facenet-pytorch`, `numpy`

---

## 📁 Folder Structure

```
age-race-estimation/
├── cropped_data/            # Cropped face images
├── models/                  # Trained model checkpoints
├── dataset.py               # Custom Dataset class
├── train.py                 # Training script
├── model.py                 # Model definition
├── utils.py                 # Helper functions (e.g., label extraction)
└── README.md
```

---

## 🧪 Training

### 1️⃣ Crop face images:

```python
from crop_faces import crop_and_replace_faces

crop_and_replace_faces(
    input_path="/content/drive/MyDrive/.../merged_data/",
    output_path="/content/drive/MyDrive/.../age_data_cropped/",
    resize_size=(128, 128)
)
```

### 2️⃣ Train the model:

```bash
python train.py --data_dir ./cropped_data --epochs 30 --lr 0.001 --race-aware
```

---

## 🧠 Model Design

The model uses a CNN backbone to extract image features, and concatenates it with a one-hot encoded race vector:

```text
Image (3x128x128) → CNN → Flatten
Race (one-hot) → FC
→ [Concat] → FC → Output (Age)
```

* **Loss**: CrossEntropyLoss (for age classification) or MSELoss (for regression)
* **Optimizer**: Adam

---

## 📈 Evaluation

The model is evaluated using:


---

## ✍️ Author

👤 **Your Name**
📧 [maihoanganh_t67@hus.edu.vn](mailto:maihoanganh_t67@hus.edu.vn)
🔗 GitHub: [@hoanganh3211](https://github.com/hoanganh3211)

---
