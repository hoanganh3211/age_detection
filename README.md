# age_detection

## Age Estimation from Facial Images with Race Awareness

This project aims to predict a person's `age` from a `cropped face image`, using a deep learning model that considers race as an auxiliary input. The dataset includes images in the format [age]_[gender]_[race]_[datetime].jpg.

---

## Dataset Format

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

## Features

 Automatically crop faces using MTCNN (facenet-pytorch)
- Extract labels directly from filename
- Train CNN model with image and race as input
- Supports both **age regression** and **age classification**
- Modular dataset loading and training pipeline

---



## Author

👤 **Your Name**
📧 [maihoanganh_t67@hus.edu.vn](mailto:maihoanganh_t67@hus.edu.vn)
🔗 GitHub: [@hoanganh3211](https://github.com/hoanganh3211)

---
