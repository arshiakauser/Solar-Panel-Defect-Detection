# Solar-Panel-Defect-Detection
# 🔹 Solar Panel Defect Detection using YOLOv9c-seg

## 📘 Overview
This project focuses on detecting and segmenting surface defects in solar panels using the **YOLOv9c-seg** model.  
The primary objective is to automate the inspection process and improve the efficiency and accuracy of identifying defective panels, reducing manual labor and maintenance downtime.

---

## 🧠 Objectives
- Develop a robust defect detection model for solar panels using YOLOv9c-seg.
- Segment defective regions with high accuracy to support maintenance planning.
- Evaluate model performance across multiple defect classes.
- Deploy an interactive application for visualizing defect detection results.

---

## 🗂️ Dataset
- **Source:** Custom dataset prepared using **Roboflow**.  
- **Total Images:** 1,500+ labeled images.  
- **Classes:** 6 defect categories — *Bird-drop, Defective, Dusty, Electrical-Damage, Non-Defective, Physical-Damage*.  
- **Structure:**
/train
/images
/labels
/valid
/images
/labels
/test
/images
/labels


---

## ⚙️ Tools & Technologies
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python |
| ML Framework | YOLOv9c-seg, PyTorch |
| Data Handling | Pandas, NumPy |
| Visualization | OpenCV, Matplotlib |
| Deployment | Streamlit |
| Version Control | Git, GitHub |

---

## 🚀 Implementation Steps
1. **Data Preprocessing:** Data cleaning, augmentation, and annotation via Roboflow.  
2. **Model Training:** Used YOLOv9c-seg for image segmentation and trained with early stopping.  
3. **Evaluation:** Measured performance using mAP@50, Precision, and Recall metrics.  
4. **Deployment:** Built a Streamlit web app to visualize real-time defect detection and segmentation results.

---

## 📊 Model Performance
| Metric | Bounding Box(B) | Segmentation Mask (M) |
|--------|-----------------|-----------------------|
| Precision | 0.786 | 0.779 |
| Recall | 0.715 | 0.689 |
| mAP@50 | 0.776 | 0.736 |
| mAP@50-95 | 0.583 | 0.501 |

---

## 💡 Insights & Learnings
- Data augmentation improved model generalization on unseen solar panel samples.  
- Learned optimization of YOLOv9c parameters for better segmentation accuracy.  
- Integrated computer vision models with a user-friendly dashboard using Streamlit.

---

## 🧩 Challenges Faced
- Managing annotation quality for small defect regions.  
- Balancing dataset across multiple defect classes.  
- Ensuring smooth inference during real-time deployment.

---

## 🖥️ Deployment
- **App Link:** 🔸(Add your Streamlit app link here once deployed)  
- **Demo Video:** 🔸(Add YouTube or Loom link, if available)

---

**Run Locally:**
```bash
git clone https://github.com/yourusername/solar-panel-defect-detection.git
cd solar-panel-defect-detection
pip install -r requirements.txt
python app.py
```

---

## 🏁 Results

The YOLOv9c-seg model achieved 77% mAP@50, efficiently detecting multiple types of defects in solar panels with accurate segmentation.
The Streamlit dashboard enables easy visualization and inspection support for maintenance engineers.

---


## 🧑‍💻 Author

- Name: Arshia Kauser Shahzaan
- LinkedIn: linkedin.com/in/arshiakauser
- GitHub: github.com/arshiakauser
- Email: 🔸(Optional — add your professional email)

---

🪪 License

This project is open-source and available under the No License.

---

## 🌟 Acknowledgements

- Dataset preparation support via Roboflow
- YOLOv9c-seg implementation by Ultralytics community
- Streamlit for enabling interactive AI app deployment
