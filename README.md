# Solar-Panel-Defect-Detection
🔹 Solar Panel Defect Detection using **YOLOv9c-seg**

---

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
- **Source:** Custom dataset prepared using [Roboflow](https://roboflow.com/). 
- **Total Images:** 1,500+ labeled images.  
- **Classes:** 6 defect categories — *Bird-drop, Defective, Dusty, Electrical-Damage, Non-Defective, Physical-Damage*.  
- **Structure:**
```text
train/
  images/
  labels/
val/
  images/
  labels/
test/
  images/
  labels/
```
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

## 📂 Project Structure
```text
solar-panel-defect-detection/
│
├── app/
│   ├── app.py                  # Streamlit web application
│   ├── requirements_app.txt    # (Optional) app-specific dependencies
│
├── train/
│   ├── train_yolov9.ipynb      # Model training notebook (Google Colab)
│   ├── requirements_train.txt  # (Optional) training dependencies
│
├── yolov9_best.pt              # Trained YOLOv9 model weights
├── requirements.txt            # Combined environment dependencies
├── README.md                   # Project documentation (this file)
└── sample_images/              # Example input images
```
---

## 💾 Clone the Repository
**Clone the repo**
```bash
git clone https://github.com/yourusername/solar-panel-defect-detection.git
```
**Navigate into the project folder**
```bash
cd solar-panel-defect-detection
```
---

## ⚙️ Environment Setup (Combined)

You can set up one environment for both training and running the app.

🧩 1. Create and Activate Environment
```bash
#Create a virtual environment
python -m venv solar_env

#Activate environment
# Windows:
solar_env\Scripts\activate
# macOS/Linux:
source solar_env/bin/activate
```
📦 2. Install Dependencies

Install all dependencies from the combined file:
```bash
pip install -r requirements.txt
```
Example requirements.txt:
```
ultralytics==8.2.77
streamlit
opencv-python
pillow
numpy
matplotlib
torch
torchvision
torchaudio
jupyter
pandas
```
---

## 🚀 Model Training (YOLOv9)

- The model was trained in Google Colab using YOLOv9.
- If you wish to retrain:
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arshiakauser/solar-panel-defect-detection/blob/main/train/train_yolov9.ipynb)

**Steps to Run Locally:**
```bash
cd train
jupyter notebook train_yolov9.ipynb
```
- Update dataset paths and parameters as needed.
- Train the model.
- Export the best model as yolov9_best.pt and move it to the project root.

**Notes:**
- You can use Google Colab GPU runtime for faster training.
- Ensure your dataset is structured as:
```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```
---

## 📊 Model Performance
| Metric    | Bounding Box(B) | Segmentation Mask (M) |
|---------- |-----------------|-----------------------|
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

**Run the streamlit app Locally:**
```bash
cd app
streamlit run app.py
```
**App Features:**

- Upload and preview solar panel images
- Real-time defect detection using trained YOLOv9 model
- Class-wise count of detected defects
- Download annotated output (optional)
- User-friendly and responsive Streamlit UI

## 🧮 Example Output
- **App Link:** 🔸 http://localhost:8501/ 
- **Demo Video:** 🔸 
  
| Original Image                               |	                              Detected Output |

<img width="977" height="603" alt="Solar_Panel_Defect_Detection_App_pic" src="https://github.com/user-attachments/assets/827afb5a-8bb6-4dbc-b4c3-39314a0e95bf" />

Detected Classes Example:
- Bird Drop: 17
- Defective: 6

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

This project is open-source and available under the MIT License.

---

## 🌟 Acknowledgements

- Dataset preparation support via Roboflow
- YOLOv9c-seg implementation by Ultralytics community
- Streamlit for enabling interactive AI app deployment
