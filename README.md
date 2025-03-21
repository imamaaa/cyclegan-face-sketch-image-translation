# CycleGAN for Person Face Sketches: Face-to-Sketch and Sketch-to-Face Translation  

## Overview  
This project implements **CycleGAN** for **image-to-image translation** between **real human faces and their corresponding sketches**. The model is trained to:  
- Convert a **real face image** into a **sketch**.  
- Convert a **sketch** back into a **real face image**.  

The **CycleGAN model** is trained end-to-end using the **Person Face Sketches dataset** and includes a **Flask-based API** for real-time image conversion.  

---

## Key Objectives  
- **Train CycleGAN** to map sketches to real faces & vice versa.
- **Handle memory limitations** by implementing **batch-wise data loading**.  
- **Save & resume model training** using **checkpointing**.  
- **Deploy the trained model** via a simple **Flask-based UI**.  
- **Allow users to upload images** (or capture from a camera) for conversion.  

---

## Dataset Details  
- **Dataset Source**: [Person Face Sketches (Kaggle)](https://www.kaggle.com/datasets/almightyj/person-face-sketches)  
- **Total Images**: ~23K (Paired real-face & sketch images)  
- **Training Set**: 20.7K images per category (photos & sketches)  
- **Validation Set**: 1K images per category  
- **Test Set**: 679 images per category  
- **Format**: **Grayscale sketches** paired with **RGB photos**  

---

## Repository Contents  
- `i201819_GenAI_B_A2_Q4.ipynb` → Jupyter Notebook containing CycleGAN training & inference
- `app.py` → Flask-based API to allow users to upload an image and perform real-time face-to-sketch or sketch-to-face conversion using the trained CycleGAN model
- `index.html` → Web interface for user interaction, allowing users to upload images and view the generated outputs
- `i201819_ImamaAmjad_Ass2.pdf` → Detailed analysis, methodology, and results
- `README.md` → Project documentation (to be expanded)  

For now, please refer to the **i201819_ImamaAmjad_Ass2.pdf** for **dataset details, model training methodology, and evaluation metrics**. The README will be **expanded later** with additional explanations and improvements.  

---

## Future Enhancements  
✅ **Expand the README** with dataset details, preprocessing steps, and architectural explanations  
✅ **Include sample outputs & qualitative comparisons** between real faces and generated sketches  
✅ **Add challenges faced and key learnings** section  
✅ **Optimize model training with better hyperparameters & additional GAN tricks**  
✅ **Enhance UI for a smoother user experience**  
✅ **Deploy as a web app or API for real-time inference**  

---
