# Image-to-Excel Marks Automation

## 📌 Project Overview

This project automates the process of extracting student marks from scanned exam documents (PDFs containing images) and converting them into structured Excel sheets. The main goal is to reduce the manual effort, time, and human errors involved in entering marks into digital systems.

Traditionally, marks are entered manually by two people: one reading the marks aloud and the other typing them into an Excel sheet. With a large number of classes and exams, this becomes extremely time-consuming. This project replaces that workflow with an automated, image-based pipeline.

---

## 🧠 Problem Statement

- Each **class** has \~70 students
- Each **student** writes **4 mid exams per Year**
- Each **mid exam** contains **5 subjects**
- Each subject has **13 internal marks** to be entered
- Around **128 classes** exist per academic year
- It takes **20–30 minutes per mid per class**

So the entire time is around **1000+ hours of work** every year

---

## ✅ Solution

This project automatically:

1. Accepts a **PDF uploaded by the user**
2. Converts the PDF into **individual images**
3. Processes each image (each image contains **2 tables of marks**)
4. Detects and extracts the numerical marks from the tables
5. Stores the extracted data in a **CSV file**
6. Converts the CSV into a **formatted Excel (.xlsx) file**

This reduces human effort, speeds up processing, and minimizes data-entry errors.

---

## 🔄 Workflow Pipeline

1. **PDF Input**\
   User uploads a PDF containing scanned exam sheets.

2. **PDF to Image Conversion**\
   Each page of the PDF is converted into an image.

3. **Table Detection**\
   Each image contains exactly **two tables**. The tables are detected and isolated.

4. **Marks Extraction**\
   Numerical values (marks) inside the tables are extracted using image processing and OCR techniques.

5. **CSV Generation**\
   Extracted marks are stored in a structured CSV format.

6. **Excel Sheet Creation**\
   The CSV data is converted into a clean and organized Excel sheet, ready for upload to the college system.

---

## 🛠️ Technologies Used

- **Python**
- **Deep Learning** - For data extraction
- **OpenCV** – Image preprocessing and table handling
- **Convolutional Neural Network's** - For digit recognission
- **PDF2Image** – PDF to image conversion
- **OCR (Tesseract / CNN-based model)** – Number extraction
- **NumPy & Pandas** – Data handling
- **OpenPyXL** – Excel file generation

---

## 📈 Impact

- 🚀 Reduces processing time from **hours to minutes**
- 👥 Eliminates the need for multiple people per task
- ❌ Minimizes human errors
- 📊 Produces consistent and structured Excel files
- 🔁 Scalable to hundreds of classes and exams

---

## 📂 Output Format

- **CSV File** – Intermediate structured data
- **Excel (.xlsx)** – Final formatted marks sheet

---

## 🔮 Future Enhancements

- Support for different table layouts
- Direct database integration
- Web-based UI for uploads and downloads
- Error detection and confidence scoring for extracted marks

---

## 📜 Conclusion

This project significantly optimizes the academic marks entry workflow by leveraging image processing and automation. It is designed to be scalable, efficient, and practical for real-world academic environments where large volumes of exam data must be digitized accurately.

