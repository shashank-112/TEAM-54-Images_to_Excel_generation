# Digit Recognition for Marks Extraction

This project uses a **CNN-based digit recognition model** to automatically extract and recognize marks written inside predefined boxes from scanned documents or images.

To ensure accurate predictions, please follow the instructions below **strictly**.

---

## ⚠️ Important Instructions (Prerequisites)

The model is trained under specific constraints. Any violation of the following rules may lead to incorrect or failed detection.

1. **Write digits only inside the designated boxes.**  
   Writing outside the boxes will not be detected.

2. **Do NOT use fractions or decimal values** such as `1/2`, `0.5`, `.5`, etc.  
   The CNN model supports **only whole numbers**.

3. **Do not strike out, overwrite, or rewrite a digit elsewhere.**  
   If a number is crossed out and rewritten in another location, it will not be recognized.

4. **Write strictly inside the box area only.**  
   Any writing outside the defined box region will be ignored.

5. **Allowed values are only:**  
   - Digits from **0 to 10**, or  
   - **Blank (empty) box**  

   This results in a total of **12 valid classes** (0–10 + blank).

6. **Only marks boxes are processed.**  
   The system does **not** detect or process **PO or CO boxes**.

7. **Only sigular digits are accepted**  
   i.e 3 is **acceptable** ,03 is **not accetable**.

---

## 📌 Notes

- Use clear and legible handwriting.
- Avoid touching box borders while writing digits.
- Ensure proper image quality (no blur or heavy shadows).

---

## 🖼️ Examples

> Add example images here to demonstrate:
> - Correct input  
> - Incorrect input  
> - Blank boxes  