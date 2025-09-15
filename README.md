# 💸 Expense Tracker with NLP + ML Model
What it involves: Collecting a dataset of expense sentences → labeling them with categories → training a text classification model (e.g., fine-tuning Count-vectorizer, or even a smaller LLM).  Pros:  Fully custom → can adapt to your own categories like Subscriptions, Rent, Groceries, EMI, etc.  Once trained, it runs locally and cheaply.

This project combines the power of **NLP** and a **custom Machine Learning text classifier** to automatically categorize expenses from plain text sentences.  

---

## 🚀 What it Involves
1. **Collecting a dataset** of expense-related sentences.  
   - Example: *"Paid ₹500 for Netflix"* → `Subscriptions`  
   - *"₹2000 rent transfer"* → `Rent`  
2. **Labeling the dataset** with predefined categories (e.g., `Subscriptions`, `Rent`, `Groceries`, `EMI`, etc.).  
3. **Training a text classification model**  
   - Options: 
     - Use a **lightweight LLM** for local inference  
     - Train a **custom ML pipeline** with Scikit-learn / spaCy 
4. **Deploying the model** for fast and cheap categorization of new expenses.

---

## ✅ Pros
- 🔹 **Fully Customizable** → Works with your own categories  
- 🔹 **Adaptable** → Learns your unique spending patterns  
- 🔹 **Local & Cheap** → Once trained, it runs offline without extra costs  
- 🔹 **Hybrid Approach** → ML model + LLM fallback  

---

## 🛠️ Tech Stack
- **Python** 🐍  
- **Scikit-learn** (ML text classification)  
- **Pandas & NumPy** (data handling)  
- **Joblib** (model saving/loading)  
- **Optional LLM API** for fallback classification
- **LLM Integration** (optional for edge cases)
- **Dataset labeling tools** (Prodi.gy / Label Studio / custom script)

---

## 📊 Example Flow
```text
Input  : "Paid ₹1200 for groceries at Big Bazaar"
Process: → Text Preprocessing → Model Inference  
Output : Category = "Groceries"
```
---
## 🚀 Deployed on Streamlit
## 📸 Screenshots

<table>
  <tr>
    <img width="1904" height="922" alt="image" src="https://github.com/user-attachments/assets/404fffb3-e9cf-444b-8188-2bf9d0540ca5" />
    <img width="1906" height="912" alt="image" src="https://github.com/user-attachments/assets/7efea1de-e9aa-4b4c-b1ac-d18557eb5949" />
    <img width="1906" height="921" alt="image" src="https://github.com/user-attachments/assets/f38c88b0-0329-475e-9c36-12da77ed9cc2" />
  </tr>
</table>


