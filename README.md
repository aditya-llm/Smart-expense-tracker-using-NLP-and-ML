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
   <img width="1907" height="911" alt="image" src="https://github.com/user-attachments/assets/16d1b2f5-5b50-4cb8-bef1-c5c824c28491" />
   <img width="1901" height="912" alt="image" src="https://github.com/user-attachments/assets/40344a09-9818-4c5e-b0b1-ed416c38d205" />
   <img width="1905" height="915" alt="image" src="https://github.com/user-attachments/assets/004d34f4-1a28-4874-80e9-c9d1b6c6c17f" />
  </tr>
</table>

