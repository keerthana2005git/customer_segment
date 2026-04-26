# 🛒 Customer Segmentation App

A real-time customer segmentation dashboard built with Streamlit and K-Means Clustering.

## Features
- Upload any customer CSV or use built-in sample data
- Auto-segments customers into named business groups
- Elbow Method chart to justify cluster count
- Interactive 3D Plotly visualization
- Predict segment for any new customer instantly
- Download segmented results as CSV
- INR conversion with Lakhs/Crores formatting

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud (Free)

1. Push this folder to a GitHub repository
2. Go to https://streamlit.io/cloud
3. Sign in with GitHub
4. Click "New App"
5. Select your repo → branch → set `app.py` as the main file
6. Click Deploy — your live link is ready in 2 minutes!

## Dataset Format
Your CSV must have these columns:
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

## Tech Stack
- Python, Streamlit, Scikit-learn, Pandas, Plotly, Seaborn, Matplotlib
