import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
try:
    pd.options.mode.use_inf_as_na = True
except Exception:
    pass

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 2rem; }
    .segment-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 5px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    h1 { color: #2c3e50; }
    .stDownloadButton > button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
CLUSTER_NAMES = {
    0: ("💎 Premium Customers",   "#8e44ad"),
    1: ("💰 Budget Shoppers",     "#e67e22"),
    2: ("🔥 Impulsive Buyers",    "#e74c3c"),
    3: ("🧠 Careful Spenders",    "#2980b9"),
    4: ("⚖️  Average Customers",  "#27ae60"),
}

USD_TO_INR = 83

def format_inr(value):
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    return f"₹{value/1e5:.2f} L"

def load_and_process(df_raw):
    df = df_raw.copy()
    df.rename(columns={
        'Annual Income (k$)': 'Annual Income (USDk)',
        'Spending Score (1-100)': 'Spending Score'
    }, inplace=True)
    df['Annual Income (₹)'] = df['Annual Income (USDk)'] * 1000 * USD_TO_INR
    df['Annual Income (₹ Readable)'] = df['Annual Income (₹)'].apply(format_inr)
    return df

def run_kmeans(df, k=5):
    X = df[['Annual Income (₹)', 'Spending Score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Segment'] = df['Cluster'].map(lambda c: CLUSTER_NAMES.get(c, (str(c), "#888"))[0])
    return df, X_scaled, kmeans

def compute_wcss(X_scaled):
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    return wcss

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.title("🛒 Customer Segmentation Dashboard")
st.markdown("Upload your customer CSV **or** use the built-in sample data to instantly segment customers using K-Means Clustering.")
st.markdown("---")

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    data_source = st.radio("Data Source", ["Use Sample Data", "Upload My CSV"])

    uploaded_file = None
    if data_source == "Upload My CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        st.caption("Required columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)")

    st.markdown("---")
    k_value = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5, step=1)
    show_elbow = st.checkbox("Show Elbow Method Chart", value=True)
    show_3d = st.checkbox("Show 3D Cluster Plot", value=True)

    st.markdown("---")
    st.markdown("**Made by Keerthana**")
    st.markdown("Customer Segmentation using K-Means")

# ─────────────────────────────────────────────
#  Load Data
# ─────────────────────────────────────────────
@st.cache_data
def get_sample_data():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'CustomerID': range(1, n+1),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Age': np.random.randint(18, 70, n),
        'Annual Income (k$)': np.random.randint(15, 140, n),
        'Spending Score (1-100)': np.random.randint(1, 100, n),
    })

if data_source == "Upload My CSV" and uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
elif data_source == "Use Sample Data":
    df_raw = get_sample_data()
    st.info("ℹ️ Using generated sample data (200 customers). Upload your own CSV from the sidebar.")
else:
    st.warning("👈 Please upload a CSV file from the sidebar to get started.")
    st.stop()

# ─────────────────────────────────────────────
#  Process & Cluster
# ─────────────────────────────────────────────
df = load_and_process(df_raw)
df, X_scaled, kmeans_model = run_kmeans(df, k=k_value)

# ─────────────────────────────────────────────
#  KPI Row
# ─────────────────────────────────────────────
st.subheader("📊 Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", len(df))
c2.metric("Average Age", f"{df['Age'].mean():.0f} yrs")
c3.metric("Avg Annual Income", format_inr(df['Annual Income (₹)'].mean()))
c4.metric("Avg Spending Score", f"{df['Spending Score'].mean():.0f} / 100")

st.markdown("---")

# ─────────────────────────────────────────────
#  EDA Row
# ─────────────────────────────────────────────
st.subheader("🔍 Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x='Gender', palette='coolwarm', ax=ax1)
    ax1.set_title("Gender Distribution")
    ax1.set_xlabel("")
    st.pyplot(fig1)
    plt.close()

with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.histplot(df['Age'], bins=20, kde=True, color='steelblue', ax=ax2)
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# ─────────────────────────────────────────────
#  Elbow Method
# ─────────────────────────────────────────────
if show_elbow:
    st.subheader("📐 Elbow Method — Finding Optimal K")
    wcss = compute_wcss(X_scaled)
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.plot(range(1, 11), wcss, marker='o', color='coral', linewidth=2)
    ax3.axvline(x=k_value, color='green', linestyle='--', label=f'Selected K={k_value}')
    ax3.set_xlabel("Number of Clusters (K)")
    ax3.set_ylabel("WCSS")
    ax3.set_title("Elbow Method for Optimal K")
    ax3.legend()
    st.pyplot(fig3)
    plt.close()
    st.caption("The 'elbow' point is where adding more clusters gives diminishing returns. That's the optimal K.")
    st.markdown("---")

# ─────────────────────────────────────────────
#  2D Cluster Plot
# ─────────────────────────────────────────────
st.subheader(f"🎯 Customer Segments (K={k_value})")
col_plot, col_info = st.columns([2, 1])

with col_plot:
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    colors = [CLUSTER_NAMES.get(c, (str(c), "#888"))[1] for c in sorted(df['Cluster'].unique())]
    palette = {CLUSTER_NAMES.get(c, (str(c), "#888"))[0]: CLUSTER_NAMES.get(c, (str(c), "#888"))[1]
               for c in sorted(df['Cluster'].unique())}
    sns.scatterplot(
        data=df,
        x='Annual Income (₹)', y='Spending Score',
        hue='Segment', palette=palette,
        s=80, alpha=0.85, ax=ax4
    )
    ax4.set_title("Customer Segmentation (Income vs Spending Score)")
    ax4.set_xlabel("Annual Income (₹)")
    ax4.set_ylabel("Spending Score (1-100)")
    ax4.legend(loc='upper left', fontsize=8)
    st.pyplot(fig4)
    plt.close()

with col_info:
    st.markdown("**Segment Breakdown**")
    for cluster_id in sorted(df['Cluster'].unique()):
        name, color = CLUSTER_NAMES.get(cluster_id, (f"Cluster {cluster_id}", "#888"))
        count = len(df[df['Cluster'] == cluster_id])
        pct = count / len(df) * 100
        st.markdown(f"""
        <div class="segment-card" style="border-left-color:{color}">
            <b style="color:{color}">{name}</b><br>
            <span style="font-size:13px;color:#555">{count} customers ({pct:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
#  3D Plot
# ─────────────────────────────────────────────
if show_3d:
    st.subheader("🌐 3D Interactive Cluster View")
    df['Cluster Label'] = df['Cluster'].astype(str)
    fig_3d = px.scatter_3d(
        df, x='Age', y='Annual Income (₹)', z='Spending Score',
        color='Segment',
        title="3D Customer Segmentation",
        opacity=0.8,
        height=550
    )
    fig_3d.update_traces(marker=dict(size=4))
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown("---")

# ─────────────────────────────────────────────
#  Predict Single Customer
# ─────────────────────────────────────────────
st.subheader("🔮 Predict Segment for a New Customer")
st.caption("Enter a customer's details to instantly find which segment they belong to.")

p1, p2, p3 = st.columns(3)
with p1:
    input_age = st.number_input("Age", min_value=18, max_value=100, value=30)
with p2:
    input_income = st.number_input("Annual Income (k$)", min_value=1, max_value=300, value=60)
with p3:
    input_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("🎯 Predict My Segment"):
    income_inr = input_income * 1000 * USD_TO_INR
    scaler2 = StandardScaler()
    all_X = df[['Annual Income (₹)', 'Spending Score']].values
    scaler2.fit(all_X)
    new_point = scaler2.transform([[income_inr, input_score]])
    pred_cluster = kmeans_model.predict(new_point)[0]
    seg_name, seg_color = CLUSTER_NAMES.get(pred_cluster, (f"Cluster {pred_cluster}", "#888"))
    st.markdown(f"""
    <div class="segment-card" style="border-left-color:{seg_color}; font-size:16px;">
        <b>This customer belongs to:</b><br>
        <span style="font-size:22px; color:{seg_color}; font-weight:bold">{seg_name}</span><br>
        <span style="color:#555; font-size:13px">Annual Income: {format_inr(income_inr)} &nbsp;|&nbsp; Spending Score: {input_score}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
#  Data Table + Download
# ─────────────────────────────────────────────
st.subheader("📋 Segmented Customer Data")
display_cols = ['CustomerID', 'Gender', 'Age', 'Annual Income (₹ Readable)', 'Spending Score', 'Segment']
st.dataframe(df[display_cols], use_container_width=True, height=300)

csv_out = df[display_cols].to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Segmented Data as CSV",
    data=csv_out,
    file_name="segmented_customers.csv",
    mime="text/csv"
)
