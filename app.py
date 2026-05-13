import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SegmentAI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= COLORS =================
COLORS = {
    "bg": "#050816",
    "card": "#111827",
    "accent": "#B6FF00",
    "text": "#F3F4F6",
    "muted": "#9CA3AF",
    "green": "#22c55e",
    "yellow": "#eab308",
    "purple": "#a855f7",
    "red": "#ef4444"
}

# ================= CUSTOM CSS =================
st.markdown(f"""
<style>

.stApp {{
    background-color: {COLORS['bg']};
    color: white;
}}

section[data-testid="stSidebar"] {{
    background-color: #08110A;
    border-right: 1px solid rgba(255,255,255,0.05);
}}

.hero {{
    padding: 3rem;
    border-radius: 30px;
    background: linear-gradient(135deg,#0f172a,#1e293b);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem;
}}

.glass-card {{
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 24px;
    border-radius: 24px;
    margin-bottom: 20px;
    transition: 0.3s ease;
}}

.glass-card:hover {{
    transform: translateY(-5px);
    border: 1px solid rgba(182,255,0,0.3);
}}

.insight-title {{
    font-size: 24px;
    font-weight: 700;
    color: white;
}}

.insight-desc {{
    color: #b0b0b0;
    line-height: 1.7;
    margin-top: 10px;
}}

.green-text {{
    color: #4ADE80;
    font-weight: 600;
    margin-top: 16px;
}}

.footer {{
    text-align:center;
    color:#9CA3AF;
    padding:20px;
    margin-top:40px;
}}

</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown("""
# 🛒 SegmentAI
### QUICK-COMMERCE
""")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Dataset",
        "Analytics",
        "Clusters",
        "Personas",
        "Insights"
    ]
)

st.sidebar.markdown("---")

k = st.sidebar.slider(
    "Number of Clusters",
    2,
    10,
    5
)

# ================= HERO =================
st.markdown("""
<div class="hero">
    <h1 style="font-size:55px;">   Customer Segmentation</h1>
    <p style="font-size:22px;color:#cbd5e1;">
        Advanced customer segmentation and business intelligence dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader(
    "📤 Upload Customer CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ================= SEARCH =================
    st.sidebar.markdown("### 🔍 Search")
    search = st.sidebar.text_input("Search Customer")

    if search:
        df = df[df.astype(str).apply(
            lambda row: row.str.contains(search, case=False).any(),
            axis=1
        )]

    # ================= FEATURES =================
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    default_features = []

    if 'Annual Income (k$)' in numeric_cols:
        default_features.append('Annual Income (k$)')

    if 'Spending Score (1-100)' in numeric_cols:
        default_features.append('Spending Score (1-100)')

    features = st.sidebar.multiselect(
        "Select Features",
        numeric_cols,
        default=default_features
    )

    if len(features) >= 2:

        # ================= PREPROCESSING =================
        X = df[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ================= ELBOW METHOD =================
        wcss = []

        for i in range(1, 11):
            model = KMeans(
                n_clusters=i,
                init='k-means++',
                random_state=42
            )
            model.fit(X_scaled)
            wcss.append(model.inertia_)

        # ================= KMEANS =================
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            random_state=42
        )

        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # ================= PCA =================
        n_components = min(3, X_scaled.shape[1])

        pca = PCA(n_components=n_components)

        pca_result = pca.fit_transform(X_scaled)

        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]

        if n_components == 3:
            df['PCA3'] = pca_result[:, 2]

        # ================= OVERVIEW =================
        if menu == "Overview":

            st.subheader("📊 Analytics Dashboard")

            c1, c2, c3, c4, c5 = st.columns(5)

            total_customers = len(df)

            avg_income = round(
                df['Annual Income (k$)'].mean(),
                2
            ) if 'Annual Income (k$)' in df.columns else 0

            avg_spending = round(
                df['Spending Score (1-100)'].mean(),
                2
            ) if 'Spending Score (1-100)' in df.columns else 0

            cluster_count = df['Cluster'].nunique()

            male_ratio = 0

            if 'Gender' in df.columns:
                male_ratio = round(
                    (df[df['Gender'] == 'Male'].shape[0] / len(df)) * 100,
                    1
                )

            c1.metric("👥 Customers", total_customers)
            c2.metric("💰 Avg Income", avg_income)
            c3.metric("🛒 Avg Spending", avg_spending)
            c4.metric("🧠 Clusters", cluster_count)
            c5.metric("👨 Male Ratio", f"{male_ratio}%")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:

                cluster_fig = px.histogram(
                    df,
                    x='Cluster',
                    color='Cluster',
                    template='plotly_dark',
                    title='Customer Cluster Distribution'
                )

                cluster_fig.update_layout(
                    paper_bgcolor="#050816",
                    plot_bgcolor="#050816"
                )

                st.plotly_chart(cluster_fig, use_container_width=True)

            with col2:

                donut_fig = px.pie(
                    df,
                    names='Gender' if 'Gender' in df.columns else 'Cluster',
                    hole=0.6,
                    template='plotly_dark'
                )

                donut_fig.update_layout(
                    paper_bgcolor="#050816",
                    plot_bgcolor="#050816"
                )

                st.plotly_chart(donut_fig, use_container_width=True)

        # ================= DATASET =================
        if menu == "Dataset":

            st.subheader("📄 Customer Dataset")

            st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇ Download Dataset",
                df.to_csv(index=False),
                "clustered_customers.csv"
            )

        # ================= ANALYTICS =================
        if menu == "Analytics":

            st.subheader("📈 Elbow Method")

            elbow_df = pd.DataFrame({
                'K': range(1, 11),
                'WCSS': wcss
            })

            elbow_fig = px.line(
                elbow_df,
                x='K',
                y='WCSS',
                markers=True,
                template='plotly_dark'
            )

            elbow_fig.update_layout(
                paper_bgcolor="#050816",
                plot_bgcolor="#050816"
            )

            st.plotly_chart(elbow_fig, use_container_width=True)

            st.subheader("🔥 Correlation Heatmap")

            corr = df[numeric_cols].corr()

            heatmap = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns
            ))

            heatmap.update_layout(
                template='plotly_dark',
                paper_bgcolor="#050816",
                plot_bgcolor="#050816"
            )

            st.plotly_chart(heatmap, use_container_width=True)

        # ================= CLUSTERS =================
        if menu == "Clusters":

            st.subheader("🌌 PCA Cluster Visualization")

            if n_components == 3:

                scatter_3d = px.scatter_3d(
                    df,
                    x='PCA1',
                    y='PCA2',
                    z='PCA3',
                    color=df['Cluster'].astype(str),
                    hover_data=features,
                    size='Spending Score (1-100)' if 'Spending Score (1-100)' in df.columns else None,
                    template='plotly_dark'
                )

                scatter_3d.update_layout(
                    paper_bgcolor="#050816",
                    plot_bgcolor="#050816",
                    height=700
                )

                st.plotly_chart(scatter_3d, use_container_width=True)

            else:

                scatter_2d = px.scatter(
                    df,
                    x='PCA1',
                    y='PCA2',
                    color=df['Cluster'].astype(str),
                    hover_data=features,
                    size='Spending Score (1-100)' if 'Spending Score (1-100)' in df.columns else None,
                    template='plotly_dark'
                )

                scatter_2d.update_layout(
                    paper_bgcolor="#050816",
                    plot_bgcolor="#050816",
                    height=700
                )

                st.plotly_chart(scatter_2d, use_container_width=True)

            if 'Annual Income (k$)' in df.columns and 'Spending Score (1-100)' in df.columns:

                st.subheader("💸 Income vs Spending")

                income_fig = px.scatter(
                    df,
                    x='Annual Income (k$)',
                    y='Spending Score (1-100)',
                    color=df['Cluster'].astype(str),
                    size='Spending Score (1-100)',
                    hover_data=features,
                    template='plotly_dark'
                )

                income_fig.update_layout(
                    paper_bgcolor="#050816",
                    plot_bgcolor="#050816",
                    height=650
                )

                st.plotly_chart(income_fig, use_container_width=True)

        # ================= PERSONAS =================
        if menu == "Personas":

            st.subheader("🧬 Customer Personas")

            personas = [
                ("💎 Premium Customers", "High-income and high-spending loyal users."),
                ("🔥 High Spenders", "Aggressive purchasing behavior and high basket value."),
                ("🛒 Frequent Buyers", "Regular repeat customers with strong activity."),
                ("💰 Budget Browsers", "Price-sensitive users responding to discounts."),
                ("🚀 Gen-Z Shoppers", "Trend-driven younger audience.")
            ]

            for title, desc in personas:

                st.markdown(f"""
                <div class="glass-card">
                    <div class="insight-title">{title}</div>
                    <div class="insight-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        # ================= INSIGHTS =================
        if menu == "Insights":

            st.markdown("""
            <h1 style='font-size:50px;'>✨ Recommended Actions</h1>
            <p style='color:#9CA3AF;font-size:20px;'>
            Plug-and-play AI playbook for growth and retention teams.
            </p>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("""
                <div class="glass-card">
                    <div class="insight-title">👑 Launch premium subscription tier</div>
                    <div class="insight-desc">
                    Premium customers show significantly higher spending behavior.
                    Offer cashback, priority delivery and VIP rewards.
                    </div>
                    <div class="green-text">
                    ✨ Target ~22% of customer base
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">
                    <div class="insight-title">🔥 Upsell impulse purchases</div>
                    <div class="insight-desc">
                    High spenders respond strongly to combo offers and checkout recommendations.
                    </div>
                    <div class="green-text">
                    ✨ Increase AOV by ~18%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:

                st.markdown("""
                <div class="glass-card">
                    <div class="insight-title">🌱 Aggressive discount campaign</div>
                    <div class="insight-desc">
                    Budget shoppers convert effectively through flash sales and coupon strategies.
                    </div>
                    <div class="green-text">
                    ✨ Estimated +14% conversion
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card">
                    <div class="insight-title">🚀 Run Gen-Z social campaigns</div>
                    <div class="insight-desc">
                    Younger shoppers engage strongly with influencer and gamified marketing.
                    </div>
                    <div class="green-text">
                    ✨ Reach ~28% of active users
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:

    st.info("📤 Upload a CSV dataset to start customer segmentation.")

# ================= FOOTER =================
st.markdown("""
<div class="footer">
✨ SegmentAI · K-Means + PCA · Built for Quick-Commerce
</div>
""", unsafe_allow_html=True)