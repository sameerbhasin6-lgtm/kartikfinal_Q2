import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
import os
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dynamic Pricing Strategy Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
    .metric-container { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-left: 5px solid #3b82f6; text-align: center; }
    .metric-label { font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #1e293b; margin: 5px 0; }
    .metric-sub { font-size: 0.85rem; color: #16a34a; font-weight: 600; }
    .insight-card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; height: 100%; }
    .recommendation-box { background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 15px; margin-bottom: 15px; border-radius: 0 8px 8px 0; font-size: 0.9rem; color: #334155; }
    .price-card { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 15px; text-align: center; transition: transform 0.2s; }
    .price-card:hover { transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .price-title { font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; height: 35px; display: flex; align-items: center; justify-content: center; }
    .price-val { font-size: 1.25rem; font-weight: 800; color: #0f172a; }
    .bundle-highlight { background: linear-gradient(135deg, #2563eb, #1d4ed8); border: none; color: white; transform: scale(1.05); box-shadow: 0 8px 15px -3px rgba(37, 99, 235, 0.4); }
    .bundle-highlight .price-title { color: #bfdbfe; }
    .bundle-highlight .price-val { color: white; }
</style>
""", unsafe_allow_html=True)

# --- 1. ROBUST DATA LOADING ---
def robust_numeric_converter(series):
    """
    Forces a column to numeric, handling currencies/commas.
    Returns the series if it contains mostly numbers, else None.
    """
    # Convert to string, remove currency symbols/commas
    clean = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
    
    # Coerce to numeric (errors become NaN)
    numeric = pd.to_numeric(clean, errors='coerce')
    
    # Check if the column is actually useful (has > 20% valid numbers)
    # This is loose to allow for messy data but strict enough to drop "Names"
    valid_count = numeric.notna().sum()
    if valid_count / len(series) > 0.2:
        return numeric
    return None

@st.cache_data
def load_data(uploaded_file):
    """
    Loads and cleans data, keeping only valid price columns.
    """
    df = None
    if uploaded_file is not None:
        try:
            filename = uploaded_file.name.lower()
            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin1')
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        # Default Data (Dheeraj.csv content embedded)
        data = """Samsung_Smartphone,Samsung_Smart_TV_43in,Samsung_Smart_Watch,Samsung_Washing_Machine,Samsung_AC_1.5_Tonne
66262,25237,20607,37091,56028
101376,58390,33686,28562,52459
45595,35585,16770,23890,51438
48871,50971,38126,42669,47137
79338,54120,11850,33593,38939
54050,40470,26266,43422,51944
71852,49926,31845,36007,46720
51046,38983,23013,45371,43537
36667,69382,9967,42884,52488
72648,62863,15041,43710,45636
30804,40786,9167,34657,38761
35599,48543,20830,30918,42012
61681,39124,30525,35927,51014
88856,63293,21892,41002,39001"""
        df = pd.read_csv(StringIO(data))

    if df is not None:
        # 1. Create a new clean dataframe
        clean_df = pd.DataFrame()
        
        for col in df.columns:
            # Try to convert each column
            numeric_col = robust_numeric_converter(df[col])
            if numeric_col is not None:
                clean_df[col] = numeric_col
        
        # 2. Final Cleanup
        # Remove rows that are completely empty (all NaNs)
        clean_df = clean_df.dropna(how='all')
        
        # Fill remaining NaNs with 0 (assuming 0 WTP if blank) to avoid dropping useful rows
        clean_df = clean_df.fillna(0)
        
        if clean_df.empty or clean_df.shape[1] == 0:
            st.error("❌ No valid numeric data found. Please check your file columns.")
            return None
            
        return clean_df
        
    return None

# --- 2. OPTIMIZATION LOGIC ---
def calculate_baseline(df, products):
    total_rev = 0
    for prod in products:
        wtp = df[prod].values
        candidates = np.unique(wtp)
        candidates = candidates[candidates > 0] # Ignore 0s
        if len(candidates) == 0: continue
        
        best_r = 0
        for p in candidates:
            r = p * np.sum(wtp >= p)
            if r > best_r: best_r = r
        total_rev += best_r
    return total_rev

@st.cache_data(show_spinner=False)
def run_solver(df, products):
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)

    def objective(prices):
        indiv_prices = np.array(prices[:n_prods])
        bundle_price = prices[n_prods]
        
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bundle_price
        
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        buy_indiv = (~buy_bundle) & (surplus_indiv > 0)
        
        rev_bundle = np.sum(buy_bundle) * bundle_price
        items_bought_mask = (wtp_matrix >= indiv_prices) & buy_indiv[:, None]
        rev_indiv = np.sum(items_bought_mask * indiv_prices)

        return -(rev_bundle + rev_indiv)

    # Bounds
    bounds = []
    for i in range(n_prods):
        max_val = np.max(wtp_matrix[:, i])
        upper = max_val * 1.5 if max_val > 0 else 1000
        bounds.append((0, upper))
    
    max_bundle = np.max(bundle_sum_values)
    bounds.append((0, max_bundle if max_bundle > 0 else 1000))

    res = differential_evolution(objective, bounds, strategy='best1bin', maxiter=50, popsize=15, tol=0.01, seed=42)
    return res.x, -res.fun

def get_customer_details(df, products, opt_prices):
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    indiv_prices = opt_prices[:n_prods]
    bundle_price = opt_prices[n_prods]
    
    rows = []
    for i in range(len(df)):
        s_indiv = np.sum(np.maximum(wtp_matrix[i] - indiv_prices, 0))
        s_bundle = bundle_sum_values[i] - bundle_price
        
        decision = "None"
        revenue = 0
        surplus = 0
        items = "-"
        
        if s_bundle >= s_indiv and s_bundle >= 0:
            decision = "Bundle"
            revenue = bundle_price
            surplus = s_bundle
            items = "Full Bundle"
        elif s_indiv > 0:
            decision = "Individual"
            surplus = s_indiv
            bought_idx = np.where(wtp_matrix[i] >= indiv_prices)[0]
            # Clean item names for display
            clean_names = [str(products[k]).replace("Samsung_", "") for k in bought_idx]
            items = ", ".join(clean_names)
            revenue = np.sum(indiv_prices[bought_idx])
            
        rows.append({
            "ID": i + 1,
            "Decision": decision,
            "Items Bought": items,
            "Revenue": revenue,
            "Surplus": surplus
        })
    return pd.DataFrame(rows)

def generate_demand_data(df, products, opt_prices):
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    indiv_prices = opt_prices[:n_prods]
    
    max_val = np.max(bundle_sum_values) if len(bundle_sum_values) > 0 else 100
    prices = np.linspace(0, max_val, 50)
    demand = []
    
    for p in prices:
        s_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        s_bundle = bundle_sum_values - p
        buy_bundle = (s_bundle >= s_indiv) & (s_bundle >= 0)
        demand.append(np.sum(buy_bundle))
        
    return pd.DataFrame({"Price": prices, "Demand": demand})

# --- MAIN APP ---
def main():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4149/4149665.png", width=50)
        st.title("Pricing Controls")
        st.markdown("### 📂 Upload Data")
        sidebar_file = st.file_uploader("Upload File (CSV or Excel)", type=['csv', 'xlsx', 'xls'], key="sidebar_uploader")
        st.markdown("---")
        st.info("Auto-Cleaner Active: Text columns are ignored. Missing values are treated as 0.")

    st.title("🎯 Pricing Strategy Optimizer")
    st.markdown("Dynamic Mixed-Bundling Simulation & Analytics")
    
    uploaded_file = sidebar_file
    if uploaded_file is None:
        st.warning("⚠️ Using Default Data (Dheeraj.csv). Upload your own file to change analysis.")
        
    df = load_data(uploaded_file)
    
    if df is not None and not df.empty:
        products = df.columns.tolist()

        with st.spinner("Running AI Solver... Finding optimal price anchors..."):
            baseline_rev = calculate_baseline(df, products)
            opt_prices, max_rev = run_solver(df, products)
            cust_df = get_customer_details(df, products, opt_prices)
            
            total_surplus = cust_df['Surplus'].sum()
            # Prevent divide by zero if baseline is 0
            uplift = ((max_rev - baseline_rev) / baseline_rev) * 100 if baseline_rev > 0 else 0
            
            bundle_price = opt_prices[-1]
            indiv_sum = np.sum(opt_prices[:-1])
            # Prevent divide by zero
            discount = ((indiv_sum - bundle_price) / indiv_sum) * 100 if indiv_sum > 0 else 0
            bundle_adoption = len(cust_df[cust_df['Decision'] == 'Bundle'])

        # KPIS
        st.markdown("### 1. Financial Performance")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Optimized Revenue</div>
                <div class="metric-value">₹{max_rev:,.0f}</div>
                <div class="metric-sub">▲ {uplift:.1f}% Uplift (vs Linear)</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-container" style="border-left-color: #16a34a;">
                <div class="metric-label">Consumer Surplus</div>
                <div class="metric-value">₹{total_surplus:,.0f}</div>
                <div class="metric-sub" style="color:#64748b;">Customer Value Retained</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-container" style="border-left-color: #f59e0b;">
                <div class="metric-label">Bundle Conversion</div>
                <div class="metric-value">{bundle_adoption}/{len(df)}</div>
                <div class="metric-sub" style="color:#64748b;">Customers Choosing Bundle</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("---")

        # INSIGHTS
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.subheader("💡 Strategic Recommendations")
            rec_text = "High Anchor Pricing" if discount > 15 else "Value Bundling"
            
            # Avoid division by zero for products length
            unit_price = bundle_price/len(products) if len(products) > 0 else 0
            
            st.markdown(f"""
            <div class="insight-card">
                <div class="recommendation-box">
                    <strong>Strategy: {rec_text}</strong><br>
                    The AI recommends a <strong>{discount:.1f}% discount</strong> on the bundle. 
                    Individual prices are set high to "anchor" value perception.
                </div>
                <div class="recommendation-box" style="border-left-color: #ec4899; background-color: #fdf2f8;">
                    <strong>Marketing Angle</strong><br>
                    Highlight the savings of <strong>₹{(indiv_sum - bundle_price):,.0f}</strong> compared to buying items separately.
                </div>
                <div class="recommendation-box" style="border-left-color: #f59e0b; background-color: #fffbeb;">
                    <strong>Competitive Edge</strong><br>
                    Effective unit price is <strong>₹{unit_price:,.0f}</strong> per item.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.subheader("👥 Customer Purchase Breakdown")
            st.dataframe(
                cust_df,
                column_config={
                    "Revenue": st.column_config.NumberColumn(format="₹%d"),
                    "Surplus": st.column_config.ProgressColumn(format="₹%d", min_value=0, max_value=int(cust_df['Surplus'].max()) if not cust_df.empty else 100),
                    "Decision": st.column_config.TextColumn(),
                },
                use_container_width=True,
                height=380,
                hide_index=True
            )

        st.write("---")

        # PRICING
        st.subheader("🏷️ Optimal Pricing Mix")
        cols = st.columns(len(products) + 1)
        for i, prod in enumerate(products):
            p_val = opt_prices[i]
            name = str(prod).replace("Samsung_", "").replace("_", " ")
            with cols[i]:
                st.markdown(f"""
                <div class="price-card">
                    <div class="price-title">{name}</div>
                    <div class="price-val">₹{p_val:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                
        with cols[-1]:
            st.markdown(f"""
            <div class="price-card bundle-highlight">
                <div class="price-title">ALL-IN BUNDLE</div>
                <div class="price-val">₹{bundle_price:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("---")

        # DEMAND CURVE
        st.subheader("📉 Bundle Demand Sensitivity")
        demand_data = generate_demand_data(df, products, opt_prices)
        fig = px.line(demand_data, x="Price", y="Demand", 
                      labels={"Price": "Bundle Price (₹)", "Demand": "Number of Buyers"})
        fig.add_vline(x=bundle_price, line_dash="dash", line_color="green", annotation_text="Optimal Price")
        fig.update_traces(line_color="#3b82f6", fill="tozeroy", fillcolor="rgba(59, 130, 246, 0.1)")
        fig.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Unable to read data. Ensure your file has columns with Willingness-To-Pay numbers.")

if __name__ == "__main__":
    main()
