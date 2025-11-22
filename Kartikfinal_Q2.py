import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dynamic Pricing Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main container styling */
    .main { background-color: #f8fafc; }
    
    /* Metric Cards */
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 5px solid #3b82f6;
    }
    .metric-label { font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 2rem; color: #1e293b; font-weight: 700; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; }
    .positive { color: #16a34a; }
    
    /* Insight Box */
    .insight-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .insight-header { font-size: 1.1rem; font-weight: bold; color: #1e293b; margin-bottom: 15px; display: flex; align-items: center; }
    .recommendation {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        padding: 12px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }
    
    /* Pricing Cards */
    .pricing-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
    .price-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }
    .price-card:hover { transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .price-title { font-size: 0.8rem; color: #64748b; font-weight: bold; text-transform: uppercase; height: 40px; display: flex; align-items: center; justify-content: center; }
    .price-tag { font-size: 1.2rem; font-weight: bold; color: #0f172a; margin: 5px 0; }
    .bundle-highlight {
        background: linear-gradient(135deg, #4f46e5, #3b82f6);
        color: white !important;
        border: none;
    }
    .bundle-highlight .price-title, .bundle-highlight .price-tag { color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file)
    # Default Data (Samsung)
    csv_content = """Samsung_Smartphone,Samsung_Smart_TV_43in,Samsung_Smart_Watch,Samsung_Washing_Machine,Samsung_AC_1.5_Tonne
46371,38390,10695,22633,46883
95402,37411,5179,28348,54920
32191,55430,19183,29138,60153
114921,47641,28643,48809,56291
46083,50986,13906,50804,49542
64405,55260,12398,29402,26565
103194,49171,17219,43524,38325
23787,58884,30076,46407,41890
93946,41870,17478,33432,46575
58329,62791,23984,47980,45185
68996,32708,17702,56427,49056
32188,52276,15916,53980,46529
55920,64810,17258,42960,54298
66398,39468,25363,33289,32665
50826,42141,24927,29293,67821
32615,41404,17073,57167,58115
38382,49750,13900,30765,44412
84204,44885,10859,34863,46128
20665,32725,3779,30417,64284
69383,64890,10091,36191,62398
117683,37521,11550,24967,46031
14969,48720,23064,45343,33189
33053,50829,27022,36404,66709
66450,27818,24401,34430,52347
24070,31982,23999,33078,53377
59917,72289,37707,38257,42576
67266,53457,34118,44263,57226
59797,27095,13925,49530,50043
45257,59211,11026,46647,48000
68386,47874,11250,35024,54738
60248,30620,30173,46334,45454
36176,54107,9741,36971,46979
57996,47575,15294,39824,52310
87763,37532,19157,40865,51989
65853,63974,21090,33966,64907
51822,40906,20662,35954,50164
42733,40659,25943,41403,44002
26835,40227,20095,31647,42982
38412,57901,28261,43126,49510
16075,55157,9129,32851,36597
109151,65738,28578,38405,50746
37095,38547,20525,46484,45953
79637,58819,12948,19569,44899
33590,50056,35462,29707,39715
40600,41517,32533,37780,48608
48451,27292,28054,24772,31506
42894,60501,30957,28602,51197
77875,39015,16093,43198,42923
100704,52191,13057,19820,60756
86872,49220,12011,39054,49334
32512,22583,29193,20544,55947
75670,76553,27336,25214,58427
32095,60253,35800,33743,59563
92391,47160,28152,53704,59209
54864,54090,21559,37396,40926
43581,46846,10948,31763,43472
32763,56840,11555,34240,53787
16242,34194,17935,37974,49593
38604,45902,25344,39308,51248
92493,42434,11771,30527,52969
77761,46314,31159,28325,36337
103696,46029,22897,36171,38760
28234,34650,14775,40011,51945
60093,72973,29720,45635,51637
39130,24820,20731,38596,57181
56579,23887,24289,41398,53398
42669,43667,16157,35905,40285
40592,51765,16315,37921,33878
58761,36415,5366,39748,45973
34595,48690,19325,29584,45437
51650,55871,25125,51387,55531
64566,31878,34629,32629,52805
30566,74280,30015,38911,59275
95286,42473,9214,37775,48817
54048,52596,35755,47155,43021
80148,32887,11146,25111,48817
64812,55300,7272,48587,40570
69482,61695,24760,26104,33252
70019,32566,30669,26405,48709
86847,33740,33184,47551,46570
61688,40532,27276,30211,31021
40302,24231,12538,39101,54161
28121,74587,11269,35435,52109
61135,50958,8116,51400,34111
24797,51293,20335,44342,53845
38218,70716,18221,32363,34652
24153,61218,9965,45512,49132
56993,60249,17750,29255,51561
33002,42566,19994,39203,48693
61447,42411,23991,43282,54922
77100,40919,16721,41725,45826
80408,42804,14889,40347,40478
42051,43542,22414,46169,48392
56998,34491,22909,37655,44621
57987,43106,13957,51021,38045
51892,45914,5966,26416,48021
64909,41541,32912,32892,40053
36330,35748,24954,44375,48001
106484,43666,12449,31650,50849
27990,46983,19956,34966,55999"""
    return pd.read_csv(StringIO(csv_content))

# --- 2. OPTIMIZATION ENGINE ---

def calculate_baseline(df, products):
    """Calculates revenue if we only use separate pricing (no bundle)."""
    total_rev = 0
    for prod in products:
        wtp = df[prod].values
        candidates = np.unique(wtp)
        best_r = 0
        for p in candidates:
            r = p * np.sum(wtp >= p)
            if r > best_r: best_r = r
        total_rev += best_r
    return total_rev

@st.cache_data(show_spinner=False)
def solve_pricing(df, products):
    """
    Simulates Excel Evolutionary Solver using Differential Evolution.
    Finds optimal [P1, P2, ..., Pn, BundlePrice].
    """
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)

    def objective(prices):
        indiv_prices = np.array(prices[:n_prods])
        bundle_price = prices[n_prods]

        # Logic: Customer chooses Max Surplus
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bundle_price
        
        # Vectorized Choice
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        buy_indiv = (~buy_bundle) & (surplus_indiv > 0)
        
        # Revenue Calculation
        rev_bundle = np.sum(buy_bundle) * bundle_price
        
        # For indiv revenue, we must check which items they bought
        # Mask of items bought by indiv buyers
        items_bought_mask = (wtp_matrix >= indiv_prices) & buy_indiv[:, None]
        rev_indiv = np.sum(items_bought_mask * indiv_prices)

        return -(rev_bundle + rev_indiv) # Minimize negative revenue

    # Set Bounds
    bounds = []
    for i in range(n_prods):
        max_w = np.max(wtp_matrix[:, i])
        bounds.append((0, max_w * 1.5)) # Allow anchor prices higher than WTP
    bounds.append((0, np.max(bundle_sum_values)))

    res = differential_evolution(objective, bounds, strategy='best1bin', maxiter=50, popsize=15, tol=0.01, seed=42)
    return res.x, -res.fun

def get_customer_breakdown(df, products, optimal_prices):
    """Generates the customer-wise decision table."""
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    
    indiv_prices = optimal_prices[:n_prods]
    bundle_price = optimal_prices[n_prods]
    
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
            items = "All Items"
        elif s_indiv > 0:
            decision = "Individual"
            surplus = s_indiv
            bought_indices = np.where(wtp_matrix[i] >= indiv_prices)[0]
            items = ", ".join([products[k] for k in bought_indices])
            revenue = np.sum(indiv_prices[bought_indices])
            
        rows.append({
            "Customer ID": i + 1,
            "Decision": decision,
            "Items Bought": items,
            "Revenue": revenue,
            "Consumer Surplus": surplus
        })
    return pd.DataFrame(rows)

def generate_demand_curve(df, products, optimal_prices):
    """Generates demand curve data by varying bundle price while keeping indiv prices fixed."""
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    indiv_prices = optimal_prices[:n_prods]
    
    # Sweep bundle price from 0 to Max Bundle Sum
    max_val = np.max(bundle_sum_values)
    price_points = np.linspace(0, max_val, 100)
    demand = []
    
    for bp in price_points:
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bp
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        demand.append(np.sum(buy_bundle))
        
    return pd.DataFrame({"Price": price_points, "Demand": demand})

# --- MAIN APP ---

def main():
    st.title("Dynamic Pricing Optimization Engine")
    st.markdown("Upload your dataset (Product WTPs) to auto-generate optimal bundling strategies.")

    # 0. Data Input
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload WTP CSV", type=["csv"])
    
    df = load_data(uploaded_file)
    products = df.columns.tolist()
    
    if st.button("Run Optimization Solver") or uploaded_file is None:
        with st.spinner("Running Differential Evolution Solver... Analyzing Customer WTPs..."):
            # Run Calculations
            baseline_rev = calculate_baseline(df, products)
            opt_prices, max_rev = solve_pricing(df, products)
            customer_df = get_customer_breakdown(df, products, opt_prices)
            
            total_surplus = customer_df['Consumer Surplus'].sum()
            uplift = ((max_rev - baseline_rev) / baseline_rev) * 100
            
            # Calculate Stats for AI Insights
            bundle_price = opt_prices[-1]
            sum_indiv_opt = np.sum(opt_prices[:-1])
            discount = ((sum_indiv_opt - bundle_price) / sum_indiv_opt) * 100
            bundle_adoption = (len(customer_df[customer_df['Decision'] == 'Bundle']) / len(df)) * 100
            
            # --- SECTION 1: METRICS ---
            st.markdown("### 1. Financial Overview")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Total Revenue (Optimized)</div>
                    <div class="metric-value">₹{max_rev:,.0f}</div>
                    <div class="metric-delta positive">▲ {uplift:.1f}% vs Separate</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                 st.markdown(f"""
                <div class="metric-box" style="border-left-color: #22c55e;">
                    <div class="metric-label">Consumer Surplus</div>
                    <div class="metric-value">₹{total_surplus:,.0f}</div>
                    <div class="metric-delta" style="color:#64748b;">Value Retained by Users</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                 st.markdown(f"""
                <div class="metric-box" style="border-left-color: #f59e0b;">
                    <div class="metric-label">Bundle Adoption</div>
                    <div class="metric-value">{bundle_adoption:.0f}%</div>
                    <div class="metric-delta" style="color:#64748b;">Conversion Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("---")

            # --- SECTION 2: SPLIT VIEW (AI & Customers) ---
            c_left, c_right = st.columns([1, 2])
            
            with c_left:
                st.subheader("2. AI Strategic Insights")
                
                # Dynamic Text Generation based on stats
                strategy_text = "Volume Driver" if discount > 15 else "Premium Extraction"
                marketing_focus = "Value-for-Money" if discount > 15 else "Exclusivity & Convenience"
                
                st.markdown(f"""
                <div class="insight-card">
                    <div class="recommendation">
                        <strong>🎯 Pricing Strategy: {strategy_text}</strong><br>
                        The solver suggests a <strong>{discount:.1f}% discount</strong> on the bundle. 
                        Individual prices are set high to act as anchors, making the bundle price of 
                        <strong>₹{bundle_price:,.0f}</strong> the rational choice for most buyers.
                    </div>
                    <div class="recommendation" style="border-left-color: #ec4899;">
                        <strong>📢 Marketing Angle: {marketing_focus}</strong><br>
                        Focus marketing on the "Total Ecosystem Savings". 
                        Highlight that buying the bundle saves <strong>₹{(sum_indiv_opt - bundle_price):,.0f}</strong> 
                        compared to individual items.
                    </div>
                    <div class="recommendation" style="border-left-color: #f59e0b;">
                        <strong>📉 Competitor Analysis</strong><br>
                        Your optimal bundle price effectively prices each item at 
                        <strong>₹{(bundle_price/len(products)):,.0f}</strong> avg. 
                        Use this unit metric to undercut single-product competitors.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_right:
                st.subheader("Customer Purchase Decisions")
                st.dataframe(
                    customer_df,
                    column_config={
                        "Customer ID": st.column_config.NumberColumn(format="#%d"),
                        "Revenue": st.column_config.NumberColumn(format="₹%d"),
                        "Consumer Surplus": st.column_config.ProgressColumn(
                            format="₹%d",
                            min_value=0,
                            max_value=int(customer_df['Consumer Surplus'].max()),
                        ),
                        "Decision": st.column_config.TextColumn(),
                    },
                    use_container_width=True,
                    height=350,
                    hide_index=True
                )

            st.write("---")

            # --- SECTION 3: PRICING MIXES ---
            st.subheader("3. Optimal Pricing Mix")
            st.markdown("The solver calculated these price points to maximize total revenue:")
            
            cols = st.columns(len(products) + 1)
            
            # Individual Prices
            for i, prod in enumerate(products):
                p_opt = opt_prices[i]
                with cols[i]:
                    st.markdown(f"""
                    <div class="price-card">
                        <div class="price-title">{prod}</div>
                        <div class="price-tag">₹{p_opt:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Bundle Price
            with cols[-1]:
                st.markdown(f"""
                <div class="price-card bundle-highlight">
                    <div class="price-title">ALL-IN BUNDLE</div>
                    <div class="price-tag">₹{bundle_price:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("---")

            # --- SECTION 4: DEMAND CURVE ---
            st.subheader("4. Bundle Demand Sensitivity")
            
            demand_data = generate_demand_curve(df, products, opt_prices)
            
            fig = px.line(
                demand_data, x="Price", y="Demand",
                title="Projected Bundle Sales at Different Price Points",
                labels={"Price": "Bundle Price (₹)", "Demand": "Number of Buyers"}
            )
            
            # Add vertical line for optimal price
            fig.add_vline(x=bundle_price, line_dash="dash", line_color="green", annotation_text="Optimal Price")
            fig.update_layout(height=400, hovermode="x unified")
            fig.update_traces(line_color='#3b82f6', fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)')
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
