"""
Professional Trading Scanner Dashboard
Built with Streamlit for interactive monitoring
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from database import DatabaseManager
from scanner import EliteSwingScanner, ScannerConfig
from watchlist_monitor import WatchlistMonitor, WatchlistConfig

# Page configuration
st.set_page_config(
    page_title="Elite Trading Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .qualified-stock {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-info {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def get_database():
    """Get database connection"""
    db = DatabaseManager()
    db.create_tables()
    return db

db = get_database()

# Sidebar
with st.sidebar:
    #st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=Elite+Scanner", use_container_width=True)
    # Logo/Header (using emoji instead of external image)
    st.markdown("# 📈 Elite Scanner")
    st.markdown("---")
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        ["📊 Dashboard", "🎯 Qualified Stocks", "👁️ Watchlist", "🔔 Alerts", 
         "📈 Scan History", "⚙️ Run Scan", "📁 Data Management"]
    )
    
    st.markdown("---")
    st.markdown("### Market Status")
    
    # Get latest scan date
    scan_history = db.get_scan_history(limit=1)
    if not scan_history.empty:
        last_scan = scan_history.iloc[0]
        st.info(f"Last Scan: {last_scan['Scan_Date'].strftime('%Y-%m-%d %H:%M')}")
        st.success(f"Market: {last_scan['Market_Regime']}")
    else:
        st.warning("No scans yet")

# Main content area
st.markdown('<p class="main-header">📈 Elite Swing Trading Scanner</p>', unsafe_allow_html=True)
st.markdown("---")

# Dashboard Page
if page == "📊 Dashboard":
    st.header("Dashboard Overview")
    
    # Get latest scan results
    latest_results = db.get_latest_scan_results()
    qualified_stocks = db.get_latest_scan_results(qualified_only=True)
    watchlist = db.get_active_watchlist()
    alerts = db.get_unread_alerts()
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Qualified Stocks",
            value=len(qualified_stocks) if not qualified_stocks.empty else 0,
            delta="Ready to Trade"
        )
    
    with col2:
        st.metric(
            label="Watchlist Stocks",
            value=len(watchlist) if not watchlist.empty else 0,
            delta="Tracking"
        )
    
    with col3:
        st.metric(
            label="Unread Alerts",
            value=len(alerts) if not alerts.empty else 0,
            delta="New"
        )
    
    with col4:
        total_scanned = latest_results['Symbol'].nunique() if not latest_results.empty else 0
        st.metric(
            label="Total Stocks Scanned",
            value=total_scanned,
            delta="Latest Scan"
        )
    
    st.markdown("---")
    
    # Two column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📌 Top Qualified Stocks")
        
        if not qualified_stocks.empty:
            # Display top 10 qualified stocks
            display_df = qualified_stocks.head(10)[
                ['Symbol', 'Current_Price', 'Total_Score', 'Filters_Passed', 
                 'RS_Composite', 'Dist_to_Resistance_%', 'Risk_%']
            ].copy()
            
            display_df.columns = ['Symbol', 'Price', 'Score', 'Filters', 'RS', 'To Breakout %', 'Risk %']
            display_df['Price'] = display_df['Price'].round(2)
            display_df['Score'] = display_df['Score'].round(1)
            display_df['RS'] = display_df['RS'].round(2)
            display_df['To Breakout %'] = display_df['To Breakout %'].round(2)
            display_df['Risk %'] = display_df['Risk %'].round(2)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = qualified_stocks.to_csv(index=False)
            st.download_button(
                label="📥 Download Qualified Stocks CSV",
                data=csv,
                file_name=f"qualified_stocks_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No qualified stocks in latest scan")
    
    with col_right:
        st.subheader("🔔 Recent Alerts")
        
        if not alerts.empty:
            for _, alert in alerts.head(5).iterrows():
                severity = alert['Severity']
                alert_class = f"alert-{severity.lower()}"
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <strong>{alert['Symbol']}</strong> - {alert['Type']}<br>
                    <small>{alert['Message']}</small><br>
                    <small>{alert['Date'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No new alerts")
    
    st.markdown("---")
    
    # Charts
    st.subheader("📊 Analytics")
    
    if not latest_results.empty:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Score distribution
            fig = px.histogram(
                latest_results,
                x='Total_Score',
                nbins=20,
                title='Score Distribution',
                labels={'Total_Score': 'Score', 'count': 'Number of Stocks'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            # Filters passed distribution
            filter_counts = latest_results['Filters_Passed'].value_counts().sort_index()
            fig = px.bar(
                x=filter_counts.index,
                y=filter_counts.values,
                title='Stocks by Filters Passed',
                labels={'x': 'Filters Passed', 'y': 'Number of Stocks'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Qualified Stocks Page
elif page == "🎯 Qualified Stocks":
    st.header("Qualified Stocks - Ready for Trading")
    
    qualified = db.get_latest_scan_results(qualified_only=True)
    
    if not qualified.empty:
        st.success(f"Found {len(qualified)} qualified stocks")
        
        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            min_score = st.slider("Minimum Score", 0, 100, 70)
        
        with col_f2:
            max_risk = st.slider("Maximum Risk %", 0.0, 20.0, 10.0)
        
        with col_f3:
            near_breakout_only = st.checkbox("Near Breakout Only")
        
        # Apply filters
        filtered = qualified[qualified['Total_Score'] >= min_score]
        filtered = filtered[filtered['Risk_%'] <= max_risk]
        
        if near_breakout_only:
            filtered = filtered[filtered['Near_Breakout'] == True]
        
        st.markdown(f"**Showing {len(filtered)} stocks after filters**")
        
        # Display detailed view
        for _, stock in filtered.iterrows():
            with st.expander(f"📈 {stock['Symbol']} - Score: {stock['Total_Score']:.1f}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Price Information**")
                    st.write(f"Current Price: ₹{stock['Current_Price']:.2f}")
                    st.write(f"Stop Loss: ₹{stock['Stop_Loss']:.2f}")
                    st.write(f"Risk: {stock['Risk_%']:.2f}%")
                
                with col2:
                    st.markdown("**Technical Indicators**")
                    st.write(f"RSI: {stock['RSI_14']:.1f}")
                    st.write(f"MA 21: ₹{stock['MA_21']:.2f}")
                    st.write(f"MA 50: ₹{stock['MA_50']:.2f}")
                
                with col3:
                    st.markdown("**Relative Strength**")
                    st.write(f"RS 3M: {stock['RS_3M']:.2f}")
                    st.write(f"RS 6M: {stock['RS_6M']:.2f}")
                    st.write(f"RS Composite: {stock['RS_Composite']:.2f}")
                
                st.markdown("---")
                st.write(f"**Resistance:** ₹{stock['Resistance']:.2f} ({stock['Dist_to_Resistance_%']:.2f}% away)")
                st.write(f"**Filters Passed:** {stock['Filters_Passed']}/7")
                st.write(f"**VCP Pattern:** {'✅ Yes' if stock['VCP_Qualified'] else '❌ No'}")
                st.write(f"**Near Breakout:** {'✅ Yes' if stock['Near_Breakout'] else '❌ No'}")
    else:
        st.warning("No qualified stocks found in latest scan")

# Watchlist Page
elif page == "👁️ Watchlist":
    st.header("Watchlist - Stocks in Progress")
    
    watchlist = db.get_active_watchlist()
    
    if not watchlist.empty:
        st.info(f"Tracking {len(watchlist)} stocks")
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ["Score", "Filters Passed", "Days Tracked", "Gain %"]
        )
        
        sort_column_map = {
            "Score": "Score",
            "Filters Passed": "Filters_Passed",
            "Days Tracked": "Days_Tracked",
            "Gain %": "Gain_%"
        }
        
        watchlist_sorted = watchlist.sort_values(
            sort_column_map[sort_by],
            ascending=False
        )
        
        # Display watchlist
        for _, stock in watchlist_sorted.iterrows():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                if stock['Is_Qualified']:
                    st.markdown(f"### ✅ {stock['Symbol']}")
                else:
                    st.markdown(f"### 👁️ {stock['Symbol']}")
            
            with col2:
                st.metric("Filters", f"{stock['Filters_Passed']}/{stock['Max_Filters']}")
            
            with col3:
                st.metric("Score", f"{stock['Score']:.1f}")
            
            with col4:
                gain = stock['Gain_%']
                st.metric("Gain", f"{gain:.1f}%", delta=f"{gain:.1f}%")
            
            st.markdown(f"**Tracked for:** {stock['Days_Tracked']} days | **Entry:** ₹{stock['Entry_Price']:.2f} | **Current:** ₹{stock['Current_Price']:.2f}")
            st.markdown("---")
    else:
        st.warning("No stocks in watchlist")

# Alerts Page
elif page == "🔔 Alerts":
    st.header("Trading Alerts")
    
    # Tabs for different alert types
    tab_all, tab_critical, tab_warning, tab_info = st.tabs(
        ["All Alerts", "Critical", "Warning", "Info"]
    )
    
    alerts = db.get_unread_alerts()
    
    with tab_all:
        if not alerts.empty:
            st.dataframe(
                alerts[['Date', 'Symbol', 'Type', 'Severity', 'Message']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No unread alerts")
    
    with tab_critical:
        critical = alerts[alerts['Severity'] == 'CRITICAL'] if not alerts.empty else pd.DataFrame()
        if not critical.empty:
            for _, alert in critical.iterrows():
                st.markdown(f"""
                <div class="alert-critical">
                    <strong>{alert['Symbol']}</strong> - {alert['Type']}<br>
                    {alert['Message']}<br>
                    <small>{alert['Date'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No critical alerts")
    
    with tab_warning:
        warning = alerts[alerts['Severity'] == 'WARNING'] if not alerts.empty else pd.DataFrame()
        if not warning.empty:
            for _, alert in warning.iterrows():
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>{alert['Symbol']}</strong> - {alert['Type']}<br>
                    {alert['Message']}<br>
                    <small>{alert['Date'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No warning alerts")
    
    with tab_info:
        info = alerts[alerts['Severity'] == 'INFO'] if not alerts.empty else pd.DataFrame()
        if not info.empty:
            for _, alert in info.iterrows():
                st.markdown(f"""
                <div class="alert-info">
                    <strong>{alert['Symbol']}</strong> - {alert['Type']}<br>
                    {alert['Message']}<br>
                    <small>{alert['Date'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No info alerts")

# Scan History Page
elif page == "📈 Scan History":
    st.header("Scan Execution History")
    
    history = db.get_scan_history(limit=20)
    
    if not history.empty:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )
        
        # Chart of qualified stocks over time
        fig = px.line(
            history,
            x='Scan_Date',
            y='Qualified',
            title='Qualified Stocks Over Time',
            labels={'Scan_Date': 'Date', 'Qualified': 'Number of Qualified Stocks'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No scan history available")

# Run Scan Page
elif page == "⚙️ Run Scan":
    st.header("Run New Scan")
    
    st.warning("⚠️ Running a full scan may take 10-30 minutes depending on the number of stocks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scan_type = st.radio(
            "Scan Type",
            ["Full Universe Scan", "Watchlist Only"]
        )
    
    with col2:
        st.info(f"""
        **Full Scan:** Scan all stocks in database
        
        **Watchlist:** Update only tracked stocks
        """)
    
    if st.button("🚀 Start Scan", type="primary"):
        with st.spinner("Running scan... Please wait..."):
            try:
                if scan_type == "Full Universe Scan":
                    # Run full scan
                    from scanner import main as run_scanner
                    run_scanner()
                    st.success("✅ Full scan completed successfully!")
                else:
                    # Run watchlist scan
                    from watchlist_monitor import main as run_watchlist
                    run_watchlist()
                    st.success("✅ Watchlist scan completed successfully!")
                
                st.balloons()
                st.info("Refresh the page to see updated results")
                
            except Exception as e:
                st.error(f"❌ Scan failed: {str(e)}")

# Data Management Page
elif page == "📁 Data Management":
    st.header("Data Management")
    
    tab1, tab2, tab3 = st.tabs(["Load Stock Universe", "Export Data", "Cleanup"])
    
    with tab1:
        st.subheader("Load Stock Universe from CSV")
        
        uploaded_file = st.file_uploader("Upload CSV file with stock symbols", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:")
            st.dataframe(df.head())
            
            if st.button("Load Stocks into Database"):
                try:
                    count = db.load_stock_universe(df=df)
                    st.success(f"✅ Loaded {count} new stocks into database")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.subheader("Export Data")
        
        export_type = st.selectbox(
            "Select data to export",
            ["Qualified Stocks", "Watchlist", "All Scan Results", "Alerts"]
        )
        
        if st.button("Export to CSV"):
            try:
                if export_type == "Qualified Stocks":
                    data = db.get_latest_scan_results(qualified_only=True)
                    filename = "qualified_stocks"
                elif export_type == "Watchlist":
                    data = db.get_active_watchlist()
                    filename = "watchlist"
                elif export_type == "All Scan Results":
                    data = db.get_latest_scan_results()
                    filename = "scan_results"
                else:
                    data = db.get_unread_alerts()
                    filename = "alerts"
                
                if not data.empty:
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {export_type}",
                        data=csv,
                        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with tab3:
        st.subheader("Database Cleanup")
        
        days_to_keep = st.slider("Keep data for how many days?", 30, 365, 90)
        
        st.warning(f"⚠️ This will delete scan results and price data older than {days_to_keep} days")
        
        if st.button("🗑️ Cleanup Old Data", type="secondary"):
            try:
                deleted = db.cleanup_old_data(days_to_keep)
                st.success(f"✅ Deleted {deleted} old records")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Elite Swing Trading Scanner v5.0 | Built with Streamlit</p>
    <p>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
