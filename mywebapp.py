import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import sqlite3
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hotel Revenue Dashboard", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    conn = sqlite3.connect('hotel_revenue.db')  # Connect to SQLite DB
    df = pd.read_sql_query("SELECT * FROM hotel_data", conn)  # Read hotel_data table
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    conn.close()
    return df

df = load_data()

# --- CUSTOM TITLE ---
st.markdown("<div style='text-align: center; font-size: 40px; font-weight: 900; color: #768b45;'>🏨 Hotel Revenue Management Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 18px; color: #eec76b;'>Created by Balram Shah | Internship Project at HCL Tech</div>", unsafe_allow_html=True)

# --- SIDEBAR FILTERS ---
st.sidebar.header("📊 Filter Options")

st.sidebar.markdown("""
### 🧾 About This Dashboard

The **Hotel Revenue Dashboard** is an interactive tool to visualize hotel booking trends and predict future revenue using machine learning. It helps hotel managers:

- 📊 Analyze room demand and customer behavior
- 🤖 Predict revenue using real booking inputs
- 📈 Optimize pricing and booking strategies

""")

room_types = df['room_type'].unique()
selected_rooms = st.sidebar.multiselect("Room Type", room_types, default=room_types)

months = sorted(df['booking_month'].unique())
selected_months = st.sidebar.multiselect("Booking Month", months, default=months)

lead_min, lead_max = int(df['booking_lead_time'].min()), int(df['booking_lead_time'].max())
lead_range = st.sidebar.slider("Lead Time (days)", lead_min, lead_max, (lead_min, lead_max))

min_date = df['checkin_date'].min().date()
max_date = df['checkin_date'].max().date()
selected_dates = st.sidebar.date_input("Check-in Date Range", [min_date, max_date])

status_filter = st.sidebar.radio("Booking Status", ["All", "Only Canceled", "Only Completed"])

filtered_df = df[
    (df['room_type'].isin(selected_rooms)) &
    (df['booking_month'].isin(selected_months)) &
    (df['booking_lead_time'] >= lead_range[0]) &
    (df['booking_lead_time'] <= lead_range[1]) &
    (df['checkin_date'] >= pd.to_datetime(selected_dates[0])) &
    (df['checkin_date'] <= pd.to_datetime(selected_dates[1]))
]

if status_filter == "Only Canceled":
    filtered_df = filtered_df[filtered_df['cancellation_flag'] == 1]
elif status_filter == "Only Completed":
    filtered_df = filtered_df[filtered_df['cancellation_flag'] == 0]

# --- KPIs ---
st.markdown("### 📌 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Bookings", len(filtered_df))
col2.metric("Total Revenue", f"${filtered_df['total_revenue'].sum():,.2f}")
col3.metric("Avg Daily Rate", f"${filtered_df['avg_daily_rate'].mean():.2f}")

# --- CHARTS ---
st.markdown("## 📊 Insights & ML Results")
theme = "plotly_white"

with st.expander("📈 Revenue Trend Over Time"):
    df_time = df.set_index("checkin_date")
    fig = px.line(df_time.resample('M').sum().reset_index(), x='checkin_date', y='total_revenue',
                  title="Monthly Revenue Trend", template=theme)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("💲 Hotel vs Competitor Pricing"):
    comp = df_time[['room_price', 'competitor_price']].resample('M').mean().reset_index()
    fig = px.line(comp, x='checkin_date', y=['room_price', 'competitor_price'],
                  title="Hotel vs Competitor Price", template=theme)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("👥 Revenue by Customer Segment"):
    fig, ax = plt.subplots(figsize=(20, 6))
    sns.boxplot(data=df, x='customer_segment', y='total_revenue', palette='Set2', ax=ax)
    ax.set_title("Revenue by Customer Segment")
    st.pyplot(fig)

with st.expander("📊 Room Price vs Occupancy"):
    fig = px.scatter(df, x='room_price', y='occupancy_rate', color='room_type', template=theme,
                     title="Price vs Occupancy Rate", width=800, height=400)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🕵️‍♂️ Avg Daily Rate by Month"):
    adr = df.groupby("booking_month")["avg_daily_rate"].mean().reset_index()
    fig = px.bar(adr, x="booking_month", y="avg_daily_rate", text_auto=True, template=theme,
                 color="avg_daily_rate", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🏯 Revenue by Room Type"):
    fig = px.pie(df, names='room_type', values='total_revenue', template=theme,
                 title='Revenue Contribution by Room Type')
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📈 Booking Lead Time Distribution"):
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.histplot(df['booking_lead_time'], kde=True, ax=ax, color='skyblue')
    ax.set_title("Lead Time Distribution")
    st.pyplot(fig)

with st.expander("❌ Cancellation Rate by Segment"):
    cancel_seg = df.groupby('customer_segment')['cancellation_flag'].mean().reset_index()
    fig = px.bar(cancel_seg, x='customer_segment', y='cancellation_flag',
                 title='Cancellation % by Segment', template=theme)
    st.plotly_chart(fig, use_container_width=True)

# --- ML Prediction ---
with st.expander("🤖 Predict Revenue (ML Model)", expanded=False):
    st.markdown("Enter details below to predict expected revenue.")
    with st.form("prediction_form"):
        cols = st.columns(4)
        room_type = cols[0].selectbox("Room Type", ['Deluxe', 'Double', 'Single', 'Suite'])
        customer_segment = cols[1].selectbox("Customer Segment", ['Business', 'Group', 'Leisure', 'Solo'])
        nights_stayed = cols[2].number_input("Nights Stayed", 1, 30, 2)
        booking_lead_time = cols[3].slider("Lead Time", 0, 365, 30)

        cols = st.columns(4)
        occupancy_rate = cols[0].slider("Occupancy Rate (%)", 0, 100, 75)
        room_price = cols[1].number_input("Room Price ($)", 50.0, 1000.0)
        discount_applied = cols[2].number_input("Discount Applied ($)", 0.0, 500.0)
        season = cols[3].selectbox("Season", ['Spring', 'Summer', 'Autumn', 'Winter'])

        cols = st.columns(4)
        day_of_week = cols[0].selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        event_type = cols[1].selectbox("Event Type", ['None', 'Conference', 'Festival', 'Exhibition'])
        competitor_price = cols[2].number_input("Competitor Price ($)", 50.0, 1000.0)
        cancellation_flag = cols[3].selectbox("Cancelled?", [0, 1])

        cols = st.columns(4)
        payment_method = cols[0].selectbox("Payment Method", ['Cash', 'Online', 'Credit Card'])
        customer_rating = cols[1].slider("Customer Rating", 1.0, 5.0, 4.5, 0.1)
        extra_services = cols[2].selectbox("Extra Services", ['None', 'Spa', 'Breakfast', 'Dinner', 'All'])
        holiday_season = cols[3].selectbox("Holiday Season", [0, 1])

        cols = st.columns(4)
        final_price = room_price * (occupancy_rate / 100)
        marketing_spend = cols[0].number_input("Marketing Spend ($)", 0.0, 10000.0, 200.0)
        customer_feedback = cols[1].selectbox("Customer Feedback", ['Negative', 'Neutral', 'Positive'])
        special_event = cols[2].selectbox("Special Event", [0, 1])
        booking_month = cols[3].selectbox("Booking Month", list(range(1, 13)), index=5)

        avg_daily_rate = st.number_input("Avg Daily Rate ($)", 50.0, 1000.0)

        submitted = st.form_submit_button("Predict Revenue")

    if submitted:
        model = joblib.load("xgb_hotel_model.pkl")

        room_map = {'Deluxe': 0, 'Double': 1, 'Single': 2, 'Suite': 3}
        segment_map = {'Business': 0, 'Group': 1, 'Leisure': 2, 'Solo': 3}
        payment_map = {'Cash': 0, 'Online': 1, 'Credit Card': 2}
        season_map = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        event_map = {'None': 0, 'Conference': 1, 'Festival': 2, 'Exhibition': 3}
        feedback_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        services_map = {'None': -1, 'Spa': 0, 'Breakfast': 1, 'Dinner': 2, 'All': 3}

        input_data = pd.DataFrame({
            'room_type': [room_map[room_type]],
            'customer_segment': [segment_map[customer_segment]],
            'nights_stayed': [nights_stayed],
            'booking_lead_time': [booking_lead_time],
            'occupancy_rate': [occupancy_rate],
            'room_price': [room_price],
            'discount_applied': [discount_applied],
            'season': [season_map[season]],
            'day_of_week': [day_map[day_of_week]],
            'event_type': [event_map[event_type]],
            'competitor_price': [competitor_price],
            'demand_index': [1.0],
            'cancellation_flag': [cancellation_flag],
            'payment_method': [payment_map[payment_method]],
            'customer_rating': [customer_rating],
            'extra_services': [services_map[extra_services]],
            'holiday_season': [holiday_season],
            'final_price': [final_price],
            'marketing_spend': [marketing_spend],
            'customer_feedback': [feedback_map[customer_feedback]],
            'special_event': [special_event],
            'booking_month': [booking_month],
            'avg_daily_rate': [avg_daily_rate]
        })

        st.write("📋 Model Input Data:", input_data)
        predicted_revenue = model.predict(input_data)[0]
        st.success(f"✅ Predicted Revenue: ${predicted_revenue:,.2f}")

# --- Data Preview ---
st.markdown("---")
st.markdown("📂 Filtered Data Preview")
st.dataframe(filtered_df)

# --- Download Button ---
st.sidebar.download_button("⬇️ Download Filtered Data", data=filtered_df.to_csv(index=False).encode('utf-8'), file_name="filtered_data.csv")

# --- Footer ---
st.markdown("---")
st.markdown("© 2025 | Created by Balram Shah | Powered by Streamlit")
