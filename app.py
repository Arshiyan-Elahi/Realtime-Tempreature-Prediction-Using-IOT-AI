import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import pickle
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import calendar

# Set page configuration
st.set_page_config(page_title="Forecasting Dashboard", layout="wide", initial_sidebar_state="expanded")

# Enhanced CSS styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .main {
            background-color: #f8f9fa;
            font-family: 'Inter', sans-serif;
        }
        .metric-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #1f77b4;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }
        .chart-wrapper {
            background: white;
            border-radius: 0.8rem;
            padding: 1.2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        .stat-container {
            background: white;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            text-align: center;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .section-title {
            color: #2c3e50;
            font-size: 1.3rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            padding-left: 0.5rem;
            border-left: 4px solid #3498db;
        }
        .title {
            font-size: 2.2rem;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin: 2rem 0;
        }
        .subtitle {
            font-size: 1.5rem;
            color: #34495e;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Dashboard Settings")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 14, 7)
confidence_interval = st.sidebar.slider("Confidence Interval", 0.5, 0.95, 0.8)
view_type = st.sidebar.selectbox("Time Aggregation", ["Hourly", "Daily", "Weekly"])
chart_theme = st.sidebar.selectbox("Chart Theme", ["plotly", "plotly_dark", "plotly_white"])

# File uploaders
st.sidebar.header("Data Upload")
historical_file = st.sidebar.file_uploader("Historical Data (CSV)", type=["csv"])
realtime_file = st.sidebar.file_uploader("Real-Time Data (CSV)", type=["csv"])

# Helper functions
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, r2, mape

def create_metric_card(title, value, format=".2f"):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value:{format}}</div>
            <div class="metric-label">{title}</div>
        </div>
    """, unsafe_allow_html=True)

def detect_datetime_column(df):
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            return col
        except:
            continue
    return None

def create_heatmap(data):
    data['Hour'] = data['ds'].dt.hour
    data['DayOfWeek'] = data['ds'].dt.day_name()
    
    pivot_table = data.pivot_table(
        values='y', 
        index='Hour',
        columns='DayOfWeek',
        aggfunc='mean'
    )
    
    days_order = list(calendar.day_name)
    pivot_table = pivot_table[days_order]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Average Values by Hour and Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Hour of Day',
        height=400
    )
    
    return fig

def create_monthly_pattern(data):
    data['Month'] = data['ds'].dt.month
    data['Year'] = data['ds'].dt.year
    
    monthly_avg = data.groupby(['Year', 'Month'])['y'].mean().reset_index()
    monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(DAY=1))
    
    fig = go.Figure()
    
    for year in monthly_avg['Year'].unique():
        year_data = monthly_avg[monthly_avg['Year'] == year]
        fig.add_trace(go.Scatter(
            x=year_data['Month'],
            y=year_data['y'],
            name=str(year),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Monthly Patterns by Year',
        xaxis_title='Month',
        yaxis_title='Average Value',
        height=400,
        xaxis=dict(tickmode='array', ticktext=list(calendar.month_abbr)[1:], tickvals=list(range(1,13)))
    )
    
    return fig

def create_box_plots(data):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Hourly Distribution', 'Daily Distribution', 'Monthly Distribution')
    )
    
    fig.add_trace(
        go.Box(y=data['y'], x=data['ds'].dt.hour, name='Hourly'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(y=data['y'], x=data['ds'].dt.day_name(), name='Daily'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(y=data['y'], x=data['ds'].dt.month.map(lambda x: calendar.month_abbr[x]), name='Monthly'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_forecast_plot(forecast, actual_data, view_type):
    if view_type == "Weekly":
        forecast = forecast.set_index('ds').resample('W').mean().reset_index()
        actual_data = actual_data.set_index('ds').resample('W').mean().reset_index()
    elif view_type == "Daily":
        forecast = forecast.set_index('ds').resample('D').mean().reset_index()
        actual_data = actual_data.set_index('ds').resample('D').mean().reset_index()

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_data['ds'],
        y=actual_data['y'],
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Predicted',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,127,14,0.2)',
        line=dict(color='rgba(255,127,14,0)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title=f"{view_type} Forecast with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Tempreature  Value",
        template=chart_theme,
        showlegend=True,
        height=500
    )
    
    return fig

# Main app logic
if historical_file and realtime_file:
    try:
        # Load and process data
        historical_df = pd.read_csv(historical_file)
        realtime_df = pd.read_csv(realtime_file)
        
        # Add data info to sidebar
        st.sidebar.markdown("### Data Information")
        st.sidebar.write(f"Historical rows: {len(historical_df)}")
        st.sidebar.write(f"Realtime rows: {len(realtime_df)}")
        
        # Process datetime columns
        date_col_hist = detect_datetime_column(historical_df)
        date_col_real = detect_datetime_column(realtime_df)
        
        if not date_col_hist or not date_col_real:
            st.error("Could not detect date columns in one or both files.")
            st.stop()
        
        # Preprocess data
        historical_df[date_col_hist] = pd.to_datetime(historical_df[date_col_hist])
        realtime_df[date_col_real] = pd.to_datetime(realtime_df[date_col_real])
        
        # Rename columns
        historical_df = historical_df.rename(columns={date_col_hist: 'ds', 'Actual': 'y'})
        realtime_df = realtime_df.rename(columns={date_col_real: 'ds', 'temperature': 'y'})
        
        # Keep required columns
        historical_df = historical_df[['ds', 'y']]
        realtime_df = realtime_df[['ds', 'y']]
        
        # Handle duplicates and sort
        historical_df = historical_df.drop_duplicates(subset='ds')
        realtime_df = realtime_df.drop_duplicates(subset='ds')
        
        historical_df = historical_df.sort_values('ds')
        realtime_df = realtime_df.sort_values('ds')
        
        # Combine data
        combined_df = pd.concat([historical_df, realtime_df])
        combined_df = combined_df.drop_duplicates(subset='ds', keep='last')
        combined_df = combined_df.sort_values('ds').reset_index(drop=True)
        
        # Show processed data info
        st.sidebar.markdown("### Processed Data")
        st.sidebar.write(f"Total samples: {len(combined_df)}")
        st.sidebar.write(f"Date range: {combined_df['ds'].min().date()} to {combined_df['ds'].max().date()}")
        
        # Train Prophet model
        with st.spinner('Training model... Please wait.'):
            model = Prophet(
                interval_width=confidence_interval,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            if (combined_df['ds'].max() - combined_df['ds'].min()).days > 1:
                model.add_seasonality(
                    name='hourly',
                    period=24,
                    fourier_order=5
                )
            
            model.fit(combined_df)
        
        # Make forecast
        future = model.make_future_dataframe(periods=forecast_days * 24, freq='h')
        forecast = model.predict(future)
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📈 Main Forecast", "📊 Patterns & Distributions", "🔍 Detailed Analysis"])
        
        with tab1:
            st.markdown("<h1 class='title'>Forecasting Dashboard</h1>", unsafe_allow_html=True)
            
            # Metrics grid
            st.markdown("<div class='section-title'>Model Performance Metrics</div>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            mae, rmse, r2, mape = calculate_metrics(
                combined_df['y'],
                forecast[forecast['ds'].isin(combined_df['ds'])]['yhat']
            )
            
            with col1:
                create_metric_card("Mean Absolute Error", mae)
            with col2:
                create_metric_card("Root Mean Square Error", rmse)
            with col3:
                create_metric_card("R² Score", r2)
            with col4:
                create_metric_card("MAPE (%)", mape)
            
            # Main forecast plot
            st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
            st.markdown("<h2 class='subtitle'>Forecast Overview</h2>", unsafe_allow_html=True)
            forecast_fig = create_forecast_plot(forecast, combined_df, view_type)
            st.plotly_chart(forecast_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
                heatmap = create_heatmap(combined_df)
                st.plotly_chart(heatmap, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
                monthly_pattern = create_monthly_pattern(combined_df)
                st.plotly_chart(monthly_pattern, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Box plots (full width)
            st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
            box_plots = create_box_plots(combined_df)
            st.plotly_chart(box_plots, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
            
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
                st.markdown("<h3 class='subtitle'>Trend Components</h3>", unsafe_allow_html=True)
                fig_components = model.plot_components(forecast)
                st.pyplot(fig_components)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
                st.markdown("<h3 class='subtitle'>Actual vs Predicted</h3>", unsafe_allow_html=True)
                scatter_df = pd.DataFrame({
                    'Actual': combined_df['y'],
                    'Predicted': forecast[forecast['ds'].isin(combined_df['ds'])]['yhat']
                })
                fig_scatter = px.scatter(scatter_df, x='Actual', y='Predicted',
                                    trendline="ols",
                                    template=chart_theme)
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Error distribution
            st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
            st.markdown("<h3 class='subtitle'>Prediction Error Distribution</h3>", unsafe_allow_html=True)
            error_df = pd.DataFrame({
                'Error': combined_df['y'].values - 
                        forecast[forecast['ds'].isin(combined_df['ds'])]['yhat'].values
            })
            fig_dist = px.histogram(error_df, x='Error', nbins=50,
                                title='Prediction Error Distribution',
                                template=chart_theme)
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Add download options in sidebar
        st.sidebar.markdown("### Export Options")
        
        # Download processed data
        csv = combined_df.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download Processed Data",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
        
        # Download forecast results
        forecast_csv = forecast.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download Forecast Results",
            data=forecast_csv,
            file_name="forecast_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Detailed error information for debugging:")
        st.write(e)
    else:
        st.info("Please upload both historical and real-time data files to begin the analysis.")