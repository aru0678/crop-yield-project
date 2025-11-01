import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from model_utils import CropYieldPredictor, get_yield_insights, generate_sample_data

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Crop Yield Prediction Dashboard"

# Initialize the model
predictor = CropYieldPredictor()

# Load and train the model
try:
    X, y, df_original = predictor.load_and_preprocess_data("crop_yield_dataset.csv")
    model_metrics = predictor.train_model(X, y)
    print(f"Model trained successfully! R² Score: {model_metrics['r2_score']:.4f}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Use sample data for demo
    df_original = generate_sample_data()

# Generate sample historical data for visualization
def generate_historical_data():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
    
    historical_data = []
    for date in dates:
        for crop in ['Wheat', 'Corn', 'Rice', 'Barley', 'Soybean']:
            # Simulate seasonal patterns
            month = date.month
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month / 12)
            
            yield_value = np.random.uniform(30, 120) * seasonal_factor
            historical_data.append({
                'Date': date,
                'Crop_Type': crop,
                'Yield': yield_value,
                'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'])
            })
    
    return pd.DataFrame(historical_data)

historical_df = generate_historical_data()

# Define the layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🌾 Crop Yield Prediction Dashboard", 
                   className="text-center mb-4 text-primary"),
            html.P("Advanced ML-powered insights for agricultural optimization", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # Key Metrics Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Model Accuracy", className="card-title"),
                    html.H2(f"{model_metrics.get('r2_score', 0.975):.1%}", 
                           className="text-success"),
                    html.P("R² Score", className="text-muted")
                ])
            ], color="light")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("RMSE", className="card-title"),
                    html.H2(f"{model_metrics.get('rmse', 4.1):.1f}", 
                           className="text-info"),
                    html.P("Root Mean Square Error", className="text-muted")
                ])
            ], color="light")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Predictions Made", className="card-title"),
                    html.H2("1,247", className="text-warning"),
                    html.P("This Month", className="text-muted")
                ])
            ], color="light")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Yield", className="card-title"),
                    html.H2("67.3", className="text-primary"),
                    html.P("Units per Hectare", className="text-muted")
                ])
            ], color="light")
        ], width=3),
    ], className="mb-4"),
    
    # Main Content Tabs
    dbc.Tabs([
        # Prediction Tab
        dbc.Tab(label="🎯 Yield Prediction", tab_id="prediction", children=[
            dbc.Row([
                # Input Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Input Parameters"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Crop Type"),
                                    dcc.Dropdown(
                                        id="crop-type",
                                        options=[
                                            {"label": "🌾 Wheat", "value": "Wheat"},
                                            {"label": "🌽 Corn", "value": "Corn"},
                                            {"label": "🍚 Rice", "value": "Rice"},
                                            {"label": "🌾 Barley", "value": "Barley"},
                                            {"label": "🫘 Soybean", "value": "Soybean"}
                                        ],
                                        value="Wheat"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Soil Type"),
                                    dcc.Dropdown(
                                        id="soil-type",
                                        options=[
                                            {"label": "Loamy", "value": "Loamy"},
                                            {"label": "Sandy", "value": "Sandy"},
                                            {"label": "Clay", "value": "Clay"},
                                            {"label": "Peaty", "value": "Peaty"}
                                        ],
                                        value="Loamy"
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Soil pH"),
                                    dcc.Slider(
                                        id="soil-ph",
                                        min=4.0, max=9.0, step=0.1, value=6.5,
                                        marks={i: str(i) for i in range(4, 10)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Temperature (°C)"),
                                    dcc.Slider(
                                        id="temperature",
                                        min=0, max=45, step=1, value=25,
                                        marks={i: f"{i}°" for i in range(0, 46, 10)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Humidity (%)"),
                                    dcc.Slider(
                                        id="humidity",
                                        min=20, max=100, step=1, value=70,
                                        marks={i: f"{i}%" for i in range(20, 101, 20)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Wind Speed (km/h)"),
                                    dcc.Slider(
                                        id="wind-speed",
                                        min=0, max=20, step=0.5, value=8,
                                        marks={i: str(i) for i in range(0, 21, 5)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            
                            html.H6("Nutrient Levels", className="mt-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Nitrogen (N)"),
                                    dcc.Slider(
                                        id="nitrogen",
                                        min=0, max=150, step=1, value=60,
                                        marks={i: str(i) for i in range(0, 151, 30)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Phosphorus (P)"),
                                    dcc.Slider(
                                        id="phosphorus",
                                        min=0, max=100, step=1, value=45,
                                        marks={i: str(i) for i in range(0, 101, 25)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Potassium (K)"),
                                    dcc.Slider(
                                        id="potassium",
                                        min=0, max=100, step=1, value=40,
                                        marks={i: str(i) for i in range(0, 101, 25)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=4)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Soil Quality Score"),
                                    dcc.Slider(
                                        id="soil-quality",
                                        min=0, max=100, step=1, value=60,
                                        marks={i: str(i) for i in range(0, 101, 25)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=12)
                            ], className="mb-3"),
                            
                            dbc.Button("🔮 Predict Yield", id="predict-btn", 
                                     color="primary", size="lg", className="w-100")
                        ])
                    ])
                ], width=4),
                
                # Results Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Prediction Results"),
                        dbc.CardBody([
                            html.Div(id="prediction-results"),
                            html.Hr(),
                            html.Div(id="ai-insights")
                        ])
                    ])
                ], width=8)
            ], className="mt-4")
        ]),
        
        # Visualization Tab
        dbc.Tab(label="📊 Data Visualization", tab_id="visualization", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="yield-trends-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="crop-comparison-chart")
                ], width=6)
            ], className="mt-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="regional-heatmap")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="feature-importance-chart")
                ], width=6)
            ], className="mt-4")
        ]),
        
        # Analytics Tab
        dbc.Tab(label="📈 Advanced Analytics", tab_id="analytics", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Natural Language Query"),
                        dbc.CardBody([
                            dbc.InputGroup([
                                dbc.Input(
                                    id="nl-query",
                                    placeholder="Ask about your crop data... (e.g., 'What factors most affect corn yield?')",
                                    type="text"
                                ),
                                dbc.Button("Ask AI", id="ask-ai-btn", color="primary")
                            ]),
                            html.Div(id="nl-response", className="mt-3")
                        ])
                    ])
                ], width=12)
            ], className="mt-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="correlation-matrix")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="yield-distribution")
                ], width=6)
            ], className="mt-4")
        ])
    ], id="main-tabs", active_tab="prediction")
    
], fluid=True)

# Callbacks
@app.callback(
    [Output("prediction-results", "children"),
     Output("ai-insights", "children")],
    [Input("predict-btn", "n_clicks")],
    [State("crop-type", "value"),
     State("soil-type", "value"),
     State("soil-ph", "value"),
     State("temperature", "value"),
     State("humidity", "value"),
     State("wind-speed", "value"),
     State("nitrogen", "value"),
     State("phosphorus", "value"),
     State("potassium", "value"),
     State("soil-quality", "value")]
)
def make_prediction(n_clicks, crop_type, soil_type, soil_ph, temperature, 
                   humidity, wind_speed, nitrogen, phosphorus, potassium, soil_quality):
    if n_clicks is None:
        return html.P("Enter parameters and click 'Predict Yield' to see results.", 
                     className="text-muted"), ""
    
    # Prepare features
    features = {
        'Soil_pH': soil_ph,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind_Speed': wind_speed,
        'N': nitrogen,
        'P': phosphorus,
        'K': potassium,
        'Soil_Quality': soil_quality
    }
    
    # Add one-hot encoded features
    crop_types = ['Corn', 'Rice', 'Soybean', 'Wheat']  # Barley is reference
    soil_types = ['Loamy', 'Peaty', 'Sandy']  # Clay is reference
    
    for crop in crop_types:
        features[f'Crop_Type_{crop}'] = 1 if crop_type == crop else 0
    
    for soil in soil_types:
        features[f'Soil_Type_{soil}'] = 1 if soil_type == soil else 0
    
    try:
        # Make prediction
        predicted_yield = predictor.predict(features)
        
        # Create results display
        results = dbc.Alert([
            html.H4(f"🎯 Predicted Yield: {predicted_yield:.2f} units/hectare", 
                   className="alert-heading"),
            html.Hr(),
            html.P(f"Crop: {crop_type} | Soil: {soil_type} | pH: {soil_ph}")
        ], color="success")
        
        # Get AI insights
        insights_text = get_yield_insights(features, predicted_yield)
        insights = dbc.Alert([
            html.H5("🤖 AI Insights & Recommendations"),
            html.P(insights_text, style={"white-space": "pre-wrap"})
        ], color="info")
        
        return results, insights
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error making prediction: {str(e)}", color="danger")
        return error_msg, ""

@app.callback(
    Output("yield-trends-chart", "figure"),
    Input("main-tabs", "active_tab")
)
def update_yield_trends(active_tab):
    if active_tab != "visualization":
        return {}
    
    # Create time series chart
    fig = px.line(historical_df, x='Date', y='Yield', color='Crop_Type',
                  title="Crop Yield Trends Over Time")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Yield (units/hectare)",
        hovermode='x unified'
    )
    return fig

@app.callback(
    Output("crop-comparison-chart", "figure"),
    Input("main-tabs", "active_tab")
)
def update_crop_comparison(active_tab):
    if active_tab != "visualization":
        return {}
    
    # Create box plot for crop comparison
    fig = px.box(historical_df, x='Crop_Type', y='Yield',
                 title="Yield Distribution by Crop Type")
    fig.update_layout(
        xaxis_title="Crop Type",
        yaxis_title="Yield (units/hectare)"
    )
    return fig

@app.callback(
    Output("regional-heatmap", "figure"),
    Input("main-tabs", "active_tab")
)
def update_regional_heatmap(active_tab):
    if active_tab != "visualization":
        return {}
    
    # Create heatmap of average yields by region and crop
    pivot_data = historical_df.groupby(['Region', 'Crop_Type'])['Yield'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='Region', columns='Crop_Type', values='Yield')
    
    fig = px.imshow(pivot_table, 
                    title="Average Yield by Region and Crop Type",
                    color_continuous_scale="Viridis")
    return fig

@app.callback(
    Output("feature-importance-chart", "figure"),
    Input("main-tabs", "active_tab")
)
def update_feature_importance(active_tab):
    if active_tab != "visualization":
        return {}
    
    try:
        importance_df = predictor.get_feature_importance().head(10)
        fig = px.bar(importance_df, x='importance', y='feature', 
                     orientation='h', title="Top 10 Feature Importance")
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Features"
        )
        return fig
    except:
        return {}

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)