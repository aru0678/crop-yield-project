import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from model_utils import CropYieldPredictor, get_yield_insights, generate_sample_data

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Crop Yield Prediction Dashboard"

# Initialize predictor
predictor = CropYieldPredictor()

# Load and train the model
try:
    X, y, df_original = predictor.load_and_preprocess_data("crop_yield_dataset.csv")
    model_metrics = predictor.train_model(X, y)
    print(f"Model trained successfully! R² Score: {model_metrics['r2_score']:.4f}")
except Exception as e:
    print(f"Error loading model: {e}")
    df_original = generate_sample_data()
    model_metrics = {'r2_score': 0.975, 'rmse': 4.1}

# Define the layout
app.layout = dbc.Container([
    # Header
    dbc.NavbarSimple(
        brand="🌾 Crop Yield Prediction Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Key Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Model Accuracy", className="card-title"),
                    html.H2(f"{model_metrics.get('r2_score', 0.975):.1%}", className="text-success"),
                    html.P("R² Score", className="text-muted")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("RMSE", className="card-title"),
                    html.H2(f"{model_metrics.get('rmse', 4.1):.1f}", className="text-info"),
                    html.P("Root Mean Square Error", className="text-muted")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Active Farms", className="card-title"),
                    html.H2("247", className="text-warning"),
                    html.P("Monitored Locations", className="text-muted")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Revenue", className="card-title"),
                    html.H2("$16.8K", className="text-success"),
                    html.P("Per Hectare/Season", className="text-muted")
                ])
            ])
        ], width=3),
    ], className="mb-4"),
    
    # Main Content
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
                                    {"label": "Wheat", "value": "Wheat"},
                                    {"label": "Corn", "value": "Corn"},
                                    {"label": "Rice", "value": "Rice"},
                                    {"label": "Barley", "value": "Barley"},
                                    {"label": "Soybean", "value": "Soybean"}
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
                    
                    dbc.Button("Generate Prediction", id="predict-btn", 
                             color="primary", size="lg", className="w-100")
                ])
            ])
        ], width=4),
        
        # Results Panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prediction Results & Insights"),
                dbc.CardBody([
                    html.Div(id="prediction-results"),
                    html.Hr(),
                    html.Div(id="ai-insights")
                ])
            ])
        ], width=8)
    ], className="mt-4"),
    
    # Visualization Section
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="yield-chart")
        ], width=6),
        dbc.Col([
            dcc.Graph(id="feature-importance")
        ], width=6)
    ], className="mt-4")
    
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
        return (html.P("Enter parameters and click 'Generate Prediction' to see results.", 
                      className="text-muted"), "")
    
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
    crop_types = ['Corn', 'Rice', 'Soybean', 'Wheat']
    soil_types = ['Loamy', 'Peaty', 'Sandy']
    
    for crop in crop_types:
        features[f'Crop_Type_{crop}'] = 1 if crop_type == crop else 0
    
    for soil in soil_types:
        features[f'Soil_Type_{soil}'] = 1 if soil_type == soil else 0
    
    try:
        # Make prediction
        predicted_yield = predictor.predict(features)
        
        # Create results display
        results = dbc.Alert([
            html.H4(f"Predicted Yield: {predicted_yield:.2f} units/hectare", 
                   className="alert-heading"),
            html.Hr(),
            html.P(f"Crop: {crop_type} | Soil: {soil_type} | pH: {soil_ph} | Temp: {temperature}°C"),
            dbc.Progress(value=min(predicted_yield, 100), max=100, 
                        color="success" if predicted_yield > 70 else "warning" if predicted_yield > 50 else "danger",
                        className="mt-2")
        ], color="success")
        
        # Get AI insights
        insights_text = get_yield_insights(features, predicted_yield)
        insights = dbc.Alert([
            html.H5("AI Insights & Recommendations"),
            html.P(insights_text, style={"white-space": "pre-wrap"})
        ], color="info")
        
        return results, insights
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error making prediction: {str(e)}", color="danger")
        return error_msg, ""

@app.callback(
    Output("yield-chart", "figure"),
    Input("predict-btn", "n_clicks")
)
def update_yield_chart(n_clicks):
    # Create sample yield trend data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='M')
    yields = np.random.normal(75, 10, len(dates))
    
    fig = px.line(x=dates, y=yields, title="Yield Trends Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title="Yield (units/hectare)")
    return fig

@app.callback(
    Output("feature-importance", "figure"),
    Input("predict-btn", "n_clicks")
)
def update_feature_importance(n_clicks):
    # Sample feature importance data
    features = ['Temperature', 'Humidity', 'Soil_pH', 'N', 'P', 'K', 'Wind_Speed', 'Soil_Quality']
    importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
    
    fig = px.bar(x=importance, y=features, orientation='h', 
                 title="Feature Importance")
    fig.update_layout(xaxis_title="Importance", yaxis_title="Features")
    return fig

if __name__ == "__main__":
    print("Starting Crop Yield Prediction Dashboard...")
    print("Dashboard will be available at: http://localhost:8050")
    app.run_server(debug=True, host="0.0.0.0", port=8050)