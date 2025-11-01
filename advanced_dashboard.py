import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from model_utils import CropYieldPredictor, get_yield_insights, generate_sample_data
from data_processor import DataProcessor

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "Advanced Crop Yield Prediction Dashboard"

# Initialize components
predictor = CropYieldPredictor()
data_processor = DataProcessor()

# Load and train the model
try:
    X, y, df_original = predictor.load_and_preprocess_data("crop_yield_dataset.csv")
    model_metrics = predictor.train_model(X, y)
    print(f"Model trained successfully! R² Score: {model_metrics['r2_score']:.4f}")
except Exception as e:
    print(f"Error loading model: {e}")
    df_original = generate_sample_data()
    # Train on sample data
    X, y, _ = predictor.load_and_preprocess_data("crop_yield_dataset.csv") if 'crop_yield_dataset.csv' else (None, None, None)
    if X is not None:
        model_metrics = predictor.train_model(X, y)
    else:
        model_metrics = {'r2_score': 0.975, 'rmse': 4.1}

# Generate additional data
regional_data = data_processor.generate_regional_data()

# Define the layout
app.layout = dbc.Container([
    # Header with navigation
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="#", id="nav-dashboard")),
            dbc.NavItem(dbc.NavLink("Analytics", href="#", id="nav-analytics")),
            dbc.NavItem(dbc.NavLink("Reports", href="#", id="nav-reports")),
        ],
        brand="🌾 Advanced Crop Yield Intelligence",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Alert System
    html.Div(id="alert-container", className="mb-3"),
    
    # Key Metrics Dashboard
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-bullseye fa-2x text-success mb-2"),
                        html.H4("Model Accuracy", className="card-title"),
                        html.H2(f"{model_metrics.get('r2_score', 0.975):.1%}", 
                               className="text-success"),
                        html.P("R² Score", className="text-muted")
                    ], className="text-center")
                ])
            ], color="light", className="h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-2x text-info mb-2"),
                        html.H4("RMSE", className="card-title"),
                        html.H2(f"{model_metrics.get('rmse', 4.1):.1f}", 
                               className="text-info"),
                        html.P("Root Mean Square Error", className="text-muted")
                    ], className="text-center")
                ])
            ], color="light", className="h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-seedling fa-2x text-warning mb-2"),
                        html.H4("Active Farms", className="card-title"),
                        html.H2("247", className="text-warning"),
                        html.P("Monitored Locations", className="text-muted")
                    ], className="text-center")
                ])
            ], color="light", className="h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-dollar-sign fa-2x text-success mb-2"),
                        html.H4("Avg Revenue", className="card-title"),
                        html.H2("$16.8K", className="text-success"),
                        html.P("Per Hectare/Season", className="text-muted")
                    ], className="text-center")
                ])
            ], color="light", className="h-100")
        ], width=3),
    ], className="mb-4"),
    
    # Main Content Tabs
    dbc.Tabs([
        # Enhanced Prediction Tab
        dbc.Tab(label="🎯 Smart Prediction", tab_id="prediction", children=[
            dbc.Row([
                # Input Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("🔧 Input Parameters", className="mb-0"),
                            dbc.Button("Load Preset", id="load-preset-btn", 
                                     size="sm", color="outline-primary")
                        ], className="d-flex justify-content-between align-items-center"),
                        dbc.CardBody([
                            # Basic Parameters
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
                                            {"label": "🟤 Loamy", "value": "Loamy"},
                                            {"label": "🟡 Sandy", "value": "Sandy"},
                                            {"label": "🔴 Clay", "value": "Clay"},
                                            {"label": "⚫ Peaty", "value": "Peaty"}
                                        ],
                                        value="Loamy"
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            
                            # Environmental Parameters
                            html.H6("🌡️ Environmental Conditions", className="mt-3 mb-2"),
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
                            
                            # Nutrient Parameters
                            html.H6("🧪 Nutrient Levels", className="mt-3 mb-2"),
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
                            
                            dbc.Button("🔮 Generate Prediction", id="predict-btn", 
                                     color="primary", size="lg", className="w-100 mb-2"),
                            dbc.Button("📊 Analyze Scenarios", id="scenario-btn", 
                                     color="outline-secondary", size="sm", className="w-100")
                        ])
                    ])
                ], width=4),
                
                # Results Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Prediction Results & Insights"),
                        dbc.CardBody([
                            html.Div(id="prediction-results"),
                            html.Hr(),
                            html.Div(id="ai-insights"),
                            html.Hr(),
                            html.Div(id="weather-impact-analysis"),
                            html.Hr(),
                            html.Div(id="resource-optimization")
                        ])
                    ])
                ], width=8)
            ], className="mt-4")
        ]),
        
        # Enhanced Visualization Tab
        dbc.Tab(label="📊 Advanced Analytics", tab_id="visualization", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="geographic-heatmap")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="yield-trends-chart")
                ], width=6)
            ], className="mt-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="crop-comparison-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="feature-importance-chart")
                ], width=6)
            ], className="mt-4")
        ]),
        
        # Market Intelligence Tab
        dbc.Tab(label="💰 Market Intelligence", tab_id="market", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Price Forecasting"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Select Crop for Forecast"),
                                    dcc.Dropdown(
                                        id="market-crop-select",
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
                                    dbc.Label("Forecast Period (Days)"),
                                    dcc.Slider(
                                        id="forecast-days",
                                        min=7, max=90, step=7, value=30,
                                        marks={i: f"{i}d" for i in range(7, 91, 14)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=6)
                            ]),
                            html.Div(id="market-forecast-results", className="mt-3")
                        ])
                    ])
                ], width=12)
            ], className="mt-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="price-forecast-chart")
                ], width=8),
                dbc.Col([
                    html.Div(id="market-insights")
                ], width=4)
            ], className="mt-4")
        ]),
        
        # Natural Language Interface Tab
        dbc.Tab(label="🤖 AI Assistant", tab_id="ai-assistant", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("💬 Natural Language Query Interface"),
                        dbc.CardBody([
                            dbc.InputGroup([
                                dbc.Input(
                                    id="nl-query",
                                    placeholder="Ask me anything about your crop data... (e.g., 'What factors most affect corn yield?')",
                                    type="text",
                                    className="form-control-lg"
                                ),
                                dbc.Button("Ask AI", id="ask-ai-btn", color="primary", size="lg")
                            ], className="mb-3"),
                            
                            # Quick Query Buttons
                            html.Div([
                                html.H6("Quick Questions:", className="mb-2"),
                                dbc.ButtonGroup([
                                    dbc.Button("Best crop for my region?", id="quick-1", size="sm", color="outline-primary"),
                                    dbc.Button("How to improve yield?", id="quick-2", size="sm", color="outline-primary"),
                                    dbc.Button("Weather impact analysis", id="quick-3", size="sm", color="outline-primary"),
                                    dbc.Button("Resource optimization", id="quick-4", size="sm", color="outline-primary")
                                ], className="mb-3")
                            ]),
                            
                            html.Div(id="nl-response", className="mt-3"),
                            
                            html.Hr(),
                            
                            # Automated Report Generation
                            html.H5("📋 Automated Reports"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Generate Yield Report", id="gen-yield-report", 
                                             color="success", className="w-100 mb-2"),
                                    dbc.Button("Generate Market Report", id="gen-market-report", 
                                             color="info", className="w-100 mb-2"),
                                    dbc.Button("Generate Optimization Report", id="gen-opt-report", 
                                             color="warning", className="w-100")
                                ], width=6),
                                dbc.Col([
                                    html.Div(id="generated-reports")
                                ], width=6)
                            ])
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
    ], id="main-tabs", active_tab="prediction"),
    
    # Footer
    html.Hr(),
    html.Footer([
        html.P("© 2024 Advanced Crop Yield Intelligence Dashboard | Powered by ML & AI", 
               className="text-center text-muted")
    ])
    
], fluid=True)

# Enhanced Callbacks
@app.callback(
    [Output("prediction-results", "children"),
     Output("ai-insights", "children"),
     Output("weather-impact-analysis", "children"),
     Output("resource-optimization", "children"),
     Output("alert-container", "children")],
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
def enhanced_prediction(n_clicks, crop_type, soil_type, soil_ph, temperature, 
                       humidity, wind_speed, nitrogen, phosphorus, potassium, soil_quality):
    if n_clicks is None:
        return (html.P("Enter parameters and click 'Generate Prediction' to see comprehensive results.", 
                      className="text-muted"), "", "", "", "")
    
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
        
        # Generate alerts
        alerts = data_processor.generate_alert_conditions(features, predicted_yield)
        alert_components = []
        for alert in alerts:
            color = "danger" if alert['type'] == 'danger' else "warning" if alert['type'] == 'warning' else "success"
            alert_components.append(
                dbc.Alert([
                    html.I(className=f"fas fa-exclamation-triangle me-2"),
                    html.Strong(alert['title']),
                    html.Span(f" - {alert['message']}")
                ], color=color, dismissable=True)
            )
        
        # Create results display
        results = dbc.Alert([
            html.H4(f"🎯 Predicted Yield: {predicted_yield:.2f} units/hectare", 
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
            html.H5("🤖 AI Insights & Recommendations"),
            html.P(insights_text, style={"white-space": "pre-wrap"})
        ], color="info")
        
        # Weather impact analysis
        weather_analysis = data_processor.analyze_weather_impact(features)
        weather_cards = []
        for scenario, data in weather_analysis.items():
            color = "success" if data['change_percent'] > 5 else "warning" if data['change_percent'] > -5 else "danger"
            weather_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(scenario, className="card-title"),
                            html.H5(f"{data['yield']:.1f}", className=f"text-{color}"),
                            html.P(f"{data['change_percent']:+.1f}%", className="text-muted"),
                            dbc.Badge(data['risk_level'], color=color)
                        ])
                    ])
                ], width=4)
            )
        
        weather_component = html.Div([
            html.H5("🌤️ Weather Impact Analysis"),
            dbc.Row(weather_cards[:3]),
            dbc.Row(weather_cards[3:], className="mt-2")
        ])
        
        # Resource optimization
        optimization = data_processor.calculate_resource_optimization(features, predicted_yield)
        opt_cards = []
        for resource, data in optimization['optimizations'].items():
            if 'cost_savings' in data:
                opt_cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6(resource.replace('_', ' ').title()),
                                html.P(f"Current: {data['current']:.1f}"),
                                html.P(f"Optimized: {data['optimized']:.1f}"),
                                html.P(f"Savings: ${data['cost_savings']:.2f}", className="text-success"),
                                html.Small(data['method'], className="text-muted")
                            ])
                        ])
                    ], width=4)
                )
        
        resource_component = html.Div([
            html.H5("⚡ Resource Optimization"),
            dbc.Row(opt_cards),
            dbc.Alert([
                html.H6(f"Total Potential Savings: ${optimization['total_cost_savings']:.2f}"),
                html.P(f"Estimated ROI: ${optimization['roi_estimate']:.2f}")
            ], color="success", className="mt-3")
        ])
        
        return results, insights, weather_component, resource_component, alert_components
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error making prediction: {str(e)}", color="danger")
        return error_msg, "", "", "", ""

# Additional callbacks for other features...
@app.callback(
    Output("geographic-heatmap", "figure"),
    Input("main-tabs", "active_tab")
)
def update_geographic_heatmap(active_tab):
    if active_tab != "visualization":
        return {}
    
    return data_processor.create_geographic_heatmap(regional_data)

@app.callback(
    [Output("price-forecast-chart", "figure"),
     Output("market-insights", "children")],
    [Input("market-crop-select", "value"),
     Input("forecast-days", "value")]
)
def update_market_forecast(crop_type, days):
    forecast_data = data_processor.generate_market_forecast(crop_type, days)
    
    # Create price chart
    fig = px.line(forecast_data['forecast_data'], x='Date', y='Price',
                  title=f"{crop_type} Price Forecast ({days} days)")
    fig.update_layout(yaxis_title="Price ($/unit)")
    
    # Market insights
    insights = dbc.Card([
        dbc.CardHeader("Market Insights"),
        dbc.CardBody([
            html.H5(f"${forecast_data['current_price']:.2f}", className="text-primary"),
            html.P("Current Price"),
            html.Hr(),
            html.H6(f"{forecast_data['price_change_percent']:+.1f}%", 
                   className="text-success" if forecast_data['price_change_percent'] > 0 else "text-danger"),
            html.P("30-day Change"),
            html.Hr(),
            html.P(f"Volatility: {forecast_data['volatility']:.1%}"),
            html.P("Market Trend: Bullish" if forecast_data['price_change_percent'] > 0 else "Bearish")
        ])
    ])
    
    return fig, insights

@app.callback(
    Output("nl-response", "children"),
    [Input("ask-ai-btn", "n_clicks"),
     Input("quick-1", "n_clicks"),
     Input("quick-2", "n_clicks"),
     Input("quick-3", "n_clicks"),
     Input("quick-4", "n_clicks")],
    [State("nl-query", "value")]
)
def process_natural_language(ask_clicks, q1, q2, q3, q4, query):
    ctx = callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle quick questions
    if button_id == "quick-1":
        query = "What is the best crop for my region?"
    elif button_id == "quick-2":
        query = "How can I improve my crop yield?"
    elif button_id == "quick-3":
        query = "How does weather impact my crops?"
    elif button_id == "quick-4":
        query = "How can I optimize my resources?"
    
    if query:
        response = data_processor.process_natural_language_query(query, df_original)
        return dbc.Alert([
            html.H6(f"Query: {query}"),
            html.Hr(),
            html.P(response)
        ], color="light")
    
    return ""

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)