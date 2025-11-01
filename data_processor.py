import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataProcessor:
    def __init__(self):
        self.weather_cache = {}
        self.market_cache = {}
        
    def generate_regional_data(self, num_regions=10):
        """Generate synthetic regional data for heatmap visualization"""
        np.random.seed(42)
        
        regions = [f"Region_{i+1}" for i in range(num_regions)]
        crops = ['Wheat', 'Corn', 'Rice', 'Barley', 'Soybean']
        
        regional_data = []
        for region in regions:
            lat = np.random.uniform(25, 45)  # Latitude range
            lon = np.random.uniform(-120, -70)  # Longitude range
            
            for crop in crops:
                # Simulate yield based on location and crop type
                base_yield = np.random.uniform(40, 100)
                climate_factor = 1 + 0.2 * np.sin(lat * np.pi / 180)  # Latitude effect
                
                regional_data.append({
                    'Region': region,
                    'Crop_Type': crop,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Predicted_Yield': base_yield * climate_factor,
                    'Historical_Avg': base_yield * climate_factor * 0.9,
                    'Soil_Quality': np.random.uniform(30, 90),
                    'Temperature_Avg': np.random.uniform(15, 35),
                    'Rainfall_mm': np.random.uniform(300, 1200)
                })
        
        return pd.DataFrame(regional_data)
    
    def create_geographic_heatmap(self, regional_data, metric='Predicted_Yield'):
        """Create a geographic heatmap using Plotly"""
        fig = px.scatter_mapbox(
            regional_data,
            lat='Latitude',
            lon='Longitude',
            color=metric,
            size='Soil_Quality',
            hover_data=['Region', 'Crop_Type', metric],
            color_continuous_scale='Viridis',
            title=f'Geographic Distribution of {metric}',
            mapbox_style='open-street-map',
            zoom=3,
            center={'lat': 35, 'lon': -95}
        )
        
        fig.update_layout(
            height=600,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        return fig
    
    def analyze_weather_impact(self, base_features: Dict) -> Dict:
        """Simulate weather impact analysis"""
        weather_scenarios = {
            'Current': 1.0,
            'Drought (-20% rainfall)': 0.85,
            'Excessive Rain (+30% rainfall)': 0.92,
            'Heat Wave (+5°C)': 0.88,
            'Cold Snap (-5°C)': 0.90,
            'Optimal Conditions': 1.15
        }
        
        results = {}
        base_yield = 65.0  # Simulated base yield
        
        for scenario, factor in weather_scenarios.items():
            results[scenario] = {
                'yield': base_yield * factor,
                'change_percent': (factor - 1) * 100,
                'risk_level': self._assess_risk_level(factor)
            }
        
        return results
    
    def _assess_risk_level(self, factor):
        """Assess risk level based on yield factor"""
        if factor >= 1.1:
            return 'Low Risk'
        elif factor >= 0.95:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    def generate_market_forecast(self, crop_type: str, days_ahead: int = 30) -> Dict:
        """Generate synthetic market price forecasting"""
        np.random.seed(42)
        
        # Base prices per crop ($/unit)
        base_prices = {
            'Wheat': 250,
            'Corn': 180,
            'Rice': 320,
            'Barley': 200,
            'Soybean': 400
        }
        
        base_price = base_prices.get(crop_type, 250)
        dates = pd.date_range(start=datetime.now(), periods=days_ahead, freq='D')
        
        # Simulate price volatility
        prices = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            # Add trend and random walk
            trend = 0.001 * i  # Slight upward trend
            volatility = np.random.normal(0, 0.02)  # 2% daily volatility
            
            current_price *= (1 + trend + volatility)
            prices.append({
                'Date': date,
                'Price': current_price,
                'Volume': np.random.uniform(1000, 5000),
                'Trend': 'Bullish' if volatility > 0 else 'Bearish'
            })
        
        forecast_df = pd.DataFrame(prices)
        
        return {
            'forecast_data': forecast_df,
            'current_price': base_price,
            'predicted_price_30d': forecast_df.iloc[-1]['Price'],
            'price_change_percent': ((forecast_df.iloc[-1]['Price'] - base_price) / base_price) * 100,
            'volatility': forecast_df['Price'].std() / forecast_df['Price'].mean()
        }
    
    def calculate_resource_optimization(self, features: Dict, predicted_yield: float) -> Dict:
        """Calculate resource optimization recommendations"""
        
        # Current resource usage (simulated)
        current_resources = {
            'water_usage': features.get('Humidity', 70) * 2,  # L/m²
            'fertilizer_n': features.get('N', 60),  # kg/ha
            'fertilizer_p': features.get('P', 45),  # kg/ha
            'fertilizer_k': features.get('K', 40),  # kg/ha
            'energy_usage': 150,  # kWh/ha
        }
        
        # Optimization recommendations
        optimizations = {
            'water_usage': {
                'current': current_resources['water_usage'],
                'optimized': current_resources['water_usage'] * 0.85,
                'savings_percent': 15,
                'method': 'Drip irrigation system'
            },
            'fertilizer_n': {
                'current': current_resources['fertilizer_n'],
                'optimized': min(current_resources['fertilizer_n'] * 1.1, 80),
                'savings_percent': -10 if current_resources['fertilizer_n'] < 70 else 5,
                'method': 'Precision application based on soil testing'
            },
            'energy_usage': {
                'current': current_resources['energy_usage'],
                'optimized': current_resources['energy_usage'] * 0.92,
                'savings_percent': 8,
                'method': 'Solar-powered irrigation pumps'
            }
        }
        
        # Calculate cost savings
        cost_per_unit = {
            'water_usage': 0.001,  # $/L
            'fertilizer_n': 1.2,   # $/kg
            'energy_usage': 0.12   # $/kWh
        }
        
        total_savings = 0
        for resource, data in optimizations.items():
            if resource in cost_per_unit:
                savings = (data['current'] - data['optimized']) * cost_per_unit[resource]
                data['cost_savings'] = savings
                total_savings += savings
        
        return {
            'optimizations': optimizations,
            'total_cost_savings': total_savings,
            'roi_estimate': total_savings * 4,  # Assume 4x ROI over season
            'environmental_impact': {
                'water_saved': optimizations['water_usage']['current'] - optimizations['water_usage']['optimized'],
                'carbon_reduction': optimizations['energy_usage']['current'] * 0.5 * 0.08  # kg CO2
            }
        }
    
    def generate_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Generate correlation matrix visualization"""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        fig.update_layout(
            width=600,
            height=600
        )
        
        return fig
    
    def create_yield_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create yield distribution visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Yield Distribution', 'Yield by Soil pH', 
                          'Yield by Temperature', 'Yield vs Soil Quality'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Histogram of yield distribution
        fig.add_trace(
            go.Histogram(x=df['Crop_Yield'], name='Yield Distribution', nbinsx=30),
            row=1, col=1
        )
        
        # Yield vs Soil pH
        fig.add_trace(
            go.Scatter(x=df['Soil_pH'], y=df['Crop_Yield'], 
                      mode='markers', name='pH vs Yield', opacity=0.6),
            row=1, col=2
        )
        
        # Yield vs Temperature
        fig.add_trace(
            go.Scatter(x=df['Temperature'], y=df['Crop_Yield'], 
                      mode='markers', name='Temp vs Yield', opacity=0.6),
            row=2, col=1
        )
        
        # Yield vs Soil Quality
        fig.add_trace(
            go.Scatter(x=df['Soil_Quality'], y=df['Crop_Yield'], 
                      mode='markers', name='Quality vs Yield', opacity=0.6),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Yield Analysis Dashboard",
            showlegend=False
        )
        
        return fig
    
    def process_natural_language_query(self, query: str, df: pd.DataFrame) -> str:
        """Process natural language queries about the data"""
        query_lower = query.lower()
        
        # Simple keyword-based responses (in production, use proper NLP)
        if 'correlation' in query_lower or 'relationship' in query_lower:
            if 'temperature' in query_lower:
                corr = df['Temperature'].corr(df['Crop_Yield'])
                return f"Temperature has a correlation of {corr:.3f} with crop yield. {'Strong positive' if corr > 0.7 else 'Moderate positive' if corr > 0.3 else 'Weak'} relationship."
            
            elif 'ph' in query_lower or 'soil ph' in query_lower:
                corr = df['Soil_pH'].corr(df['Crop_Yield'])
                return f"Soil pH has a correlation of {corr:.3f} with crop yield. Optimal pH range appears to be 6.0-7.0 for most crops."
        
        elif 'best' in query_lower or 'highest' in query_lower:
            if 'crop' in query_lower:
                best_crop = df.groupby('Crop_Type')['Crop_Yield'].mean().idxmax()
                avg_yield = df.groupby('Crop_Type')['Crop_Yield'].mean().max()
                return f"Based on historical data, {best_crop} shows the highest average yield at {avg_yield:.2f} units per hectare."
        
        elif 'improve' in query_lower or 'increase' in query_lower:
            return """To improve crop yield, focus on:
            1. Maintaining optimal soil pH (6.0-7.0)
            2. Ensuring adequate nitrogen levels (60-80 ppm)
            3. Managing temperature stress
            4. Improving soil quality through organic matter
            5. Optimizing irrigation timing"""
        
        elif 'weather' in query_lower or 'climate' in query_lower:
            return """Weather significantly impacts crop yield:
            - Temperature: Optimal range varies by crop (20-30°C for most)
            - Humidity: 60-80% is generally favorable
            - Rainfall: Consistent moisture is key, avoid extremes
            - Wind: Moderate speeds help with pollination and disease prevention"""
        
        else:
            return f"I analyzed your query about '{query}'. Based on the data, here are key insights: The dataset shows yield varies significantly with environmental conditions. Temperature, soil pH, and nutrient levels are primary factors. Would you like specific analysis on any of these factors?"
    
    def generate_alert_conditions(self, features: Dict, predicted_yield: float) -> List[Dict]:
        """Generate alerts for significant conditions"""
        alerts = []
        
        # Yield deviation alerts
        expected_yield_range = (50, 90)  # Normal range
        if predicted_yield < expected_yield_range[0]:
            alerts.append({
                'type': 'warning',
                'title': 'Low Yield Alert',
                'message': f'Predicted yield ({predicted_yield:.1f}) is below normal range. Consider soil amendments.',
                'severity': 'high' if predicted_yield < 40 else 'medium'
            })
        elif predicted_yield > expected_yield_range[1]:
            alerts.append({
                'type': 'success',
                'title': 'Excellent Yield Potential',
                'message': f'Predicted yield ({predicted_yield:.1f}) is above average. Maintain current conditions.',
                'severity': 'low'
            })
        
        # Environmental alerts
        if features.get('Soil_pH', 7) < 5.5:
            alerts.append({
                'type': 'warning',
                'title': 'Acidic Soil Alert',
                'message': 'Soil pH is too low. Consider lime application.',
                'severity': 'medium'
            })
        
        if features.get('Temperature', 25) > 35:
            alerts.append({
                'type': 'danger',
                'title': 'Heat Stress Alert',
                'message': 'High temperature may stress crops. Increase irrigation.',
                'severity': 'high'
            })
        
        if features.get('N', 60) < 40:
            alerts.append({
                'type': 'info',
                'title': 'Low Nitrogen',
                'message': 'Nitrogen levels are low. Consider fertilizer application.',
                'severity': 'medium'
            })
        
        return alerts