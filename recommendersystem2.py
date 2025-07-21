import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# Create the Dash app
app = dash.Dash(__name__)

# CRITICAL: This line exposes the server to Render
server = app.server

# Simple sample data (since CSV files aren't on Render)
sample_neighborhoods = {
    'Mission': {'safety_score': 65, 'walkability_score': 85, 'avg_rent': 3500, 'crime_count': 45},
    'Marina': {'safety_score': 88, 'walkability_score': 92, 'avg_rent': 4200, 'crime_count': 12},
    'SoMa': {'safety_score': 58, 'walkability_score': 78, 'avg_rent': 4000, 'crime_count': 67},
    'Castro': {'safety_score': 82, 'walkability_score': 95, 'avg_rent': 3800, 'crime_count': 23},
    'Sunset': {'safety_score': 75, 'walkability_score': 65, 'avg_rent': 2800, 'crime_count': 18}
}

sample_dogs = [
    {'name': 'Max', 'breed': 'German Shepherd', 'size': 'Large', 'protection_score': 85, 'monthly_cost': 125},
    {'name': 'Bella', 'breed': 'Golden Retriever', 'size': 'Large', 'protection_score': 60, 'monthly_cost': 115},
    {'name': 'Charlie', 'breed': 'Pit Bull Mix', 'size': 'Medium', 'protection_score': 78, 'monthly_cost': 95},
    {'name': 'Luna', 'breed': 'Border Collie', 'size': 'Medium', 'protection_score': 55, 'monthly_cost': 90},
    {'name': 'Rocky', 'breed': 'Rottweiler', 'size': 'Large', 'protection_score': 92, 'monthly_cost': 135}
]

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🐕 SF Dog Crisis Dashboard", 
                style={'text-align': 'center', 'color': '#2c3e50', 'margin-bottom': '10px'}),
        html.P("Prototype version - Real data dashboard coming soon!", 
               style={'text-align': 'center', 'color': '#7f8c8d', 'margin-bottom': '30px'})
    ], style={'background': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)', 
              'padding': '30px', 'border-radius': '10px', 'margin-bottom': '20px'}),

    # User Input
    html.Div([
        html.H3("Select Your Neighborhood:", style={'color': '#2c3e50', 'margin-bottom': '20px'}),
        
        dcc.Dropdown(
            id='neighborhood-dropdown',
            options=[{'label': f"📍 {neighborhood}", 'value': neighborhood} 
                    for neighborhood in sample_neighborhoods.keys()],
            placeholder="🏠 Select your SF neighborhood...",
            style={'margin-bottom': '20px', 'font-size': '16px'}
        ),
        
        html.Button("Find My Perfect Guardian", id='find-match-button', n_clicks=0,
                   style={'background': '#e74c3c', 'color': 'white', 'border': 'none', 
                         'padding': '15px 30px', 'border-radius': '25px', 'font-size': '16px',
                         'cursor': 'pointer', 'width': '100%'})
    ], style={'background': '#ffffff', 'padding': '25px', 'border-radius': '10px', 
              'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'margin-bottom': '20px'}),

    # Results
    html.Div(id='results-container')
    
], style={'max-width': '1200px', 'margin': '0 auto', 'padding': '20px', 
          'font-family': 'Arial, sans-serif', 'background-color': '#f8f9fa'})

# Callback
@app.callback(
    Output('results-container', 'children'),
    [Input('find-match-button', 'n_clicks')],
    [dash.dependencies.State('neighborhood-dropdown', 'value')]
)
def update_results(n_clicks, neighborhood):
    if n_clicks == 0 or not neighborhood:
        return html.Div([
            html.H4("👆 Select your neighborhood above to see recommendations",
                   style={'text-align': 'center', 'color': '#7f8c8d', 'margin': '50px'})
        ])
    
    # Get neighborhood data
    neighborhood_data = sample_neighborhoods[neighborhood]
    
    # Simple recommendation logic
    recommended_dog = sample_dogs[0]  # For now, always recommend first dog
    
    return html.Div([
        # Neighborhood stats
        html.Div([
            html.H3(f"📊 {neighborhood} Analysis", style={'color': '#2c3e50', 'margin-bottom': '20px'}),
            
            html.Div([
                html.Div([
                    html.H4("Safety Score", style={'margin': '0', 'color': '#7f8c8d', 'font-size': '14px'}),
                    html.H2(f"{neighborhood_data['safety_score']}/100", 
                           style={'margin': '5px 0', 'color': '#27ae60', 'font-size': '24px'}),
                ], style={'background': '#ffffff', 'padding': '20px', 'border-radius': '8px', 
                         'text-align': 'center', 'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.H4("Walkability", style={'margin': '0', 'color': '#7f8c8d', 'font-size': '14px'}),
                    html.H2(f"{neighborhood_data['walkability_score']}/100", 
                           style={'margin': '5px 0', 'color': '#3498db', 'font-size': '24px'}),
                ], style={'background': '#ffffff', 'padding': '20px', 'border-radius': '8px', 
                         'text-align': 'center', 'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.H4("Avg Rent", style={'margin': '0', 'color': '#7f8c8d', 'font-size': '14px'}),
                    html.H2(f"${neighborhood_data['avg_rent']:,}", 
                           style={'margin': '5px 0', 'color': '#9b59b6', 'font-size': '24px'}),
                ], style={'background': '#ffffff', 'padding': '20px', 'border-radius': '8px', 
                         'text-align': 'center', 'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.H4("Crime Reports", style={'margin': '0', 'color': '#7f8c8d', 'font-size': '14px'}),
                    html.H2(f"{neighborhood_data['crime_count']}", 
                           style={'margin': '5px 0', 'color': '#e74c3c', 'font-size': '24px'}),
                ], style={'background': '#ffffff', 'padding': '20px', 'border-radius': '8px', 
                         'text-align': 'center', 'width': '23%', 'display': 'inline-block'})
            ])
        ], style={'margin-bottom': '20px'}),
        
        # Dog recommendation
        html.Div([
            html.H3("🐕 Your Perfect Guardian", style={'color': '#2c3e50', 'margin-bottom': '20px'}),
            
            html.Div([
                html.H2(recommended_dog['name'], style={'color': '#2c3e50', 'margin-bottom': '10px'}),
                html.P(f"{recommended_dog['breed']} • {recommended_dog['size']}", 
                       style={'color': '#7f8c8d', 'font-size': '16px', 'margin-bottom': '15px'}),
                html.P(f"Protection Score: {recommended_dog['protection_score']}/100", 
                       style={'color': '#e74c3c', 'font-weight': 'bold', 'margin-bottom': '10px'}),
                html.P(f"Monthly Cost: ${recommended_dog['monthly_cost']}", 
                       style={'color': '#27ae60', 'font-weight': 'bold', 'margin-bottom': '15px'}),
                html.P("This is a perfect match for your neighborhood's safety needs!", 
                       style={'color': '#34495e', 'line-height': '1.5'})
            ])
        ], style={'background': '#ffffff', 'padding': '25px', 'border-radius': '10px', 
                 'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'})
    ])

# Run the server
if __name__ == '__main__':
    # Get port from environment variable (Render requirement)
    port = int(os.environ.get('PORT', 8050))
    
    # Run with Render settings
    app.run_server(
        debug=False,    # Turn off debug for production
        host='0.0.0.0', # Required for Render
        port=port       # Use Render's assigned port
    )
