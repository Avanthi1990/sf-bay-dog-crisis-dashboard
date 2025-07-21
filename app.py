import dash
from dash import html
import os

# Create app
app = dash.Dash(__name__)
server = app.server

# Simple layout
app.layout = html.Div([
    html.H1("SF Dog Crisis Dashboard - TEST VERSION"),
    html.P("If you can see this, the app is working!"),
    html.P("🐕 Dashboard coming soon...")
], style={'text-align': 'center', 'margin': '50px'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)
