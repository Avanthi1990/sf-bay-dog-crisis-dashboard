import dash
import os
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealDataSFDogMatcher:
    def __init__(self):
        """Initialize the interactive Dash application with real data"""
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
        ])

        # Load real data
        self.real_data = self._load_real_data()
        if not self.real_data:
            print("Failed to load real data. Check if CSV files exist.")
            return

        self.setup_layout()
        self.setup_callbacks()

    def _load_real_data(self):
        """Load and process your actual CSV files"""
        try:
            print("Loading real CSV data...")

            # FIXED: Use relative paths that work on Render
            rental_df = pd.read_csv('Cleaned_rental.csv')
            crime_df = pd.read_csv('Cleaned_sfpd.csv')
            walkability_df = pd.read_csv('Walkability_with_Neighborhoods.csv')
            organizations_df = pd.read_csv('petfinder_organizations_20250624_235502.csv')
            animals_df = pd.read_csv('petfinder_animals_20250624_235502.csv')

            print(f"Loaded rental data: {len(rental_df)} records")
            print(f"Loaded crime data: {len(crime_df)} records")
            print(f"Loaded walkability data: {len(walkability_df)} records")
            print(f"Loaded organizations: {len(organizations_df)} records")
            print(f"Loaded animals: {len(animals_df)} records")

            # Process neighborhood data
            neighborhoods = self._process_neighborhood_data(rental_df, crime_df, walkability_df)

            # Process dog data
            dogs = self._process_dog_data(animals_df, organizations_df)

            print(f"Processed {len(neighborhoods)} neighborhoods")
            print(f"Processed {len(dogs)} dogs")

            return {'neighborhoods': neighborhoods, 'dogs': dogs}

        except Exception as e:
            print(f"Error loading real data: {e}")
            print("Make sure these files exist in your directory:")
            print("   - Cleaned_rental.csv")
            print("   - Cleaned_sfpd.csv")
            print("   - Walkability_with_Neighborhoods.csv")
            print("   - petfinder_organizations_20250624_235502.csv")
            print("   - petfinder_animals_20250624_235502.csv")
            return None

    def _process_neighborhood_data(self, rental_df, crime_df, walkability_df):
        """Process neighborhood data from CSV files"""
        neighborhoods = {}

        # Get unique neighborhoods from rental data
        unique_neighborhoods = rental_df['analysis_neighborhood'].dropna().unique()

        for neighborhood in unique_neighborhoods:
            try:
                # Crime data
                neighborhood_crimes = crime_df[
                    crime_df['Analysis Neighborhood'] == neighborhood
                    ]
                crime_count = len(neighborhood_crimes)

                # Walkability data
                neighborhood_walk = walkability_df[
                    walkability_df['neighborhood_name'] == neighborhood
                    ]
                walk_score = neighborhood_walk['NatWalkInd'].mean() if len(neighborhood_walk) > 0 else 10

                # Rental data
                neighborhood_rent = rental_df[
                    rental_df['analysis_neighborhood'] == neighborhood
                    ]
                avg_rent = neighborhood_rent['rent_avg'].mean() if len(neighborhood_rent) > 0 else 3000

                # Calculate derived metrics
                # Safety Score: fewer crimes = higher score
                max_crimes = 200  # Reasonable max for normalization
                safety_score = max(0, 100 - (crime_count / max_crimes * 100))

                # Walkability Score: scale to 0-100
                walkability_score = min(100, (walk_score / 20) * 100)

                # Determine protection need based on safety
                if safety_score < 40:
                    protection_need = 'Very High'
                elif safety_score < 60:
                    protection_need = 'High'
                elif safety_score < 80:
                    protection_need = 'Medium'
                else:
                    protection_need = 'Low'

                # Budget calculation (5% of rent for pet expenses)
                budget_limit = max(50, avg_rent * 0.05)  # Minimum $50

                neighborhoods[neighborhood] = {
                    'safety_score': round(safety_score, 1),
                    'walkability_score': round(walkability_score, 1),
                    'avg_rent': round(avg_rent, 0),
                    'crime_count': crime_count,
                    'protection_need': protection_need,
                    'budget_limit': round(budget_limit, 0),
                    'walk_index': round(walk_score, 1)
                }

            except Exception as e:
                print(f"Warning: Could not process neighborhood {neighborhood}: {e}")
                continue

        return neighborhoods

    def _process_dog_data(self, animals_df, organizations_df):
        """Process dog data from CSV files"""
        dogs = []

        # Filter for dogs only
        dogs_df = animals_df[animals_df['type'] == 'Dog'].copy()

        # SF and nearby zip codes for filtering
        sf_zip_codes = ['94102', '94103', '94104', '94105', '94107', '94108', '94109', '94110',
                        '94111', '94112', '94114', '94115', '94116', '94117', '94118', '94121',
                        '94122', '94123', '94124', '94127', '94131', '94132', '94133', '94134']

        # Filter organizations to SF area
        sf_orgs = organizations_df[
            (organizations_df['address_state'] == 'CA') &
            (organizations_df['address_postcode'].isin(sf_zip_codes))
            ]

        # If no SF orgs found, use broader CA filter
        if len(sf_orgs) == 0:
            sf_orgs = organizations_df[
                organizations_df['address_state'] == 'CA'
                ].head(20)  # Limit for performance

        # Filter dogs to SF organizations
        sf_org_ids = sf_orgs['organization_id'].tolist()
        sf_dogs = dogs_df[dogs_df['organization_id'].isin(sf_org_ids)].copy()

        # If still no dogs, take first 20 dogs from any CA organization
        if len(sf_dogs) == 0:
            ca_orgs = organizations_df[organizations_df['address_state'] == 'CA']
            ca_org_ids = ca_orgs['organization_id'].tolist()
            sf_dogs = dogs_df[dogs_df['organization_id'].isin(ca_org_ids)].head(20)

        print(f"Found {len(sf_dogs)} dogs in SF area from {len(sf_orgs)} organizations")

        # Process each dog
        for _, dog in sf_dogs.iterrows():
            try:
                # Basic info
                name = str(dog.get('name', f"Dog {dog.get('animal_id', 'Unknown')}")[:30])
                breed = str(dog.get('primary_breed', 'Mixed Breed'))[:30]
                size = str(dog.get('size', 'Medium'))
                age = str(dog.get('age', 'Adult'))

                # Calculate protection score based on size and breed
                protection_score = self._calculate_protection_score(size, breed, age)

                # Calculate other scores
                family_score = self._calculate_family_score(dog)
                training_score = self._calculate_training_score(dog, age)

                # Calculate monthly cost based on size
                monthly_cost = self._calculate_monthly_cost(size, dog.get('special_needs', False))

                # Get shelter information
                shelter_info = self._get_shelter_info(dog.get('organization_id'), sf_orgs)

                # Parse boolean fields safely
                good_with_children = self._parse_boolean_safe(dog.get('good_with_children'))
                house_trained = self._parse_boolean_safe(dog.get('house_trained'))

                # Description
                description = str(dog.get('description', 'Loving companion looking for a home'))
                if len(description) > 150:
                    description = description[:150] + "..."

                dogs.append({
                    'name': name,
                    'breed': breed,
                    'size': size,
                    'age': age,
                    'protection_score': protection_score,
                    'family_score': family_score,
                    'training_score': training_score,
                    'monthly_cost': monthly_cost,
                    'good_with_children': good_with_children,
                    'house_trained': house_trained,
                    'shelter': shelter_info['name'],
                    'shelter_phone': shelter_info['phone'],
                    'description': description,
                    'adoption_url': str(dog.get('url', 'https://petfinder.com')),
                    'photo_url': str(dog.get('photo_medium', ''))
                })

            except Exception as e:
                print(f"Warning: Could not process dog {dog.get('animal_id', 'Unknown')}: {e}")
                continue

        return dogs

    def _calculate_protection_score(self, size, breed, age):
        """Calculate protection score based on dog characteristics"""
        score = 40  # Base score

        # Size factor
        if size == 'Large':
            score += 30
        elif size == 'Medium':
            score += 20
        elif size == 'Extra Large':
            score += 35
        elif size == 'Small':
            score += 5

        # Breed factor (protective breeds)
        protective_breeds = ['german shepherd', 'pit bull', 'rottweiler', 'mastiff',
                             'doberman', 'boxer', 'bulldog', 'belgian', 'akita']
        breed_lower = breed.lower()
        if any(pb in breed_lower for pb in protective_breeds):
            score += 25

        # Age factor
        if age in ['Adult', 'Senior']:
            score += 10  # Mature dogs often better for protection
        elif age == 'Young':
            score += 5

        return min(100, score)

    def _calculate_family_score(self, dog):
        """Calculate family friendliness score"""
        score = 60  # Base score

        # Good with children
        if self._parse_boolean_safe(dog.get('good_with_children')):
            score += 30

        # House trained
        if self._parse_boolean_safe(dog.get('house_trained')):
            score += 15

        # Shots current
        if self._parse_boolean_safe(dog.get('shots_current')):
            score += 10

        # Spayed/neutered
        if self._parse_boolean_safe(dog.get('spayed_neutered')):
            score += 10

        # Not special needs
        if not self._parse_boolean_safe(dog.get('special_needs')):
            score += 5

        return min(100, score)

    def _calculate_training_score(self, dog, age):
        """Calculate training ease score"""
        score = 50  # Base score

        # House trained
        if self._parse_boolean_safe(dog.get('house_trained')):
            score += 30

        # Age factor
        if age == 'Young':
            score += 20  # Easier to train when young
        elif age == 'Adult':
            score += 15
        elif age == 'Baby':
            score += 10

        # Not special needs
        if not self._parse_boolean_safe(dog.get('special_needs')):
            score += 15

        return min(100, score)

    def _calculate_monthly_cost(self, size, special_needs):
        """Calculate estimated monthly cost"""
        base_costs = {
            'Small': 70,
            'Medium': 95,
            'Large': 125,
            'Extra Large': 155
        }

        cost = base_costs.get(size, 95)

        if special_needs:
            cost *= 1.4  # 40% increase for special needs

        return round(cost, 0)

    def _get_shelter_info(self, org_id, organizations_df):
        """Get shelter information for a given organization ID"""
        try:
            org = organizations_df[organizations_df['organization_id'] == org_id]
            if len(org) > 0:
                org = org.iloc[0]
                return {
                    'name': str(org.get('name', 'Local Shelter'))[:40],
                    'phone': str(org.get('phone', '(415) 555-0000')),
                    'email': str(org.get('email', 'info@shelter.org'))
                }
        except:
            pass

        return {
            'name': 'Local SF Shelter',
            'phone': '(415) 555-0000',
            'email': 'info@shelter.org'
        }

    def _parse_boolean_safe(self, value):
        """Safely parse boolean-like values"""
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ['true', 'yes', '1', 'y']
        return bool(value)

    def setup_layout(self):
        """Setup the Dash app layout"""
        if not self.real_data:
            self.app.layout = html.Div([
                html.H1("Data Loading Error", style={'text-align': 'center', 'color': 'red'}),
                html.P("Could not load CSV files. Please check that all required files are in the directory.",
                       style={'text-align': 'center', 'margin': '50px'})
            ])
            return

        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1([
                    html.I(className="fas fa-shield-dog", style={'margin-right': '15px', 'color': '#e74c3c'}),
                    "SF Safety Guardian: Real Data Matching"
                ], style={
                    'text-align': 'center',
                    'color': '#2c3e50',
                    'margin-bottom': '10px',
                    'font-family': 'Arial, sans-serif'
                }),
                html.P(
                    f"Live data: {len(self.real_data['neighborhoods'])} neighborhoods • {len(self.real_data['dogs'])} rescue dogs",
                    style={
                        'text-align': 'center',
                        'font-size': '18px',
                        'color': '#7f8c8d',
                        'margin-bottom': '30px'
                    }
                )
            ], style={
                'background': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
                'padding': '30px',
                'border-radius': '10px',
                'margin-bottom': '20px'
            }),

            # User Input Section
            html.Div([
                html.H3([
                    html.I(className="fas fa-map-marker-alt", style={'margin-right': '10px'}),
                    "Tell Us About You"
                ], style={'color': '#2c3e50', 'margin-bottom': '20px'}),

                dcc.Dropdown(
                    id='neighborhood-dropdown',
                    options=[
                        {'label': f"📍 {neighborhood}", 'value': neighborhood}
                        for neighborhood in sorted(self.real_data['neighborhoods'].keys())
                    ],
                    placeholder="🏠 Select your SF neighborhood...",
                    style={'margin-bottom': '20px', 'font-size': '16px'}
                ),

                html.Div([
                    html.Div([
                        html.Label("🐕 Preferred Dog Size:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                        dcc.Dropdown(
                            id='size-preference',
                            options=[
                                {'label': '🐕 Any Size', 'value': 'Any'},
                                {'label': '🐕‍🦺 Small (< 25 lbs)', 'value': 'Small'},
                                {'label': '🐕 Medium (25-60 lbs)', 'value': 'Medium'},
                                {'label': '🐕‍🦺 Large (60-90 lbs)', 'value': 'Large'},
                                {'label': '🐕 Extra Large (90+ lbs)', 'value': 'Extra Large'}
                            ],
                            value='Any',
                            style={'margin-bottom': '15px'}
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.Label("🛡️ Protection Priority:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                        dcc.Dropdown(
                            id='protection-preference',
                            options=[
                                {'label': '🤝 Companionship Focus', 'value': 'Low'},
                                {'label': '👁️ Some Protection', 'value': 'Medium'},
                                {'label': '🛡️ High Protection', 'value': 'High'},
                                {'label': '🚨 Maximum Protection', 'value': 'Very High'}
                            ],
                            value='Medium',
                            style={'margin-bottom': '15px'}
                        )
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                ], style={'margin-bottom': '20px'}),

                html.Button(
                    [html.I(className="fas fa-search", style={'margin-right': '8px'}),
                     "Find My Perfect Guardian"],
                    id='find-match-button',
                    n_clicks=0,
                    style={
                        'background': 'linear-gradient(45deg, #e74c3c, #c0392b)',
                        'color': 'white',
                        'border': 'none',
                        'font-size': '18px',
                        'border-radius': '25px',
                        'cursor': 'pointer',
                        'width': '100%',
                        'font-weight': 'bold'
                    }
                )
            ], style={
                'background': '#ffffff',
                'padding': '25px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'margin-bottom': '20px'
            }) if len(recommended_dogs) > 1 else html.Div(),

            # Call to Action
            html.Div([
                html.H3("Ready to Save a Life?", style={'text-align': 'center', 'color': '#2c3e50'}),
                html.P(f"Contact {top_dog['shelter']} today to meet {top_dog['name']}",
                       style={'text-align': 'center', 'font-size': '18px', 'margin-bottom': '20px'}),

                html.Div([
                    html.A([
                        html.I(className="fas fa-phone", style={'margin-right': '8px'}),
                        f"Call {top_dog['shelter_phone']}"
                    ], href=f"tel:{top_dog['shelter_phone']}",
                        style={
                            'background': '#27ae60',
                            'color': 'white',
                            'padding': '15px 25px',
                            'text-decoration': 'none',
                            'border-radius': '25px',
                            'margin-right': '15px',
                            'display': 'inline-block'
                        }),

                    html.A([
                        html.I(className="fas fa-external-link-alt", style={'margin-right': '8px'}),
                        "View Online Profile"
                    ], href=top_dog['adoption_url'], target='_blank',
                        style={
                            'background': '#3498db',
                            'color': 'white',
                            'padding': '15px 25px',
                            'text-decoration': 'none',
                            'border-radius': '25px',
                            'display': 'inline-block'
                        })
                ], style={'text-align': 'center'})
            ], style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'padding': '30px',
                'border-radius': '10px',
                'text-align': 'center'
            })
        ])

    def _create_metric_card(self, title, value, color, subtitle):
        """Create a metric card component"""
        return html.Div([
            html.H4(title, style={'margin': '0', 'color': '#7f8c8d', 'font-size': '14px'}),
            html.H2(value, style={'margin': '5px 0', 'color': color, 'font-size': '24px'}),
            html.P(subtitle, style={'margin': '0', 'color': '#bdc3c7', 'font-size': '12px'})
        ], style={
            'background': '#ffffff',
            'padding': '20px',
            'border-radius': '8px',
            'text-align': 'center',
            'width': '22%',
            'box-shadow': f'0 2px 4px {color}22'
        })

    def _get_safety_color(self, score):
        """Get color based on safety score"""
        if score >= 80: return '#27ae60'
        elif score >= 60: return '#f39c12'
        elif score >= 40: return '#e67e22'
        else: return '#e74c3c'

    def _get_crime_color(self, count):
        """Get color based on crime count"""
        if count <= 30: return '#27ae60'
        elif count <= 60: return '#f39c12'
        elif count <= 100: return '#e67e22'
        else: return '#e74c3c'

    def _create_dog_recommendation_card(self, dog, impact_metrics):
        """Create the main dog recommendation card"""
        return html.Div([
            html.Div([
                # Dog image placeholder
                html.Div([
                    html.I(className="fas fa-dog", style={'font-size': '80px', 'color': '#bdc3c7'})
                ], style={
                    'width': '150px',
                    'height': '150px',
                    'background': '#ecf0f1',
                    'border-radius': '10px',
                    'display': 'flex',
                    'align-items': 'center',
                    'justify-content': 'center',
                    'margin-right': '25px'
                }),

                # Dog details
                html.Div([
                    html.H2(f"{dog['name']}", style={'color': '#2c3e50', 'margin-bottom': '10px'}),
                    html.P(f"{dog['breed']} • {dog['size']} • {dog['age']}",
                           style={'color': '#7f8c8d', 'font-size': '16px', 'margin-bottom': '15px'}),

                    html.Div([
                        html.Span("Match Score: ", style={'font-weight': 'bold'}),
                        html.Span(f"{dog['compatibility_score']:.0f}/100",
                                  style={'color': '#e74c3c', 'font-weight': 'bold', 'font-size': '18px'})
                    ], style={'margin-bottom': '10px'}),

                    html.Div([
                        html.Span("Monthly Cost: ", style={'font-weight': 'bold'}),
                        html.Span(f"${dog['monthly_cost']}", style={'color': '#27ae60', 'font-weight': 'bold'})
                    ], style={'margin-bottom': '15px'}),

                    html.P(dog['description'], style={'color': '#34495e', 'line-height': '1.5'})
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'align-items': 'flex-start', 'margin-bottom': '25px'}),

            # Benefits
            html.Div([
                html.H4("Why This Is Your Perfect Match:", style={'color': '#2c3e50', 'margin-bottom': '15px'}),

                html.Div([
                    html.Div([
                        html.I(className="fas fa-shield-alt", style={'color': '#e74c3c', 'margin-right': '8px'}),
                        f"Safety boost: +{impact_metrics.get('safety_improvement', 0):.0f} points"
                    ], style={'margin-bottom': '8px'}),

                    html.Div([
                        html.I(className="fas fa-walking", style={'color': '#3498db', 'margin-right': '8px'}),
                        f"Exercise motivation: {dog['monthly_exercise']:.0f} min/day"
                    ], style={'margin-bottom': '8px'}),

                    html.Div([
                        html.I(className="fas fa-eye", style={'color': '#9b59b6', 'margin-right': '8px'}),
                        f"Deterrent effect: {dog['deterrent_effect']}"
                    ], style={'margin-bottom': '8px'}),

                    html.Div([
                        html.I(className="fas fa-home", style={'color': '#27ae60', 'margin-right': '8px'}),
                        "House-trained" if dog['house_trained'] else "Needs house training"
                    ], style={'margin-bottom': '8px'}),

                    html.Div([
                        html.I(className="fas fa-baby", style={'color': '#f39c12', 'margin-right': '8px'}),
                        "Great with children" if dog['good_with_children'] else "Adult-only household"
                    ])
                ], style={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px'})
            ])
        ])

    def _create_impact_visualization(self, impact_metrics):
        """Create impact visualization charts"""

        # Safety improvement gauge
        safety_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=impact_metrics.get('new_safety_score', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Your New Safety Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#e74c3c"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': impact_metrics.get('current_safety_score', 0)
                }
            }
        ))
        safety_fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))

        # Impact metrics
        impact_data = {
            'Metric': ['Dogs Saved', 'Safety Points', 'Walking Hours/Month', 'Community Impact'],
            'Value': [
                impact_metrics.get('dogs_saved', 0),
                impact_metrics.get('safety_improvement', 0),
                impact_metrics.get('monthly_walking_hours', 0),
                5 if impact_metrics.get('community_connections') == 'High' else 3
            ],
            'Icon': ['Dog', 'Shield', 'Walking', 'Community']
        }

        impact_fig = go.Figure(data=[
            go.Bar(
                x=impact_data['Value'],
                y=impact_data['Metric'],
                orientation='h',
                marker_color=['#e74c3c', '#3498db', '#27ae60', '#9b59b6'],
                text=[f"{icon} {val}" for icon, val in zip(impact_data['Icon'], impact_data['Value'])],
                textposition='inside'
            )
        ])
        impact_fig.update_layout(
            title="Your Adoption Impact",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )

        return html.Div([
            html.Div([
                dcc.Graph(figure=safety_fig)
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(figure=impact_fig)
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ])

    def _create_alternative_dog_card(self, dog):
        """Create alternative dog option card"""
        return html.Div([
            html.Div([
                html.I(className="fas fa-dog", style={'font-size': '40px', 'color': '#bdc3c7'})
            ], style={
                'width': '80px',
                'height': '80px',
                'background': '#ecf0f1',
                'border-radius': '8px',
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center',
                'margin-bottom': '15px'
            }),

            html.H4(dog['name'], style={'color': '#2c3e50', 'margin-bottom': '5px'}),
            html.P(f"{dog['breed']}", style={'color': '#7f8c8d', 'font-size': '14px', 'margin-bottom': '10px'}),
            html.P(f"Match: {dog['compatibility_score']:.0f}/100",
                   style={'color': '#e74c3c', 'font-weight': 'bold', 'margin-bottom': '10px'}),
            html.P(f"${dog['monthly_cost']}/month",
                   style={'color': '#27ae60', 'font-weight': 'bold'})
        ], style={
            'background': '#f8f9fa',
            'padding': '20px',
            'border-radius': '8px',
            'text-align': 'center',
            'width': '30%',
            'border': '2px solid #ecf0f1'
        })

    def run_server(self, debug=True, host='127.0.0.1', port=8050):
        """Run the Dash server"""
        if not self.real_data:
            print("Cannot start server - data loading failed")
            return

        print("Starting SF Safety-Dog Matching App with REAL DATA...")
        print(f"Open your browser to: http://{host}:{port}")
        print("Features:")
        print("   • Real SF neighborhood crime and walkability data")
        print("   • Live Petfinder rescue dog data")
        print("   • Personalized matching algorithm")
        print("   • Impact visualization")
        print("   • Direct shelter contact links")
        print(f"Data loaded: {len(self.real_data['neighborhoods'])} neighborhoods, {len(self.real_data['dogs'])} dogs")

        self.app.run(debug=debug, host=host, port=port)

def main():
    """Main function to run the interactive app with real data"""
    print("SF Safety Guardian - Real Data Interactive System")
    print("=" * 65)

    # Create the app
    app = RealDataSFDogMatcher()

    # FIXED: Single run_server call with Render configuration
    # Get port from environment variable (Render requirement)
    port = int(os.environ.get('PORT', 8050))
    
    # Run the server with Render settings
    app.run_server(
        debug=False,    # Turn off debug for production
        host='0.0.0.0', # Required for Render
        port=port       # Use Render's assigned port
    )

# FIXED: Single main execution point
if __name__ == "__main__":
    main()0, 0, 0, 0.1)',
                'margin-bottom': '20px'
            }),

            # Results Section
            html.Div(id='results-container', children=[
                html.Div([
                    html.H4("👆 Select your neighborhood above to see personalized recommendations",
                            style={'text-align': 'center', 'color': '#7f8c8d', 'margin': '50px'})
                ])
            ])
        ], style={
            'max-width': '1200px',
            'margin': '0 auto',
            'padding': '20px',
            'font-family': 'Arial, sans-serif',
            'background-color': '#f8f9fa'
        })

    def setup_callbacks(self):
        """Setup interactive callbacks"""
        if not self.real_data:
            return

        @self.app.callback(
            Output('results-container', 'children'),
            [Input('find-match-button', 'n_clicks')],
            [State('neighborhood-dropdown', 'value'),
             State('size-preference', 'value'),
             State('protection-preference', 'value')]
        )
        def update_recommendations(n_clicks, neighborhood, size_pref, protection_pref):
            if n_clicks == 0 or not neighborhood:
                return html.Div([
                    html.H4("👆 Select your neighborhood above to see personalized recommendations",
                            style={'text-align': 'center', 'color': '#7f8c8d', 'margin': '50px'})
                ])

            # Get neighborhood data
            neighborhood_data = self.real_data['neighborhoods'].get(neighborhood)
            if not neighborhood_data:
                return html.Div([
                    html.H3("Neighborhood Data Not Found", style={'color': '#e74c3c', 'text-align': 'center'}),
                    html.P(f"Could not find data for {neighborhood}. Please try another neighborhood.",
                           style={'text-align': 'center', 'margin': '20px'})
                ])

            # Filter and rank dogs
            recommended_dogs = self._get_personalized_recommendations(
                neighborhood_data, size_pref, protection_pref
            )

            # Calculate impact metrics
            impact_metrics = self._calculate_impact_metrics(neighborhood_data, recommended_dogs)

            return self._create_results_layout(
                neighborhood, neighborhood_data, recommended_dogs, impact_metrics
            )

    def _get_personalized_recommendations(self, neighborhood_data, size_pref, protection_pref):
        """Get personalized dog recommendations based on user preferences"""

        dogs = self.real_data['dogs'].copy()

        # Filter by size preference
        if size_pref != 'Any':
            dogs = [dog for dog in dogs if dog['size'] == size_pref]

        # Score dogs based on neighborhood needs and user preferences
        for dog in dogs:
            # Base compatibility score
            safety_match = self._calculate_safety_match(neighborhood_data, dog)
            budget_match = self._calculate_budget_match(neighborhood_data, dog)
            preference_match = self._calculate_preference_match(protection_pref, dog)

            # Combined score
            dog['compatibility_score'] = (
                    safety_match * 0.4 +
                    budget_match * 0.3 +
                    preference_match * 0.3
            )

            # Calculate specific benefits
            dog['safety_improvement'] = min(30, dog['protection_score'] * 0.3)
            dog['monthly_exercise'] = 60 + (dog['protection_score'] * 0.5)
            dog['deterrent_effect'] = 'High' if dog['protection_score'] >= 80 else 'Medium' if dog['protection_score'] >= 60 else 'Low'

        # Sort by compatibility and return top 5
        dogs.sort(key=lambda x: x['compatibility_score'], reverse=True)
        return dogs[:5]

    def _calculate_safety_match(self, neighborhood_data, dog):
        """Calculate how well dog matches neighborhood safety needs"""
        protection_need = neighborhood_data['protection_need']
        dog_protection = dog['protection_score']

        if protection_need == 'Very High':
            return min(100, dog_protection * 1.2)
        elif protection_need == 'High':
            return dog_protection
        elif protection_need == 'Medium':
            return 100 - abs(dog_protection - 60) * 0.5
        else:  # Low
            return 100 - (dog_protection - 40) * 0.3 if dog_protection > 40 else 100

    def _calculate_budget_match(self, neighborhood_data, dog):
        """Calculate budget compatibility"""
        budget_limit = neighborhood_data['budget_limit']
        dog_cost = dog['monthly_cost']

        if dog_cost <= budget_limit:
            return 100
        elif dog_cost <= budget_limit * 1.2:
            return 80
        else:
            return max(0, 100 - (dog_cost - budget_limit) / budget_limit * 100)

    def _calculate_preference_match(self, protection_pref, dog):
        """Calculate how well dog matches user protection preference"""
        pref_scores = {
            'Low': (0, 40),
            'Medium': (40, 70),
            'High': (70, 85),
            'Very High': (85, 100)
        }

        min_score, max_score = pref_scores[protection_pref]
        dog_score = dog['protection_score']

        if min_score <= dog_score <= max_score:
            return 100
        else:
            distance = min(abs(dog_score - min_score), abs(dog_score - max_score))
            return max(0, 100 - distance * 2)

    def _calculate_impact_metrics(self, neighborhood_data, recommended_dogs):
        """Calculate the impact of adoption on both user and dogs"""

        if not recommended_dogs:
            return {}

        top_dog = recommended_dogs[0]

        # Safety improvement
        current_safety = neighborhood_data['safety_score']
        safety_boost = top_dog['safety_improvement']
        new_safety_score = min(100, current_safety + safety_boost)

        # Life impact
        dogs_saved = len(recommended_dogs)  # Dogs that could be saved from shelters

        # Community impact
        walking_minutes = top_dog['monthly_exercise']
        monthly_walking_hours = walking_minutes * 30 / 60  # Convert to monthly hours

        return {
            'current_safety_score': current_safety,
            'new_safety_score': new_safety_score,
            'safety_improvement': safety_boost,
            'dogs_saved': dogs_saved,
            'monthly_walking_hours': monthly_walking_hours,
            'deterrent_effect': top_dog['deterrent_effect'],
            'community_connections': 'High' if walking_minutes > 75 else 'Medium'
        }

    def _create_results_layout(self, neighborhood, neighborhood_data, recommended_dogs, impact_metrics):
        """Create the results layout with all visualizations and recommendations"""

        if not recommended_dogs:
            return html.Div([
                html.H3("No Perfect Matches Found", style={'color': '#e74c3c', 'text-align': 'center'}),
                html.P("Try adjusting your preferences or consider nearby neighborhoods.",
                       style={'text-align': 'center', 'margin': '20px'})
            ])

        top_dog = recommended_dogs[0]

        return html.Div([
            # Neighborhood Analysis
            html.Div([
                html.H3([
                    html.I(className="fas fa-chart-line", style={'margin-right': '10px'}),
                    f"Your {neighborhood} Analysis (Real Data)"
                ], style={'color': '#2c3e50', 'margin-bottom': '20px'}),

                # Metrics Cards
                html.Div([
                    self._create_metric_card(
                        "Safety Score",
                        f"{neighborhood_data['safety_score']}/100",
                        self._get_safety_color(neighborhood_data['safety_score']),
                        f"Protection needed: {neighborhood_data['protection_need']}"
                    ),
                    self._create_metric_card(
                        "Walkability",
                        f"{neighborhood_data['walkability_score']}/100",
                        '#3498db',
                        f"Walk Index: {neighborhood_data['walk_index']}"
                    ),
                    self._create_metric_card(
                        "Avg Rent",
                        f"${neighborhood_data['avg_rent']:,}",
                        '#9b59b6',
                        f"Dog budget: ${neighborhood_data['budget_limit']}/month"
                    ),
                    self._create_metric_card(
                        "Crime Reports",
                        f"{neighborhood_data['crime_count']} incidents",
                        self._get_crime_color(neighborhood_data['crime_count']),
                        "From real SFPD data"
                    )
                ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'})
            ], style={
                'background': '#ffffff',
                'padding': '25px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'margin-bottom': '20px'
            }),

            # Perfect Match Section
            html.Div([
                html.H3([
                    html.I(className="fas fa-heart", style={'margin-right': '10px', 'color': '#e74c3c'}),
                    "Your Perfect Guardian (Real Rescue Dog)"
                ], style={'color': '#2c3e50', 'margin-bottom': '20px'}),

                self._create_dog_recommendation_card(top_dog, impact_metrics)
            ], style={
                'background': '#ffffff',
                'padding': '25px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'margin-bottom': '20px'
            }),

            # Impact Visualization
            html.Div([
                html.H3([
                    html.I(className="fas fa-chart-bar", style={'margin-right': '10px'}),
                    "Your Adoption Impact"
                ], style={'color': '#2c3e50', 'margin-bottom': '20px'}),

                self._create_impact_visualization(impact_metrics)
            ], style={
                'background': '#ffffff',
                'padding': '25px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'margin-bottom': '20px'
            }),

            # Alternative Options
            html.Div([
                html.H3([
                    html.I(className="fas fa-list", style={'margin-right': '10px'}),
                    "Alternative Matches"
                ], style={'color': '#2c3e50', 'margin-bottom': '20px'}),

                html.Div([
                    self._create_alternative_dog_card(dog)
                    for dog in recommended_dogs[1:]
                ], style={'display': 'flex', 'justify-content': 'space-between'})
            ], style={
                'background': '#ffffff',
                'padding': '25px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(