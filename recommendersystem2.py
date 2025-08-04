import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="🐕 SF Dog Crisis Dashboard",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample data
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

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 10px;
    }
    .dog-card {
        background: #ffffff;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    .stSelectbox > div > div {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style='color: #2c3e50; margin-bottom: 10px;'>🐕 SF Dog Crisis Dashboard</h1>
    <p style='color: #7f8c8d; margin-bottom: 0;'>Prototype version - Real data dashboard coming soon!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for user input
with st.sidebar:
    st.header("🏠 Select Your Neighborhood")
    
    neighborhood = st.selectbox(
        "Choose your SF neighborhood:",
        options=list(sample_neighborhoods.keys()),
        index=None,
        placeholder="Select a neighborhood..."
    )
    
    find_match = st.button(
        "Find My Perfect Guardian",
        type="primary",
        use_container_width=True
    )

# Main content
if not neighborhood:
    st.markdown("""
    <div style='text-align: center; margin: 50px; color: #7f8c8d;'>
        <h4>👆 Select your neighborhood in the sidebar to see recommendations</h4>
    </div>
    """, unsafe_allow_html=True)
else:
    # Get neighborhood data
    neighborhood_data = sample_neighborhoods[neighborhood]
    
    # Neighborhood Analysis Section
    st.header(f"📊 {neighborhood} Analysis")
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Safety Score",
            value=f"{neighborhood_data['safety_score']}/100",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Walkability",
            value=f"{neighborhood_data['walkability_score']}/100",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Avg Rent",
            value=f"${neighborhood_data['avg_rent']:,}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Crime Reports",
            value=f"{neighborhood_data['crime_count']}",
            delta=None
        )
    
    st.divider()
    
    # Dog Recommendation Section
    st.header("🐕 Your Perfect Guardian")
    
    # Simple recommendation logic (for now, always recommend first dog)
    recommended_dog = sample_dogs[0]
    
    # Create dog recommendation card
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(recommended_dog['name'])
            st.write(f"**Breed:** {recommended_dog['breed']}")
            st.write(f"**Size:** {recommended_dog['size']}")
            st.write(f"**Protection Score:** {recommended_dog['protection_score']}/100")
            st.write(f"**Monthly Cost:** ${recommended_dog['monthly_cost']}")
            
            st.success("This is a perfect match for your neighborhood's safety needs!")
        
        with col2:
            # Create a simple visualization for the dog's stats
            dog_stats = pd.DataFrame({
                'Metric': ['Protection Score', 'Cost Efficiency'],
                'Value': [recommended_dog['protection_score'], 100 - (recommended_dog['monthly_cost'] / 2)]
            })
            
            fig = px.bar(
                dog_stats, 
                x='Metric', 
                y='Value',
                title=f"{recommended_dog['name']}'s Stats",
                color='Value',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                height=300,
                showlegend=False,
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Additional visualizations
    st.divider()
    st.header("📈 Neighborhood Comparison")
    
    # Create comparison chart
    comparison_data = []
    for name, data in sample_neighborhoods.items():
        comparison_data.append({
            'Neighborhood': name,
            'Safety Score': data['safety_score'],
            'Walkability Score': data['walkability_score'],
            'Crime Count': data['crime_count'],
            'Avg Rent': data['avg_rent']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.scatter(
            comparison_df,
            x='Safety Score',
            y='Walkability Score',
            size='Crime Count',
            color='Avg Rent',
            hover_name='Neighborhood',
            title='Safety vs Walkability (Bubble size = Crime Count)',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            comparison_df.sort_values('Safety Score'),
            x='Neighborhood',
            y='Safety Score',
            title='Safety Scores by Neighborhood',
            color='Safety Score',
            color_continuous_scale='RdYlGn'
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-top: 50px;'>
    <p>🐕 SF Dog Crisis Dashboard - Helping San Francisco residents find their perfect canine guardian</p>
    <p><em>This is a prototype version. Real data integration coming soon!</em></p>
</div>
""", unsafe_allow_html=True)
