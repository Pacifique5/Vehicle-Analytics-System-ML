import plotly.express as px
import plotly.offline as opy
import plotly.graph_objects as go
import pandas as pd
import json

def create_rwanda_map(df, height=700):
    """
    Create Rwanda geographical map with district boundaries and client distribution using GeoJSON
    """
    
    # Load GeoJSON file
    with open('dummy-data/rwanda_districts.geojson', 'r') as f:
        geojson_data = json.load(f)
    
    # Calculate client count by district
    district_data = df.groupby('district').agg(
        Client_Count=('client_name', 'size')
    ).reset_index()
    district_data.columns = ['District', 'Client_Count']
    
    # District coordinates (center points for labels)
    district_coords = {
        'Gasabo': {'lat': -1.9536, 'lon': 30.1380},
        'Kicukiro': {'lat': -1.9892, 'lon': 30.1038},
        'Nyarugenge': {'lat': -1.9706, 'lon': 30.0587},
        'Bugesera': {'lat': -2.2833, 'lon': 30.2000},
        'Gatsibo': {'lat': -1.5833, 'lon': 30.4167},
        'Kayonza': {'lat': -1.8833, 'lon': 30.6333},
        'Kirehe': {'lat': -2.2167, 'lon': 30.7167},
        'Ngoma': {'lat': -2.1667, 'lon': 30.5167},
        'Nyagatare': {'lat': -1.3000, 'lon': 30.3333},
        'Rwamagana': {'lat': -1.9500, 'lon': 30.4333},
        'Burera': {'lat': -1.4833, 'lon': 29.8833},
        'Gakenke': {'lat': -1.6833, 'lon': 29.7833},
        'Gicumbi': {'lat': -1.5833, 'lon': 30.0667},
        'Musanze': {'lat': -1.5000, 'lon': 29.6333},
        'Rulindo': {'lat': -1.7667, 'lon': 30.0667},
        'Gisagara': {'lat': -2.5833, 'lon': 29.8333},
        'Huye': {'lat': -2.5833, 'lon': 29.7333},
        'Kamonyi': {'lat': -2.0333, 'lon': 29.9667},
        'Muhanga': {'lat': -2.0833, 'lon': 29.7500},
        'Nyamagabe': {'lat': -2.4500, 'lon': 29.4833},
        'Nyanza': {'lat': -2.3500, 'lon': 29.7500},
        'Nyaruguru': {'lat': -2.6833, 'lon': 29.4333},
        'Ruhango': {'lat': -2.2333, 'lon': 29.7833},
        'Karongi': {'lat': -2.0000, 'lon': 29.3833},
        'Ngororero': {'lat': -1.8333, 'lon': 29.5333},
        'Nyabihu': {'lat': -1.6333, 'lon': 29.5000},
        'Nyamasheke': {'lat': -2.3500, 'lon': 29.1167},
        'Rubavu': {'lat': -1.6833, 'lon': 29.3333},
        'Rusizi': {'lat': -2.4833, 'lon': 28.9000},
        'Rutsiro': {'lat': -1.9667, 'lon': 29.3333}
    }
    
    # Add coordinates to district data
    district_data['lat'] = district_data['District'].map(lambda x: district_coords.get(x, {}).get('lat'))
    district_data['lon'] = district_data['District'].map(lambda x: district_coords.get(x, {}).get('lon'))
    
    # Create text labels
    district_data['Label'] = district_data.apply(
        lambda row: f"{row['District']}<br>{row['Client_Count']}", 
        axis=1
    )
    
    # Create choropleth map with GeoJSON
    fig = px.choropleth(
        district_data,
        geojson=geojson_data,
        locations='District',
        featureidkey='properties.NAME_2',
        color='Client_Count',
        hover_name='District',
        hover_data={'Client_Count': True, 'District': False},
        color_continuous_scale='RdYlGn',
        title='Rwanda Districts - Vehicle Clients Distribution',
        labels={'Client_Count': 'Number of Clients'}
    )
    
    # Add district names and client counts as text on the map
    fig.add_trace(go.Scattergeo(
        lon=district_data['lon'],
        lat=district_data['lat'],
        text=district_data['Label'],
        mode='text',
        textfont=dict(
            size=9,
            color='white',
            family='Arial Black'
        ),
        textposition='middle center',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Update layout to focus on Rwanda
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcoastlines=True,
        coastlinecolor='#34495e',
        showland=True,
        landcolor='rgb(243, 243, 243)',
        showocean=False,
        oceancolor='rgb(204, 229, 255)',
        showlakes=False,
        lakecolor='rgb(200, 230, 255)',
        showcountries=False,
        countrycolor='white',
        countrywidth=1
    )
    
    fig.update_layout(
        height=height,
        title={
            'text': 'Rwanda Districts - Vehicle Clients Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return opy.plot(fig, output_type="div")


def get_district_summary_table(df):
    """
    Create a summary table of clients by province and district
    """
    summary = df.groupby(['province', 'district']).agg({
        'client_name': 'count',
        'estimated_income': 'mean',
        'selling_price': 'mean'
    }).reset_index()
    
    summary.columns = ['Province', 'District', 'Total Clients', 'Avg Income', 'Avg Vehicle Price']
    summary = summary.sort_values(['Province', 'Total Clients'], ascending=[True, False])
    
    return summary.to_html(
        classes="table table-bordered table-striped table-sm table-hover",
        float_format="%.2f",
        justify="center",
        index=False
    )
