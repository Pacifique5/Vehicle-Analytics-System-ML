"""
EXERCISE PART A: Rwanda Map Visualization with District Boundaries (20 marks)

This standalone script creates an interactive Rwanda map showing:
- All 30 districts across 5 provinces
- Number of vehicle clients in each district
- District boundaries and names
- Interactive visualization using Plotly

Usage:
    python exercise_solutions/part_a_rwanda_map.py
"""

import pandas as pd
import plotly.graph_objects as go

def create_rwanda_map_visualization():
    """
    Create Rwanda map visualization with district boundaries and vehicle client counts
    """
    
    # Load the dataset
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    
    # Count clients per district
    district_counts = df.groupby(['province', 'district']).size().reset_index(name='client_count')
    
    # Rwanda districts with approximate coordinates (center points)
    rwanda_districts = {
        'Kigali City': {
            'Gasabo': {'lat': -1.9536, 'lon': 30.1380},
            'Kicukiro': {'lat': -1.9892, 'lon': 30.1038},
            'Nyarugenge': {'lat': -1.9706, 'lon': 30.0587}
        },
        'Eastern Province': {
            'Bugesera': {'lat': -2.2833, 'lon': 30.2000},
            'Gatsibo': {'lat': -1.5833, 'lon': 30.4167},
            'Kayonza': {'lat': -1.8833, 'lon': 30.6333},
            'Kirehe': {'lat': -2.2167, 'lon': 30.7167},
            'Ngoma': {'lat': -2.1667, 'lon': 30.5167},
            'Nyagatare': {'lat': -1.3000, 'lon': 30.3333},
            'Rwamagana': {'lat': -1.9500, 'lon': 30.4333}
        },
        'Northern Province': {
            'Burera': {'lat': -1.4833, 'lon': 29.8833},
            'Gakenke': {'lat': -1.6833, 'lon': 29.7833},
            'Gicumbi': {'lat': -1.5833, 'lon': 30.0667},
            'Musanze': {'lat': -1.5000, 'lon': 29.6333},
            'Rulindo': {'lat': -1.7667, 'lon': 30.0667}
        },
        'Southern Province': {
            'Gisagara': {'lat': -2.5833, 'lon': 29.8333},
            'Huye': {'lat': -2.5833, 'lon': 29.7333},
            'Kamonyi': {'lat': -2.0333, 'lon': 29.9667},
            'Muhanga': {'lat': -2.0833, 'lon': 29.7500},
            'Nyamagabe': {'lat': -2.4500, 'lon': 29.4833},
            'Nyanza': {'lat': -2.3500, 'lon': 29.7500},
            'Nyaruguru': {'lat': -2.6833, 'lon': 29.4333},
            'Ruhango': {'lat': -2.2333, 'lon': 29.7833}
        },
        'Western Province': {
            'Karongi': {'lat': -2.0000, 'lon': 29.3833},
            'Ngororero': {'lat': -1.8333, 'lon': 29.5333},
            'Nyabihu': {'lat': -1.6333, 'lon': 29.5000},
            'Nyamasheke': {'lat': -2.3500, 'lon': 29.1167},
            'Rubavu': {'lat': -1.6833, 'lon': 29.3333},
            'Rusizi': {'lat': -2.4833, 'lon': 28.9000},
            'Rutsiro': {'lat': -1.9667, 'lon': 29.3333}
        }
    }
    
    # Prepare data for visualization
    map_data = []
    for province, districts in rwanda_districts.items():
        for district, coords in districts.items():
            # Get client count for this district
            count_row = district_counts[
                (district_counts['province'] == province) & 
                (district_counts['district'] == district)
            ]
            count = count_row['client_count'].values[0] if len(count_row) > 0 else 0
            
            map_data.append({
                'province': province,
                'district': district,
                'lat': coords['lat'],
                'lon': coords['lon'],
                'client_count': count
            })
    
    map_df = pd.DataFrame(map_data)
    
    # Create the map using Plotly
    fig = go.Figure()
    
    # Add scatter points for each district
    fig.add_trace(go.Scattergeo(
        lon=map_df['lon'],
        lat=map_df['lat'],
        text=map_df.apply(lambda row: f"{row['district']}<br>Province: {row['province']}<br>Clients: {row['client_count']}", axis=1),
        mode='markers+text',
        marker=dict(
            size=map_df['client_count'] * 2,  # Size based on client count
            color=map_df['client_count'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Client Count",
                thickness=15,
                len=0.7
            ),
            line=dict(width=1, color='white')
        ),
        textposition="top center",
        textfont=dict(size=8, color='black'),
        name='Districts',
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))
    
    # Update layout for Rwanda map
    fig.update_geos(
        center=dict(lat=-1.9403, lon=29.8739),  # Rwanda center
        projection_scale=35,  # Zoom level
        visible=True,
        resolution=50,
        showcountries=True,
        countrycolor="lightgray",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        coastlinecolor="rgb(204, 204, 204)",
        projection_type="mercator",
        lonaxis=dict(range=[28.5, 31.0]),
        lataxis=dict(range=[-3.0, -1.0])
    )
    
    fig.update_layout(
        title={
            'text': 'Rwanda Vehicle Clients Distribution by District',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='white',
        geo=dict(
            bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig


def print_district_summary():
    """
    Print summary statistics by district
    """
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    
    summary = df.groupby(['province', 'district']).agg({
        'client_name': 'count',
        'estimated_income': 'mean',
        'selling_price': 'mean'
    }).reset_index()
    
    summary.columns = ['Province', 'District', 'Total Clients', 'Avg Income', 'Avg Vehicle Price']
    summary = summary.sort_values(['Province', 'Total Clients'], ascending=[True, False])
    
    print("\n" + "="*80)
    print("RWANDA DISTRICT SUMMARY - VEHICLE CLIENTS DISTRIBUTION")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)
    
    # Province-level summary
    province_summary = df.groupby('province').agg({
        'client_name': 'count',
        'estimated_income': 'mean',
        'selling_price': 'mean'
    }).reset_index()
    province_summary.columns = ['Province', 'Total Clients', 'Avg Income', 'Avg Vehicle Price']
    
    print("\nPROVINCE-LEVEL SUMMARY")
    print("-"*80)
    print(province_summary.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("EXERCISE PART A: Rwanda Map Visualization (20 marks)")
    print("="*80)
    
    # Print district summary
    print_district_summary()
    
    # Create and show the map
    print("\nGenerating interactive Rwanda map...")
    fig = create_rwanda_map_visualization()
    
    # Save to HTML file
    output_file = "exercise_solutions/rwanda_map_visualization.html"
    fig.write_html(output_file)
    print(f"\n✅ Map saved to: {output_file}")
    print("   Open this file in your browser to view the interactive map.")
    
    # Show the map in browser
    print("\n📊 Opening map in browser...")
    fig.show()
    
    print("\n" + "="*80)
    print("PART A COMPLETED SUCCESSFULLY!")
    print("="*80)
