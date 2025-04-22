import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Projet DA : températures terrestres")
st.header("Quelques exemples de visuels")
# --- Premier graphique (inchangé) ---
# Titre du graphique et des axes
title_text_mer = "Anomalies annuelles de la température de la surface des océans"
xaxis_title_mer = "Année"
yaxis_title_mer = "Anomalie de température (°C)"

# Lecture des données depuis une URL
df_mer = pd.read_csv("https://ourworldindata.org/grapher/sea-surface-temperature-anomaly.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

# Filtrage des données pour obtenir les données globales, de l'hémisphère nord et de l'hémisphère sud
world_data_mer = df_mer[df_mer['Entity'] == 'World']
northern_hemisphere_data_mer = df_mer[df_mer['Entity'] == 'Northern Hemisphere']
southern_hemisphere_data_mer = df_mer[df_mer['Entity'] == 'Southern Hemisphere']

# Création de la figure
fig_mer = go.Figure()

# Ajout des traces pour chaque ensemble de données (Global, Hémisphère Nord, Hémisphère Sud)
fig_mer.add_trace(go.Scatter(x=world_data_mer['Year'], y=world_data_mer['sea_temperature_anomaly_annual'],
                            mode='lines', name='Moyenne Globale', line=dict(color='royalblue', width=2)))
fig_mer.add_trace(go.Scatter(x=northern_hemisphere_data_mer['Year'], y=northern_hemisphere_data_mer['sea_temperature_anomaly_annual'],
                            mode='lines', name='Hémisphère Nord', line=dict(color='firebrick', width=2, dash='dash')))
fig_mer.add_trace(go.Scatter(x=southern_hemisphere_data_mer['Year'], y=southern_hemisphere_data_mer['sea_temperature_anomaly_annual'],
                            mode='lines', name='Hémisphère Sud', line=dict(color='forestgreen', width=2, dash='dot')))

# Mise à jour de la mise en page du graphique
fig_mer.update_layout(
    title=title_text_mer,
    xaxis_title=xaxis_title_mer,
    yaxis_title=yaxis_title_mer,
    hovermode="x unified",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    font=dict(size=12),
    plot_bgcolor='white',
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date",
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgray'
    )
)

# Ajout d'un sélecteur de plage pour une exploration interactive des données
fig_mer.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1 an", step="year", stepmode="backward"),
            dict(count=5, label="5 ans", step="year", stepmode="backward"),
            dict(count=10, label="10 ans", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# Affichage du premier graphique avec Streamlit
st.plotly_chart(fig_mer)


st.caption("Source : https://ourworldindata.org/")
           
st.markdown("""
Le graphique donne les anomalies annuelles de la température de la surface de la mer par rapport à la période préindustrielle pour le Monde, l'Hémisphère Nord, <br>
et l'Hémisphère Sud). <br>
Les données représentent la différence entre la température moyenne de la surface de la mer et la moyenne de 1861 à 1890, en degrés Celsius, mesuré à une profondeur de 20 centimètres pour l'hémisphère Nord, l'hémisphère Sud et le monde. <br>
Le graphique montre que les océans de l'hémisphère Nord se réchauffent plus vite que ceux de l'hémisphère Sud depuis 2013.
""", unsafe_allow_html=True)

# --- Deuxième graphique (ressemblant au premier) ---
# Chargement des 3 datasets
def load_and_prepare_terre(url, region):
    df = pd.read_csv(url, header=1)
    df = df.iloc[1:]
    df = df.replace('***', pd.NA)
    df['Region'] = region
    df = df[['Year', 'J-D', 'Region']]
    return df

df_global_terre = load_and_prepare_terre("https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv", "Monde")
df_north_terre = load_and_prepare_terre("https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv", "Hémisphère Nord")
df_south_terre = load_and_prepare_terre("https://data.giss.nasa.gov/gistemp/tabledata_v4/SH.Ts+dSST.csv", "Hémisphère Sud")

# Fusion et nettoyage
df_final_terre = pd.concat([df_global_terre, df_north_terre, df_south_terre], ignore_index=True)
df_final_terre['J-D'] = pd.to_numeric(df_final_terre['J-D'], errors='coerce')
df_final_terre['Year'] = pd.to_numeric(df_final_terre['Year'], errors='coerce')

# Titre du graphique et des axes (similaires au premier)
title_text_terre = "Anomalies annuelles de la température : Terres + Océans"
xaxis_title_terre = "Année"
yaxis_title_terre = "Anomalie de température (°C)"

# Création de la figure
fig_terre = go.Figure()

# Ajout des traces pour chaque ensemble de données (Monde, Hémisphère Nord, Hémisphère Sud)
world_data_terre = df_final_terre[df_final_terre['Region'] == 'Monde']
northern_hemisphere_data_terre = df_final_terre[df_final_terre['Region'] == 'Hémisphère Nord']
southern_hemisphere_data_terre = df_final_terre[df_final_terre['Region'] == 'Hémisphère Sud']

fig_terre.add_trace(go.Scatter(x=world_data_terre['Year'], y=world_data_terre['J-D'],
                            mode='lines', name='Moyenne Globale', line=dict(color='royalblue', width=2)))
fig_terre.add_trace(go.Scatter(x=northern_hemisphere_data_terre['Year'], y=northern_hemisphere_data_terre['J-D'],
                            mode='lines', name='Hémisphère Nord', line=dict(color='firebrick', width=2, dash='dash')))
fig_terre.add_trace(go.Scatter(x=southern_hemisphere_data_terre['Year'], y=southern_hemisphere_data_terre['J-D'],
                            mode='lines', name='Hémisphère Sud', line=dict(color='forestgreen', width=2, dash='dot')))

# Mise à jour de la mise en page du graphique
fig_terre.update_layout(
    title=title_text_terre,
    xaxis_title=xaxis_title_terre,
    yaxis_title=yaxis_title_terre,
    hovermode="x unified",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    font=dict(size=12),
    plot_bgcolor='white',
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date",
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgray'
    )
)

# Ajout d'un sélecteur de plage pour une exploration interactive des données (copie du premier graphique)
fig_terre.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1 an", step="year", stepmode="backward"),
            dict(count=5, label="5 ans", step="year", stepmode="backward"),
            dict(count=10, label="10 ans", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# Affichage du deuxième graphique avec Streamlit
st.plotly_chart(fig_terre)

st.caption("Source : https://data.giss.nasa.gov/gistemp/")

st.markdown("""
Le dataset utilisé pour ce graphique est la combinaison de 3 datasets : un par région observée (Monde, Hémisphère Nord, Hémisphère Sud). <br>
Le graphique présente les anomalies combinées de température de l'air à la surface terrestre et de l'eau à la surface de la mer, <br>
c'est-à-dire les écarts par rapport aux moyennes correspondantes de 1951 à 1980. <br>
Le graphique confirme un réchauffement plus rapide de l'hémisphère Nord depuis 2013, ce réchauffement est accentué par les températures relevées à la surface terrestre. <br>
<br>
""", unsafe_allow_html=True)

# --- Troisième graphique (titre à gauche) ---
# Fonction pour créer un graphique des anomalies de température
def create_temperature_graph(filepath):
    try:
        # Lecture du fichier CSV avec une option pour le User-Agent
        df = pd.read_csv(filepath, storage_options={'User-Agent': 'My User Agent 1.0'})
        # Conversion de la colonne 'Year' en objets datetime
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')

        # Extraction des pays uniques et filtrage des mots clés
        countries = df['Entity'].unique()
        filtered_countries = [
            country for country in countries
            if not any(keyword in country.lower() for keyword in ["ocean", "seas", "sea", "mediterranean", "world"])
        ]

        # Sélection des pays pour le graphique (y compris le monde et les pays contenant "NIAID")
        selected_countries = ['World']
        selected_countries.extend([country for country in filtered_countries if "NIAID" in country])

        # Création du graphique
        fig = go.Figure()
        for country in selected_countries:
            country_data = df[df['Entity'] == country]
            if not country_data.empty:
                # Suppression de "(NIAID)" du nom de la légende
                legend_name = country.replace(" (NIAID)", "")
                # Ajout des données du pays au graphique
                fig.add_trace(go.Scatter(x=country_data['Year'], y=country_data['temperature_anomaly'], name=legend_name, mode='lines+markers', marker=dict(size=4)))

        # Configuration de la mise en page du graphique
        fig.update_layout(
            title="Anomalies de température annuelles<br><sup>Différence entre la température moyenne de surface de la terre d'une année et la moyenne de 1991 à 2020, en degrés Celsius</sup>",
            title_x=0.0,  # Modification ici pour aligner le titre à gauche
            title_y=0.98,
            title_font=dict(size=16),
            xaxis_title="Année",
            yaxis_title="Anomalies de température (°C)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            width=1200,
            height=700,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            ),
            hovermode="x unified",
            annotations=[
                dict(
                    x=df[df['Entity'] == 'World']['Year'].max(),
                    y=df[df['Entity'] == 'World']['temperature_anomaly'].max(),
                    xref="x",
                    yref="y",
                    text="Maximum mondial",
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-40
                )
            ]
        )
        # Affichage du graphique avec Streamlit
        st.plotly_chart(fig)

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

# Chemin du fichier CSV
filepath_troisieme = "https://ourworldindata.org/grapher/annual-temperature-anomalies.csv?v=1&csvType=full&useColumnShortNames=true"
# Appel de la fonction pour créer le graphique
create_temperature_graph(filepath_troisieme)

# --- Quatrième graphique (carte mondiale animée) ---
# Lecture des données depuis une URL.
df_carte = pd.read_csv(
    "https://ourworldindata.org/grapher/annual-temperature-anomalies.csv?v=1&csvType=full&useColumnShortNames=true",
    storage_options={'User-Agent': 'Our World In Data data fetch/1.0'}
)

# Convertir la colonne 'Year' en numérique
df_carte['Year'] = pd.to_numeric(df_carte['Year'], errors='coerce')

# Création de la carte mondiale animée avec Plotly Express
fig_carte = px.choropleth(
    df_carte,
    locations="Code",
    color="temperature_anomaly",
    hover_name="Entity",
    animation_frame="Year",
    range_color=[-2, 2],
    labels={'temperature_anomaly': 'Anomalie de température (°C)', 'Year': 'Année'},
    color_continuous_scale="RdBu_r",
    projection="natural earth"
)

# Mise à jour de la mise en page
fig_carte.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="RebeccaPurple",
        showland=True,
        landcolor="LightGray",
        showocean=True,
        oceancolor="LightBlue"
    ),
    title=dict(
        text="<b>Anomalies de Température Annuelles</b><br><sup>Différence par rapport à la moyenne 1991-2020</sup>",
        x=0.45,
        y=0.95,
        xanchor='center',
        font=dict(size=24, family="Arial")
    ),
    coloraxis_colorbar=dict(
        title=dict(
            text="Anomalie (°C)",
            font=dict(size=16)
        ),
        thicknessmode="pixels",
        thickness=20,
        len=0.7,
        yanchor="top",
        y=0.9,
        tickfont=dict(size=14)
    ),
    margin=dict(r=0, t=80, l=0, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Ajout de bordures aux pays sur la carte.
fig_carte.update_traces(marker_line_width=0.5, marker_line_color="DarkSlateGrey")

# Affichage du graphique avec Streamlit
st.plotly_chart(fig_carte)

st.caption("Source : https://ourworldindata.org/")

st.markdown("""
Les deux graphiques utilisent la même source de données. <br>
Ils montrent la différence entre la température moyenne de surface d'une année et la moyenne de 1991 à 2020, en degrés Celsius. <br>
<br>
""", unsafe_allow_html=True)

# Cinquième graphique

liste_entites = [
    "Europe",
    "North America",
    "Asia",
    "China",
    "Russia"
]

df = pd.read_csv(
    "https://ourworldindata.org/grapher/annual-temperature-anomalies.csv?v=1&csvType=full&useColumnShortNames=true",
    storage_options={'User-Agent': 'Our World In Data data fetch/1.0'}
)

# Filtrer les entités et les années
df_grouped = df[df['Entity'].isin(liste_entites)].copy()
df_grouped = df_grouped[df_grouped['Year'] >= 1940]

plt.figure(figsize=(8, 5))

# Create a color palette using tab10
palette = sns.color_palette("tab10", n_colors=len(liste_entites))

# Fossil fuels and Land Use on the same plot
sns.lineplot(
    data=df_grouped,
    x='Year',
    y='temperature_response_ghg_fossil',
    linewidth=2,
    label='Fossil Fuels',
    color='yellow'  # Using a single color for Fossil Fuels
)
sns.lineplot(
    data=df_grouped,
    x='Year',
    y='temperature_response_ghg_land',
    linewidth=2,
    linestyle='--',
    label='Land Use',
    color='grey' # Using a single color for Land Use
)

for i, entity in enumerate(liste_entites):
    sns.lineplot(
        data=df_grouped[df_grouped['Entity'] == entity],
        x='Year',
        y='temperature_response_ghg_fossil',
        color=palette[i],
        linewidth=2,
        label=entity # Adding the name of the entity as the label
    )
    sns.lineplot(
        data=df_grouped[df_grouped['Entity'] == entity],
        x='Year',
        y='temperature_response_ghg_land',
        color=palette[i],
        linewidth=2,
        linestyle='--',
    )

plt.title("Contribution au réchauffement climatique par région (1900-2022)", fontsize=10)
plt.xlabel("Année", fontsize=12)
plt.ylabel("Réchauffement estimé (°C)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', labelsize=10)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='right', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.10), fontsize=6)

plt.tight_layout()
plt.show()

