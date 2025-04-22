import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

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


st.title("Analyse des émissions de CO2 mondiales")
st.write("Visualisation des données sur les émissions de CO2 provenant de Our World in Data.")

# URL brute du fichier CSV dans GitHub
url = 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv'
 
# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(url)
 
# Filtrer les données pour ne garder que les colonnes nécessaires et supprimer les lignes avec des valeurs manquantes
df_filtered = df[['country', 'year', 'co2_per_capita']].dropna(subset=['co2_per_capita'])
 
# Filtrer pour inclure uniquement les données à partir de l'année 1900
df_filtered = df_filtered[df_filtered['year'] >= 1900]
 
# Trier les données par année pour garantir l'ordre croissant
df_filtered = df_filtered.sort_values(by='year')
 
# Créer le graphique interactif avec Plotly Express
fig = px.choropleth(df_filtered,
                    locations="country",
                    locationmode="country names",
                    color="co2_per_capita",
                    hover_name="country",
                    animation_frame="year",
                    range_color=[0, 20],  # Ajuster la plage de couleurs selon vos besoins
                    color_continuous_scale="Viridis",
                    projection="natural earth",
                    title="Évolution du CO2 per capita par pays")
 
# Afficher le graphique
st.plotly_chart(fig)

st.caption("Source : https://ourworldindata.org/")
           
st.markdown("""
Ce graphique montre l’évolution des émissions de CO2 par pays par habitant au cours du temps. <br>
On voit que les émissions ont beaucoup évolué au cours du temps mais que la majorité concerne l’hémisphère nord. <br>
Les pays occidentaux ont longtemps été en tête puis une augmentation se voit en Asie et dernièrement c’est l’Arabie <br>
Saoudite qui à la plus forte émission par habitant. <br>
""", unsafe_allow_html=True)

# Neuvième graphique
### Émission mondiale de CO2 par habitant (Carte choroplèthe) ###
st.header("Émissions mondiales de CO2 par habitant au fil du temps")
st.write("Cette carte montre l'évolution des émissions annuelles de CO2 par habitant par pays depuis 1950.")

if not world_co2_data.empty:
    fig_co2_per_capita_map = px.choropleth(world_co2_data, locations='iso_code', color='co2_per_capita',
                                           animation_frame='year',
                                           hover_name='country',
                                           color_continuous_scale='pubu',
                                           projection='natural earth',
                                           title='Émission mondiale de CO2 par habitant (en tonnes)',
                                           range_color=(0, world_co2_data['co2_per_capita'].quantile(0.95))) # Utiliser un quantile

    fig_co2_per_capita_map.update_layout(geo=dict(showframe=False, showcoastlines=False),
                                        title=dict(x=0.5, font=dict(size=20)),
                                        height=600)
    fig_co2_per_capita_map.update_coloraxes(colorbar=dict(x=1, y=0.5, len=1, tickfont=dict(size=10), title="CO2 (Tonnes)"))

    st.plotly_chart(fig_co2_per_capita_map, use_container_width=True)
else:
     st.warning("Aucune donnée disponible pour afficher la carte des émissions de CO2 par habitant.")

st.caption("Source : https://ourworldindata.org/")

st.markdown("""
En observant la carte interactive des émissions de CO2 par habitant de 1950 à 2023, nous pouvons remarquer rapidement que les pays ayant les émissions <br>
de CO2 par habitant les plus élevées sont les États-Unis, le Canada, l'Australie, mais aussi - il vaut mieux zoomer pour les voir - les pays du golf <br>
persique (Qatar, Koweït, Emirats Arabes Unis, Oman et Arabie saoudite). Cependant, si l'on compare les émissions de CO2 par habitant entre les pays <br>
à différents moments du temps, il est également possible de voir des changements importants. Par exemple, nous pouvons constater que certains pays <br>
ont réduit leurs émissions de CO2 par habitant au fil du temps (ex: UK, Allemagne, Belgique, France) . Cette baisse peut être due à des efforts pour <br>
favoriser la production d'énergie renouvelable et réduire la consommation d'énergie fossile. <br>
<br>
En revanche, certaines régions ont connu une augmentation rapide de leurs émissions de CO2 par habitant au cours des dernières décennies. <br>
C'est le cas notamment de l'Asie, où plusieurs économies en développement ont connu une croissance rapide de leur émission de CO2 par habitant <br>
depuis les années 1980. Cette évolution reflète une industrialisation rapide et une croissance économique rapide dans la région. <br>
<br>
Cependant, si l'on compare les émissions de CO2 par habitant des différentes régions du monde, on peut noter que l'Amérique du Nord, l'Europe et l'Asie <br>
ont tendance à avoir des émissions de CO2 par habitant plus élevées que d'autres régions, du moins jusqu'à récemment. Le niveau de développement <br>
économique, les modes de consommation, les habitudes de transport, la composition de l'énergie et les politiques environnementales sont <br>
des facteurs qui peuvent expliquer ces différences régionales. <br>
""", unsafe_allow_html=True)


# sixième graphique

# Configuration de la page Streamlit
# st.set_page_config(layout="wide", page_title="Analyse des émissions de CO2 mondiales")


# URL brute du fichier CSV dans GitHub
url = 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv'

# Utilisation de st.cache_data pour mettre en cache le chargement et le prétraitement des données
@st.cache_data
def load_and_preprocess_data(url):
    """Charge les données et effectue un prétraitement initial."""
    df = pd.read_csv(url)

    # Colonnes à garder dans le fichier
    colonnes_a_garder = ['country', 'year', 'iso_code', 'population', 'gdp',
                         'co2', 'co2_per_capita',
                         'methane', 'nitrous_oxide',
                         'cumulative_gas_co2', 'cumulative_oil_co2','cumulative_flaring_co2','cumulative_coal_co2','cumulative_other_co2',
                         'temperature_change_from_ch4', 'temperature_change_from_co2',
                         'temperature_change_from_ghg', 'temperature_change_from_n2o']
    df = df[colonnes_a_garder].copy()

    # Les pays (lignes) à retirer du fichier (Pays qui n'en sont pas en fait !)
    A_retirer = ['Africa (GCP)', 'Asia (GCP)',
           'Asia (excl. China and India)', 'Central America (GCP)',
           'Europe (GCP)', 'Europe (excl. EU-27)', 'Europe (excl. EU-28)',
           'European Union (27)', 'European Union (28)',
           'High-income countries',
           'International aviation', 'International shipping',
           'International transport','Kuwaiti Oil Fires', 'Kuwaiti Oil Fires (GCP)',
           'Least developed countries (Jones et al. 2023)',
           'Low-income countries', 'Lower-middle-income countries',
           'Middle East (GCP)', 'Non-OECD (GCP)',
           'North America (GCP)', 'North America (excl. USA)', 'OECD (GCP)',
           'OECD (Jones et al.)', 'Oceania (GCP)',
           'Ryukyu Islands (GCP)',
           'South America (GCP)',
           'Upper-middle-income countries']
    df = df.loc[~df['country'].isin(A_retirer)]

    return df

# Chargement des données
df = load_and_preprocess_data(url)

# Filtrer les données pour les cartes choroplèthes (souvent moins de données anciennes)
world_co2_data = df.loc[df['year'] >= 1950].copy() # Utiliser .copy() pour éviter SettingWithCopyWarning

### Émission mondiale de CO2 (Carte choroplèthe) ###
st.header("Émissions mondiales de CO2 par pays au fil du temps")
st.write("Cette carte montre l'évolution des émissions annuelles de CO2 par pays depuis 1950.")

if not world_co2_data.empty:
    fig_co2_map = px.choropleth(world_co2_data, locations='iso_code', color='co2',
                                animation_frame='year',
                                hover_name='country',
                                color_continuous_scale='pubu',
                                projection='natural earth',
                                title='Émissions mondiales de CO2 (en Mt)',
                                range_color=(0, world_co2_data['co2'].quantile(0.95))) # Utiliser un quantile pour une meilleure visualisation

    fig_co2_map.update_layout(geo=dict(showframe=False, showcoastlines=False),
                              title=dict(x=0.5, font=dict(size=20)),
                              height=600) # Ajuster la hauteur pour Streamlit
    fig_co2_map.update_coloraxes(colorbar=dict(x=1, y=0.5, len=1, tickfont=dict(size=10), title="CO2 (Mt)"))


  
    st.plotly_chart(fig_co2_map, use_container_width=True)
else:
    st.warning("Aucune donnée disponible pour afficher la carte des émissions de CO2.")

st.caption("Source : https://ourworldindata.org/")

st.markdown("""
En observant la carte interactive, nous pouvons remarquer que les émissions de CO2 ont augmenté de manière constante depuis 1950. <br>
Les pays ayant les émissions les plus élevées sont les États-Unis, la Chine, l'Inde, la Russie et le Japon. <br>
Ces pays ont des émissions de CO2 plusieurs fois plus élevées que d'autres pays de la carte, ce qui indique que la majeure partie de l'émission de CO2 <br>
dans le monde provient de ces pays. <br>
Nous pouvons également observer que les émissions de CO2 ont connu une augmentation rapide à partir du milieu des années 1950 et <br>
ont augmenté de manière plus prononcée à partir des années 2000. Cette augmentation rapide peut être liée à l'augmentation de la population mondiale <br>
et à l'augmentation de la demande d'énergie pour les transports, la production manufacturière et l'électricité. <br>
Cependant, il existe également des différences régionales dans les émissions de CO2. <br>
L'Amérique du Nord, l'Europe et l'Asie ont des émissions plus élevées que l'Afrique et l'Amérique latine. <br>
Cela peut être dû à des facteurs tels que les niveaux de développement économique, les politiques environnementales et les sources d'énergie employées. <br>
""", unsafe_allow_html=True)


# septième graphique
### Émissions de CO2 par continent (Bar plot) ###
st.header("Émissions totales de CO2 par continent (1950-2023)")
st.write("Ce graphique montre la somme des émissions de CO2 par continent sur la période sélectionnée pour la carte.")

a_garder_continents = ['Europe', 'Asia', 'Africa', 'North America', 'Oceania', 'South America']
# Filtrer les données pour n'inclure que les continents et les années >= 1950
df_continent_filtered = world_co2_data.loc[world_co2_data['country'].isin(a_garder_continents)].copy()

if not df_continent_filtered.empty:
    df_continent_sum = df_continent_filtered.groupby('country')['co2'].sum().reset_index()

    # Créer explicitement la figure et les axes Matplotlib
    fig_continent_bar, ax_continent_bar = plt.subplots(figsize=(10, 6)) # Ajuster la taille

    sns.barplot(x='country', y='co2', data=df_continent_sum, ax=ax_continent_bar)
    ax_continent_bar.set_title('Émissions totales de CO₂ (Mt) par continent (1950-2023)')
    ax_continent_bar.set_xlabel('Continent')
    ax_continent_bar.set_ylabel('Émissions de CO₂ (Mt)')
    plt.xticks(rotation=45, ha='right') # Incliner les étiquettes de l'axe x
    plt.tight_layout() # Ajuster la mise en page

    st.pyplot(fig_continent_bar)
else:
     st.warning("Aucune donnée disponible pour afficher le graphique à barres par continent.")

st.caption("Source : https://ourworldindata.org/")

st.markdown("""
Ce graphique nous montre bien que l'Asie, l'Europe et l'Asie sont les continents émettant le plus d'émissions de CO2 de 1950 à 2023, <br>
au moins 5 à 6 fois plus que l'Afrique, l'Océanie et l'Amérique du Sud. <br>
""", unsafe_allow_html=True)

# Huitième graphique
### Émissions de CO2 par continent au fil du temps (Line plot) ###
st.header("Émissions de CO2 par continent au fil du temps")
st.write("Ce graphique montre l'évolution annuelle des émissions de CO2 pour chaque continent depuis 1750.")

# Utiliser le DataFrame original pour avoir les données depuis 1750 si disponibles
df_continent_all_years = df.loc[df['country'].isin(a_garder_continents)].copy()

if not df_continent_all_years.empty:
    # Créer explicitement la figure et les axes Matplotlib
    fig_continent_line, ax_continent_line = plt.subplots(figsize=(12, 7)) # Ajuster la taille

    sns.lineplot(data=df_continent_all_years, x='year', y='co2', hue='country', ax=ax_continent_line)
    ax_continent_line.set_title('Émissions de CO₂ (Mt) par continent et par année (1750-2023)')
    ax_continent_line.set_xlabel('Année')
    ax_continent_line.set_ylabel('Émissions de CO₂ (Mt)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    st.pyplot(fig_continent_line)
else:
    st.warning("Aucune donnée disponible pour afficher le graphique linéaire par continent.")

st.caption("Source : https://ourworldindata.org/")

st.markdown("""
Ce graphique nous montre bien que l'augmentation des émissions de CO2 est très importante depuis les années 1950, elle est exponentielle pour l'Asie. <br>
Les émissions de C02 concernant l'Europe et l'Amérique du Nord ont évolué un peu avant les années 1900 (révolution industrielle). Mais on voit qu'elles <br>
sont en déclin depuis les années 2000 (attention: appartenance réelle des émissions pour l'Asie ?) <br>
""", unsafe_allow_html=True)

# dixième graphique
### Émissions de CO2 par habitant par continent (Bar plot) ###
st.header("Émissions totales de CO2 par habitant par continent (1950-2023)")
st.write("Ce graphique montre la somme des émissions de CO2 par habitant par continent sur la période sélectionnée pour la carte.")

if not df_continent_filtered.empty:
    df_co2_per_capita_sum = df_continent_filtered.groupby('country')['co2_per_capita'].sum().reset_index()

    # Créer explicitement la figure et les axes Matplotlib
    fig_co2_per_capita_bar, ax_co2_per_capita_bar = plt.subplots(figsize=(10, 6)) # Ajuster la taille

    sns.barplot(x='country', y='co2_per_capita', data=df_co2_per_capita_sum, ax=ax_co2_per_capita_bar)
    ax_co2_per_capita_bar.set_title('Émissions totales de CO₂ par habitant par continent (1950-2023)')
    ax_co2_per_capita_bar.set_xlabel('Continent')
    ax_co2_per_capita_bar.set_ylabel('Émissions de CO₂ par habitant (Tonnes)')
    plt.xticks(rotation=45, ha='right') # Incliner les étiquettes de l'axe x
    plt.tight_layout()

    st.pyplot(fig_co2_per_capita_bar)
else:
    st.warning("Aucune donnée disponible pour afficher le graphique à barres des émissions par habitant par continent.")

st.caption("Source : https://ourworldindata.org/")

st.markdown("""
Ce graphique nous montre que les Etats-Unis ont les émissions de CO2 par habitant les plus élevées, suivi par l'Océanie et l'Europe entre 1950 et 2023. <br>
L'Asie a des émissions de CO2 par habitant 5 fois moins importantes que les Etats-Unis. <br>
""", unsafe_allow_html=True)

# onzième graphique
### Émissions de CO2 par habitant par continent au fil du temps (Line plot) ###
st.header("Émissions de CO2 par habitant par continent au fil du temps")
st.write("Ce graphique montre l'évolution annuelle des émissions de CO2 par habitant pour chaque continent depuis 1750.")

if not df_continent_all_years.empty:
    # Créer explicitement la figure et les axes Matplotlib
    fig_co2_per_capita_line, ax_co2_per_capita_line = plt.subplots(figsize=(12, 7)) # Ajuster la taille

    sns.lineplot(data=df_continent_all_years, x='year', y='co2_per_capita', hue='country', ax=ax_co2_per_capita_line)
    ax_co2_per_capita_line.set_title('Émissions de CO₂ par habitant par continent et par année (1750-2023)')
    ax_co2_per_capita_line.set_xlabel('Année')
    ax_co2_per_capita_line.set_ylabel('Émissions de CO₂ par habitant (Tonnes)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    st.pyplot(fig_co2_per_capita_line)
else:
     st.warning("Aucune donnée disponible pour afficher le graphique linéaire des émissions par habitant par continent.")

st.caption("Source : https://ourworldindata.org/")

st.markdown("""
Ce graphique montre bien qu'à partir des années 1850 (fin de l'ère de la révolution industrielle) les émissions de CO2 par habitant ont commencé à exploser aux Etats-Unis <br>
Elles ont atteint leur seuil maximal avant les années 2000 et commence à diminuer depuis les années 2000. La situation est similaire pour l'Europe <br>
et l'Océanie bien que les taux d'émissions de CO2 par habitant sont moins élevés que pour les Etats-Unis. <br>
En revanche, concernant l'Asie on remarque une augmentation des émissions de CO2 par habitant à partir des années 1950, cela signifie que l'Asie est <br>
actuellement en pleine croissance industrielle. <br>
""", unsafe_allow_html=True)

# douzième graphique
### Top 15 pays émetteurs de CO2 (Bar plot) ###
st.header("Top 15 des pays émetteurs de CO2")
st.write("Classement des 15 pays ayant les émissions annuelles moyennes de CO2 les plus élevées sur la période 1950-2023.")

if not world_co2_data.empty:
    # Choix des top 15 pays les plus polluants (basé sur la moyenne sur la période filtrée)
    df_moy = world_co2_data.groupby('country')['co2'].mean().reset_index()

    regions = ['World','Asia','Europe','North America','Africa','South America','Oceania']
    df_countries_only = df_moy[~df_moy['country'].isin(regions)].copy()

    # On enlève les NaN
    df_countries_only = df_countries_only.dropna(subset=['co2'])

    # On trie et garde les 15 premiers
    top_15_mean_emitters = df_countries_only.sort_values(by='co2', ascending=False).head(15).copy()

    # Création du graphique interactif
    fig_top15_bar = px.bar(top_15_mean_emitters.sort_values(by='co2', ascending=True), # Pour ordre croissant en vertical
       x='co2',
       y='country',
       orientation='h',
       title='Top 15 des pays émetteurs de CO₂ (Moyenne 1950-2023)',
       labels={'co2': 'Émissions de CO₂ (Mt)', 'country': 'Pays'},
       color='co2',
       color_continuous_scale='reds')

    # Mise à jour de la mise en page
    fig_top15_bar.update_layout(template='plotly_white',
                       height=600,
                       margin=dict(l=150, r=50, t=50, b=50))

    st.plotly_chart(fig_top15_bar, use_container_width=True)
else:
    st.warning("Aucune donnée disponible pour calculer et afficher le top 15 des pays émetteurs.")

st.caption("Source : https://ourworldindata.org/")

st.markdown("""
On voit bien que les US et la Chine sont de loin les deux plus gros émetteurs de CO2 sur ces dernières années. <br>
""", unsafe_allow_html=True)

# treizième graphique

### Boxplots sur la liste des top 15 émetteurs de CO2 ###
st.header("Distribution des émissions de CO2 pour les Top 15 pays émetteurs")
st.write("Ces boxplots montrent la distribution annuelle des émissions de CO2 pour les 15 pays les plus émetteurs (basé sur la moyenne 1950-2023), depuis 1850.")

# Liste des top 15 pays basée sur la moyenne 1950-2023 calculée ci-dessus
if 'top_15_mean_emitters' in locals() and not top_15_mean_emitters.empty:
    top15_countries_list = top_15_mean_emitters['country'].tolist()
    country_top_boxplot_data = df.loc[df['country'].isin(top15_countries_list)].copy()
    country_top_boxplot_data = country_top_boxplot_data.loc[country_top_boxplot_data['year'] >= 1850].copy()

    if not country_top_boxplot_data.empty:
        fig_boxplot = px.box(country_top_boxplot_data, x="country", y="co2", hover_data=["year"],
                             title="Boxplots des émissions de CO2 depuis 1850 - Top 15 pays émetteurs")
        fig_boxplot.update_layout(xaxis_title="Pays",
                                  yaxis_title="Émissions CO2 (Mt)",
                                  height=600,
                                  margin=dict(l=50, r=50, t=50, b=50)) # Ajuster les marges si nécessaire

        st.plotly_chart(fig_boxplot, use_container_width=True)
    else:
         st.warning("Aucune donnée suffisante depuis 1850 pour les top 15 pays afin d'afficher les boxplots.")
else:
    st.warning("Le calcul du top 15 des pays n'a pas abouti. Impossible d'afficher les boxplots.")

st.caption("Source : https://ourworldindata.org/")

st.markdown("""
Les boxplots nous permettent ici de juger la cohérence de nos données. <br>
Ils mettent en évidence que les Etats-Unis sont de gros emetteurs, et depuis longtemps, tandis qu'on voit que les valeurs pour la Chine ont beaucoup <br>
evolué. <br>
""", unsafe_allow_html=True)

# quatorzième graphique
### Émissions de CO2 des top 15 émetteurs au fil du temps (Line plot) ###
st.header("Émissions de CO2 des Top 15 pays émetteurs au fil du temps")
st.write("Ce graphique montre l'évolution annuelle des émissions de CO2 pour les 15 pays les plus émetteurs (basé sur la moyenne 1950-2023), depuis 1850.")

# Utiliser le même DataFrame filtré pour les boxplots
if 'country_top_boxplot_data' in locals() and not country_top_boxplot_data.empty:
    fig_top15_line = px.line(
        country_top_boxplot_data,
        x='year',
        y='co2',
        color='country',
        title='Émissions de CO₂ des Top 15 pays émetteurs (1850-2023)',
        labels={'year': 'Année', 'co2': 'Émissions de CO₂ (Mt)'})

    fig_top15_line.update_layout(template='plotly_white',
                                 height=600,
                                 margin=dict(l=50, r=50, t=50, b=50))

    st.plotly_chart(fig_top15_line, use_container_width=True)
else:
     st.warning("Aucune donnée suffisante depuis 1850 pour les top 15 pays afin d'afficher le graphique linéaire.")

st.caption("Source : https://ourworldindata.org/")

st.markdown("""
Depuis les années 1900, le niveau mondial d'émissions de CO2 a augmenté très rapidement. Cette augmentation s'explique par la croissance économique, <br>
l'industrialisation et l'augmentation de la population mondiale. <br>
Les États-Unis ont produit la grande majorité des émissions de CO2 jusqu'aux années 2000, date à laquelle la Chine est devenue le plus grand pollueur <br>
du monde. Cependant, les émissions de CO2 de la Chine ont augmenté de manière très rapide, dépassant celles des États-Unis. <br>
L'Inde se distingue aussi avec une augmentation importante de ces émissions de co2 depuis les années 2000. <br>
Le Japon, l'Allemagne et la Russie sont également responsables d'une quantité importante d'émissions de CO2. <br>
La courbe montre que les pays ont des trajectoires différentes pour les émissions de CO2. Par exemple, les émissions des États-Unis ont commencé à <br>
stagner depuis les années 2000, tandis que celles de la Chine ont continué à augmenter de manière prononcée. <br>
Nous pouvons également observer une tendance à la baisse pour les pays Européens où les émissions de CO2 ont connu une réduction depuis les années <br>
1990, cela peut être attribué à un développement plus sain avec une utilisation accrue de sources d'énergie renouvelable. <br>
""", unsafe_allow_html=True)
