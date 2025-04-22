import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns # Garder pour les graphiques seaborn

# --- Configuration de la page Streamlit (optionnel mais bonne pratique) ---
st.set_page_config(layout="wide", page_title="Projet DA Températures et CO2")

# --- Titre et introduction ---
st.title("Projet DA : Analyse des températures et émissions de CO2")
st.header("Quelques exemples de visuels")

# --- Constantes (URLs des données) ---
URL_SEA_TEMP = "https://ourworldindata.org/grapher/sea-surface-temperature-anomaly.csv?v=1&csvType=full&useColumnShortNames=true"
URL_GISS_TEMP_GLOBAL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
URL_GISS_TEMP_NH = "https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv"
URL_GISS_TEMP_SH = "https://data.giss.nasa.gov/gistemp/tabledata_v4/SH.Ts+dSST.csv"
URL_LAND_TEMP = "https://ourworldindata.org/grapher/annual-temperature-anomalies.csv?v=1&csvType=full&useColumnShortNames=true"
URL_CO2_DATA = 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv'

# --- Fonctions de chargement et préparation des données (avec caching) ---

@st.cache_data
def load_sea_temp_data(url: str) -> pd.DataFrame:
    """Charge les données d'anomalie de température de surface de la mer."""
    try:
        df = pd.read_csv(url, storage_options={'User-Agent': 'Our World In Data data fetch/1.0'})
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de température de surface de la mer : {e}")
        return pd.DataFrame()

@st.cache_data
def load_giss_temp_data() -> pd.DataFrame:
    """Charge et combine les données GISS de température (Terres + Océans)."""
    def load_single_giss_file(url, region):
        df = pd.read_csv(url, header=1)
        df = df.iloc[1:]
        df = df.replace('***', pd.NA)
        df['Region'] = region
        df = df[['Year', 'J-D', 'Region']].copy() # Utiliser .copy()
        df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        return df

    try:
        df_global = load_single_giss_file(URL_GISS_TEMP_GLOBAL, "Monde")
        df_north = load_single_giss_file(URL_GISS_TEMP_NH, "Hémisphère Nord")
        df_south = load_single_giss_file(URL_GISS_TEMP_SH, "Hémisphère Sud")
        df_final = pd.concat([df_global, df_north, df_south], ignore_index=True)
        return df_final
    except Exception as e:
        st.error(f"Erreur lors du chargement des données GISS : {e}")
        return pd.DataFrame()

@st.cache_data
def load_land_temp_data(url: str) -> pd.DataFrame:
    """Charge les données d'anomalie de température terrestre (OWID)."""
    try:
        df = pd.read_csv(url, storage_options={'User-Agent': 'My User Agent 1.0'})
        # Convert 'Year' to datetime early if needed for plotly 'date' type
        # df['Year'] = pd.to_datetime(df['Year'], format='%Y') # Let's keep it as integer year for simplicity unless needed for specific plotting feature
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de température terrestre : {e}")
        return pd.DataFrame()

@st.cache_data
def load_co2_data(url: str) -> pd.DataFrame:
    """Charge les données CO2 et effectue un prétraitement initial."""
    try:
        df = pd.read_csv(url)

        # Colonnes à garder
        colonnes_a_garder = [
            'country', 'year', 'iso_code', 'population', 'gdp',
            'co2', 'co2_per_capita', 'methane', 'nitrous_oxide',
            'cumulative_gas_co2', 'cumulative_oil_co2','cumulative_flaring_co2','cumulative_coal_co2','cumulative_other_co2',
            'temperature_change_from_ch4', 'temperature_change_from_co2',
            'temperature_change_from_ghg', 'temperature_change_from_n2o'
        ]
        df = df[colonnes_a_garder].copy() # Utiliser .copy()

        # Les pays (lignes) à retirer
        A_retirer = [
            'Africa (GCP)', 'Asia (GCP)', 'Asia (excl. China and India)', 'Central America (GCP)',
            'Europe (GCP)', 'Europe (excl. EU-27)', 'Europe (excl. EU-28)', 'European Union (27)',
            'European Union (28)', 'High-income countries', 'International aviation',
            'International shipping', 'International transport','Kuwaiti Oil Fires',
            'Kuwaiti Oil Fires (GCP)', 'Least developed countries (Jones et al. 2023)',
            'Low-income countries', 'Lower-middle-income countries', 'Middle East (GCP)',
            'Non-OECD (GCP)', 'North America (GCP)', 'North America (excl. USA)', 'OECD (GCP)',
            'OECD (Jones et al.)', 'Oceania (GCP)', 'Ryukyu Islands (GCP)', 'South America (GCP)',
            'Upper-middle-income countries', 'World', # Souvent la ligne 'World' fausse les comparaisons par pays
            'International Transport' # Ajout car présent parfois
        ]
        # Retirer aussi les entrées qui sont des continents
        continents_list = ['Asia', 'Europe', 'Africa', 'North America', 'Oceania', 'South America']
        A_retirer.extend(continents_list)
        A_retirer = list(set(A_retirer)) # Supprimer les doublons si ajoutés plusieurs fois

        df = df.loc[~df['country'].isin(A_retirer)].copy() # Utiliser .copy()

        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données CO2 : {e}")
        return pd.DataFrame()


@st.cache_data
def load_co2_data_with_regions(url: str) -> pd.DataFrame:
    """Charge les données CO2 et inclut les agrégations régionales."""
    try:
        df = pd.read_csv(url)

        colonnes_a_garder = [
            'country', 'year', 'iso_code', 'co2', 'co2_per_capita'
        ]
        df = df[colonnes_a_garder].copy() # Utiliser .copy()

        # Filtrer pour inclure le monde et les continents
        regions_to_keep = [
             'World','Asia','Europe','North America','Africa','South America','Oceania'
        ]
        # Ajouter les pays qui pourraient être nécessaires pour d'autres graphiques si le df principal les exclut
        # Mais pour les graphiques continentaux et mondiaux spécifiques, ce dataframe est utile.
        # On garde les données pour toutes les années disponibles.
        df_filtered = df[df['country'].isin(regions_to_keep)].copy()

        return df_filtered
    except Exception as e:
        st.error(f"Erreur lors du chargement des données CO2 avec régions : {e}")
        return pd.DataFrame()


# --- Helpers pour la mise en page Plotly ---

def apply_common_plotly_layout_updates(fig: go.Figure, xaxis_is_date: bool = False, **kwargs):
    """Applique les mises en page communes aux figures Plotly."""
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255, 255, 255, 0.7)'
        ),
        font=dict(size=12),
        plot_bgcolor='white',
        margin=dict(t=60, b=60, l=80, r=40), # Marges par défaut
    )
    # Grilles
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    # Configuration spécifique pour l'axe X de type date avec rangeslider/rangeselector
    if xaxis_is_date:
         fig.update_xaxes(
             rangeslider_visible=True,
             type="date", # Plotly gère automatiquement la conversion si la colonne est datetime ou int/float représentant l'année
             rangeselector=dict(
                 buttons=list([
                     dict(count=1, label="1 an", step="year", stepmode="backward"),
                     dict(count=5, label="5 ans", step="year", stepmode="backward"),
                     dict(count=10, label="10 ans", step="year", stepmode="backward"),
                     dict(step="all")
                 ])
             )
         )

    # Appliquer d'autres mises à jour spécifiques passées en kwargs
    fig.update_layout(**kwargs)


# --- Affichage des graphiques ---

st.subheader("Anomalies annuelles de la température de la surface des océans")
df_mer = load_sea_temp_data(URL_SEA_TEMP)

if not df_mer.empty:
    world_data_mer = df_mer[df_mer['Entity'] == 'World']
    northern_hemisphere_data_mer = df_mer[df_mer['Entity'] == 'Northern Hemisphere']
    southern_hemisphere_data_mer = df_mer[df_mer['Entity'] == 'Southern Hemisphere']

    fig_mer = go.Figure()

    fig_mer.add_trace(go.Scatter(x=world_data_mer['Year'], y=world_data_mer['sea_temperature_anomaly_annual'],
                                 mode='lines', name='Moyenne Globale', line=dict(color='royalblue', width=2)))
    fig_mer.add_trace(go.Scatter(x=northern_hemisphere_data_mer['Year'], y=northern_hemisphere_data_mer['sea_temperature_anomaly_annual'],
                                 mode='lines', name='Hémisphère Nord', line=dict(color='firebrick', width=2, dash='dash')))
    fig_mer.add_trace(go.Scatter(x=southern_hemisphere_data_mer['Year'], y=southern_hemisphere_data_mer['sea_temperature_anomaly_annual'],
                                 mode='lines', name='Hémisphère Sud', line=dict(color='forestgreen', width=2, dash='dot')))

    apply_common_plotly_layout_updates(
        fig_mer,
        xaxis_is_date=True, # Appliquer la config date
        title=None, # Titre dans le subheader Streamlit
        xaxis_title="Année",
        yaxis_title="Anomalie de température (°C)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # Exemple d'override si nécessaire
    )

    st.plotly_chart(fig_mer, use_container_width=True)
else:
    st.warning("Impossible d'afficher le graphique de température de surface de la mer car les données n'ont pas été chargées.")

st.caption("Source : https://ourworldindata.org/")
st.markdown("""
Le graphique donne les anomalies annuelles de la température de la surface de la mer par rapport à la période préindustrielle pour le Monde, l'Hémisphère Nord, <br>
et l'Hémisphère Sud). <br>
Les données représentent la différence entre la température moyenne de la surface de la mer et la moyenne de 1861 à 1890, en degrés Celsius, mesuré à une profondeur de 20 centimètres pour l'hémisphère Nord, l'hémisphère Sud et le monde. <br>
Le graphique montre que les océans de l'hémisphère Nord se réchauffent plus vite que ceux de l'hémisphère Sud depuis 2013.
""", unsafe_allow_html=True)


# --- Deuxième graphique: Anomalies de température (Terres + Océans - GISS) ---
st.subheader("Anomalies annuelles de la température : Terres + Océans (GISS)")
df_final_terre = load_giss_temp_data()

if not df_final_terre.empty:
    world_data_terre = df_final_terre[df_final_terre['Region'] == 'Monde']
    northern_hemisphere_data_terre = df_final_terre[df_final_terre['Region'] == 'Hémisphère Nord']
    southern_hemisphere_data_terre = df_final_terre[df_final_terre['Region'] == 'Hémisphère Sud']

    fig_terre = go.Figure()

    fig_terre.add_trace(go.Scatter(x=world_data_terre['Year'], y=world_data_terre['J-D'],
                                   mode='lines', name='Moyenne Globale', line=dict(color='royalblue', width=2)))
    fig_terre.add_trace(go.Scatter(x=northern_hemisphere_data_terre['Year'], y=northern_hemisphere_data_terre['J-D'],
                                   mode='lines', name='Hémisphère Nord', line=dict(color='firebrick', width=2, dash='dash')))
    fig_terre.add_trace(go.Scatter(x=southern_hemisphere_data_terre['Year'], y=southern_hemisphere_data_terre['J-D'],
                                   mode='lines', name='Hémisphère Sud', line=dict(color='forestgreen', width=2, dash='dot')))

    apply_common_plotly_layout_updates(
        fig_terre,
        xaxis_is_date=True, # Appliquer la config date
        title=None, # Titre dans le subheader Streamlit
        xaxis_title="Année",
        yaxis_title="Anomalie de température (°C)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # Exemple d'override
    )

    st.plotly_chart(fig_terre, use_container_width=True)
else:
     st.warning("Impossible d'afficher le graphique de température GISS car les données n'ont pas été chargées.")

st.caption("Source : https://data.giss.nasa.gov/gistemp/")
st.markdown("""
Le dataset utilisé pour ce graphique est la combinaison de 3 datasets : un par région observée (Monde, Hémisphère Nord, Hémisphère Sud). <br>
Le graphique présente les anomalies combinées de température de l'air à la surface terrestre et de l'eau à la surface de la mer, <br>
c'est-à-dire les écarts par rapport aux moyennes correspondantes de 1951 à 1980. <br>
Le graphique confirme un réchauffement plus rapide de l'hémisphère Nord depuis 2013, ce réchauffement est accentué par les températures relevées à la surface terrestre. <br>
<br>
""", unsafe_allow_html=True)

# --- Troisième graphique: Anomalies de température terrestre (OWID) ---
st.subheader("Anomalies annuelles de température (Terres) par pays")
st.markdown("Ce graphique montre les anomalies de température pour le Monde et certains pays (ceux contenant 'NIAID' dans le jeu de données original).")

df_land_temp = load_land_temp_data(URL_LAND_TEMP)

if not df_land_temp.empty:
    # Logique de filtrage des pays (adaptée de la fonction originale)
    countries = df_land_temp['Entity'].unique()
    filtered_countries = [
        country for country in countries
        if not any(keyword in country.lower() for keyword in ["ocean", "seas", "sea", "mediterranean", "world"])
    ]
    selected_countries = ['World']
    selected_countries.extend([country for country in filtered_countries if "NIAID" in country])

    country_data_filtered = df_land_temp[df_land_temp['Entity'].isin(selected_countries)].copy() # Utiliser .copy()
    country_data_filtered['Year'] = pd.to_datetime(country_data_filtered['Year'], format='%Y') # Convertir en datetime pour axe date

    if not country_data_filtered.empty:
        fig_land = go.Figure()
        for country in selected_countries:
            country_data = country_data_filtered[country_data_filtered['Entity'] == country]
            if not country_data.empty:
                legend_name = country.replace(" (NIAID)", "")
                fig_land.add_trace(go.Scatter(x=country_data['Year'], y=country_data['temperature_anomaly'],
                                              name=legend_name, mode='lines+markers', marker=dict(size=4)))

        apply_common_plotly_layout_updates(
             fig_land,
             xaxis_is_date=True, # Appliquer la config date
             title="Anomalies de température annuelles<br><sup>Différence par rapport à la moyenne 1991-2020</sup>", # Titre spécifique conservé dans la figure
             title_x=0.0, title_y=0.98, title_font=dict(size=16),
             xaxis_title="Année",
             yaxis_title="Anomalies de température (°C)",
             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), # Position spécifique de la légende
             annotations=[ # Annotations spécifiques
                 dict(x=country_data_filtered[country_data_filtered['Entity'] == 'World']['Year'].max(),
                      y=country_data_filtered[country_data_filtered['Entity'] == 'World']['temperature_anomaly'].max(),
                      xref="x", yref="y", text="Maximum mondial", showarrow=True, arrowhead=7, ax=0, ay=-40)
             ],
             margin=dict(t=100, b=60, l=80, r=40) # Ajuster la marge haute pour le titre long
        )

        st.plotly_chart(fig_land, use_container_width=True)
    else:
        st.warning("Aucune donnée filtrée disponible pour afficher le graphique terrestre (OWID).")
else:
    st.warning("Impossible d'afficher le graphique terrestre (OWID) car les données n'ont pas été chargées.")


st.caption("Source : https://ourworldindata.org/")
st.markdown("""
Ce graphique montre la différence entre la température moyenne de surface de la terre d'une année et la moyenne de 1991 à 2020, en degrés Celsius.
Il inclut les données pour le Monde et certains pays pour comparaison.
""", unsafe_allow_html=True)


# --- Quatrième graphique: Carte mondiale animée des anomalies de température ---
st.subheader("Carte mondiale animée des anomalies de température")
st.markdown("Cette carte visualise les anomalies de température par pays au fil du temps.")

# Utiliser le dataframe déjà chargé df_land_temp
if not df_land_temp.empty:
    df_carte = df_land_temp.copy() # Utiliser une copie si on modifie le df

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

    # La mise en page des cartes est très spécifique, donc moins de réutilisation de la helper commune
    fig_carte.update_layout(
        title=dict(
            text="<b>Anomalies de Température Annuelles</b><br><sup>Différence par rapport à la moyenne 1991-2020</sup>",
            x=0.5, y=0.95, xanchor='center', font=dict(size=20, family="Arial") # Centrer le titre de la carte
        ),
        geo=dict(
            showframe=False, showcoastlines=True, coastlinecolor="RebeccaPurple",
            showland=True, landcolor="LightGray", showocean=True, oceancolor="LightBlue"
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Anomalie (°C)", font=dict(size=14)),
            thicknessmode="pixels", thickness=20, len=0.7, yanchor="top", y=0.9, tickfont=dict(size=12)
        ),
        margin=dict(r=0, t=80, l=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # height=600 # Streamlit gère la hauteur avec use_container_width
    )

    fig_carte.update_traces(marker_line_width=0.5, marker_line_color="DarkSlateGrey")

    st.plotly_chart(fig_carte, use_container_width=True)
else:
     st.warning("Impossible d'afficher la carte animée des températures car les données n'ont pas été chargées.")

st.caption("Source : https://ourworldindata.org/")
st.markdown("""
Cette carte mondiale animée montre l'évolution de la différence entre la température moyenne de surface d'une année et la moyenne de 1991 à 2020, en degrés Celsius, par pays.
""", unsafe_allow_html=True)


# --- Section Analyse des émissions de CO2 ---
st.title("Analyse des émissions de CO2 mondiales")
st.write("Visualisation des données sur les émissions de CO2 provenant de Our World in Data.")

# Charger les données CO2 une seule fois pour cette section
df_co2_countries = load_co2_data(URL_CO2_DATA) # Contient les pays filtrés
df_co2_regions = load_co2_data_with_regions(URL_CO2_DATA) # Contient monde et continents


# --- Cinquième graphique: Évolution du CO2 per capita (Carte animée) ---
st.subheader("Évolution du CO2 per capita par pays")
st.markdown("Cette carte montre l’évolution des émissions de CO2 par habitant par pays depuis 1900.")

if not df_co2_countries.empty:
    df_filtered_co2_pc_map = df_co2_countries[df_co2_countries['year'] >= 1900].dropna(subset=['co2_per_capita']).copy()

    if not df_filtered_co2_pc_map.empty:
        fig_co2_pc_map = px.choropleth(
            df_filtered_co2_pc_map,
            locations="country",
            locationmode="country names",
            color="co2_per_capita",
            hover_name="country",
            animation_frame="year",
            range_color=[0, df_filtered_co2_pc_map['co2_per_capita'].quantile(0.95)], # Ajuster la plage de couleurs dynamiquement
            color_continuous_scale="Viridis",
            projection="natural earth",
            # title="Évolution du CO2 per capita par pays" # Titre dans subheader
        )

        fig_co2_pc_map.update_layout(
            title=None, # Titre dans subheader
            geo=dict(showframe=False, showcoastlines=False), # Mise en page géographique simplifiée
            coloraxis_colorbar=dict(title="CO2/hab (Tonnes)"),
            margin=dict(r=0, t=40, l=0, b=0) # Ajuster marges
        )

        st.plotly_chart(fig_co2_pc_map, use_container_width=True)
    else:
        st.warning("Aucune donnée filtrée suffisante (après 1900) pour afficher la carte du CO2 par habitant.")
else:
    st.warning("Impossible d'afficher la carte du CO2 par habitant car les données n'ont pas été chargées.")


st.caption("Source : https://ourworldindata.org/")
st.markdown("""
Ce graphique montre l’évolution des émissions de CO2 par pays par habitant au cours du temps. <br>
On voit que les émissions ont beaucoup évolué au cours du temps mais que la majorité concerne l’hémisphère nord. <br>
Les pays occidentaux ont longtemps été en tête puis une augmentation se voit en Asie et dernièrement c’est l’Arabie <br>
Saoudite qui à la plus forte émission par habitant. <br>
""", unsafe_allow_html=True)

# --- Sixième graphique: Émissions totales de CO2 (Carte animée) ---
st.subheader("Émissions mondiales de CO2 par pays au fil du temps (depuis 1950)")
st.markdown("Cette carte montre l'évolution des émissions annuelles *totales* de CO2 par pays depuis 1950.")

if not df_co2_countries.empty:
    world_co2_data_map = df_co2_countries.loc[df_co2_countries['year'] >= 1950].copy()

    if not world_co2_data_map.empty:
        fig_co2_map_total = px.choropleth(
            world_co2_data_map, locations='iso_code', color='co2', animation_frame='year',
            hover_name='country', color_continuous_scale='pubu', projection='natural earth',
            # title='Émissions mondiales de CO2 (en Mt)', # Titre dans subheader
            range_color=(0, world_co2_data_map['co2'].quantile(0.95)) # Utiliser un quantile
        )

        fig_co2_map_total.update_layout(
            title=None, # Titre dans subheader
            geo=dict(showframe=False, showcoastlines=False), # Mise en page géographique simplifiée
            coloraxis_colorbar=dict(title="CO2 (Mt)"),
            margin=dict(r=0, t=40, l=0, b=0), # Ajuster marges
            height=600 # Peut aider à fixer la hauteur pour les cartes animées
        )

        st.plotly_chart(fig_co2_map_total, use_container_width=True)
    else:
        st.warning("Aucune donnée filtrée suffisante (depuis 1950) pour afficher la carte des émissions de CO2.")
else:
    st.warning("Impossible d'afficher la carte des émissions de CO2 car les données n'ont pas été chargées.")

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


# --- Utiliser le dataframe avec régions pour les graphiques continentaux ---

# Septième graphique: Émissions totales de CO2 par continent (Bar plot)
st.subheader("Émissions totales cumulées de CO2 par continent (1950-2023)")
st.write("Ce graphique montre la somme des émissions annuelles de CO2 par continent sur la période 1950-2023.")

if not df_co2_regions.empty:
    # Filtrer pour les continents et les années >= 1950
    df_continent_filtered_total = df_co2_regions.loc[
        (df_co2_regions['country'].isin(['Asia', 'Europe', 'Africa', 'North America', 'Oceania', 'South America'])) &
        (df_co2_regions['year'] >= 1950)
    ].copy()

    if not df_continent_filtered_total.empty:
        df_continent_sum_total = df_continent_filtered_total.groupby('country')['co2'].sum().reset_index()

        fig_continent_bar_total, ax_continent_bar_total = plt.subplots(figsize=(10, 6))
        sns.barplot(x='country', y='co2', data=df_continent_sum_total.sort_values('co2', ascending=False), ax=ax_continent_bar_total, palette='viridis') # Trier et ajouter palette
        ax_continent_bar_total.set_title('Émissions totales cumulées de CO₂ (Mt) par continent (1950-2023)') # Titre conservé dans Matplotlib
        ax_continent_bar_total.set_xlabel('Continent')
        ax_continent_bar_total.set_ylabel('Émissions de CO₂ (Mt)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        st.pyplot(fig_continent_bar_total)
    else:
         st.warning("Aucune donnée filtrée suffisante (continents depuis 1950) pour afficher le graphique à barres.")
else:
     st.warning("Impossible d'afficher le graphique à barres par continent car les données n'ont pas été chargées.")


st.caption("Source : https://ourworldindata.org/")
st.markdown("""
Ce graphique nous montre bien que l'Asie, l'Europe et l'Amérique du Nord sont les continents émettant le plus d'émissions de CO2 de 1950 à 2023, <br>
au moins 5 à 6 fois plus que l'Afrique, l'Océanie et l'Amérique du Sud en cumulé. <br>
""", unsafe_allow_html=True)


# Huitième graphique: Émissions de CO2 par continent au fil du temps (Line plot)
st.subheader("Émissions de CO2 par continent au fil du temps (depuis 1750)")
st.write("Ce graphique montre l'évolution annuelle des émissions totales de CO2 pour chaque continent depuis 1750.")

if not df_co2_regions.empty:
    # Filtrer pour les continents, toutes les années disponibles
    df_continent_all_years_line = df_co2_regions.loc[
        df_co2_regions['country'].isin(['Asia', 'Europe', 'Africa', 'North America', 'Oceania', 'South America'])
    ].copy()

    if not df_continent_all_years_line.empty:
        fig_continent_line_total, ax_continent_line_total = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=df_continent_all_years_line, x='year', y='co2', hue='country', ax=ax_continent_line_total)
        ax_continent_line_total.set_title('Émissions de CO₂ (Mt) par continent et par année (1750-2023)') # Titre conservé dans Matplotlib
        ax_continent_line_total.set_xlabel('Année')
        ax_continent_line_total.set_ylabel('Émissions de CO₂ (Mt)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        st.pyplot(fig_continent_line_total)
    else:
        st.warning("Aucune donnée filtrée suffisante (continents depuis 1750) pour afficher le graphique linéaire.")
else:
     st.warning("Impossible d'afficher le graphique linéaire par continent car les données n'ont pas été chargées.")

st.caption("Source : https://ourworldindata.org/")
st.markdown("""
Ce graphique nous montre bien que l'augmentation des émissions de CO2 est très importante depuis les années 1950, elle est exponentielle pour l'Asie. <br>
Les émissions de C02 concernant l'Europe et l'Amérique du Nord ont évolué un peu avant les années 1900 (révolution industrielle). Mais on voit qu'elles <br>
sont en déclin depuis les années 2000 (attention: appartenance réelle des émissions pour l'Asie ?) <br>
""", unsafe_allow_html=True)


# Neuvième graphique: Émissions de CO2 par habitant par continent (Bar plot)
st.subheader("Émissions totales cumulées de CO2 par habitant par continent (1950-2023)")
st.write("Ce graphique montre la somme des émissions annuelles de CO2 par habitant par continent sur la période 1950-2023.")
# Utiliser le dataframe df_continent_filtered_total déjà filtré pour 1950+
if not df_continent_filtered_total.empty:
    df_co2_per_capita_sum_continent = df_continent_filtered_total.groupby('country')['co2_per_capita'].sum().reset_index()

    fig_co2_per_capita_bar_continent, ax_co2_per_capita_bar_continent = plt.subplots(figsize=(10, 6))
    sns.barplot(x='country', y='co2_per_capita', data=df_co2_per_capita_sum_continent.sort_values('co2_per_capita', ascending=False), ax=ax_co2_per_capita_bar_continent, palette='viridis')
    ax_co2_per_capita_bar_continent.set_title('Émissions totales cumulées de CO₂ par habitant par continent (1950-2023)') # Titre conservé dans Matplotlib
    ax_co2_per_capita_bar_continent.set_xlabel('Continent')
    ax_co2_per_capita_bar_continent.set_ylabel('Émissions de CO₂ par habitant (Tonnes)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig_co2_per_capita_bar_continent)
else:
     st.warning("Aucune donnée filtrée suffisante (continents depuis 1950) pour afficher le graphique à barres par habitant.")


st.caption("Source : https://ourworldindata.org/")
st.markdown("""
Ce graphique nous montre que les continents Amérique du Nord, l'Océanie et l'Europe ont les émissions de CO2 par habitant les plus élevées en cumulé entre 1950 et 2023. <br>
L'Asie, bien que le plus gros émetteur total, a des émissions de CO2 par habitant cumulées moins importantes que ces 3 continents sur cette période. <br>
""", unsafe_allow_html=True)

# Dixième graphique: Émissions de CO2 par habitant par continent au fil du temps (Line plot)
st.subheader("Émissions de CO2 par habitant par continent au fil du temps (depuis 1750)")
st.write("Ce graphique montre l'évolution annuelle des émissions de CO2 par habitant pour chaque continent depuis 1750.")
# Utiliser le dataframe df_continent_all_years_line déjà filtré pour les continents, toutes années
if not df_continent_all_years_line.empty:
    fig_co2_per_capita_line_continent, ax_co2_per_capita_line_continent = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=df_continent_all_years_line, x='year', y='co2_per_capita', hue='country', ax=ax_co2_per_capita_line_continent)
    ax_co2_per_capita_line_continent.set_title('Émissions de CO₂ par habitant par continent et par année (1750-2023)') # Titre conservé dans Matplotlib
    ax_co2_per_capita_line_continent.set_xlabel('Année')
    ax_co2_per_capita_line_continent.set_ylabel('Émissions de CO₂ par habitant (Tonnes)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    st.pyplot(fig_co2_per_capita_line_continent)
else:
     st.warning("Aucune donnée filtrée suffisante (continents depuis 1750) pour afficher le graphique linéaire par habitant.")

st.caption("Source : https://ourworldindata.org/")
st.markdown("""
Ce graphique montre bien qu'à partir des années 1850 (fin de l'ère de la révolution industrielle) les émissions de CO2 par habitant ont commencé à exploser aux Etats-Unis. <br>
Elles ont atteint leur seuil maximal avant les années 2000 et commence à diminuer depuis les années 2000. La situation est similaire pour l'Europe <br>
et l'Océanie bien que les taux d'émissions de CO2 par habitant soient moins élevés que pour les Etats-Unis. <br>
En revanche, concernant l'Asie on remarque une augmentation des émissions de CO2 par habitant à partir des années 1950, cela signifie que l'Asie est <br>
actuellement en pleine croissance industrielle. <br>
""", unsafe_allow_html=True)


# Onzième graphique: Top 15 pays émetteurs de CO2 (Bar plot)
st.subheader("Top 15 des pays émetteurs de CO2 (Moyenne 1950-2023)")
st.write("Classement des 15 pays ayant les émissions annuelles moyennes de CO2 les plus élevées sur la période 1950-2023.")

if not df_co2_countries.empty:
    # Filtrer les données pour la période 1950-2023 pour le calcul de la moyenne
    df_co2_1950_plus = df_co2_countries[df_co2_countries['year'] >= 1950].copy()

    if not df_co2_1950_plus.empty:
        # Calculer la moyenne par pays sur la période
        df_moy_co2 = df_co2_1950_plus.groupby('country')['co2'].mean().reset_index()

        # On enlève les NaN et trie
        df_countries_only_co2 = df_moy_co2.dropna(subset=['co2'])

        # On trie et garde les 15 premiers
        top_15_mean_emitters_co2 = df_countries_only_co2.sort_values(by='co2', ascending=False).head(15).copy()

        if not top_15_mean_emitters_co2.empty:
            fig_top15_bar_co2 = px.bar(
                top_15_mean_emitters_co2.sort_values(by='co2', ascending=True), # Pour ordre croissant en vertical
                x='co2',
                y='country',
                orientation='h',
                # title='Top 15 des pays émetteurs de CO₂ (Moyenne 1950-2023)', # Titre dans subheader
                labels={'co2': 'Émissions de CO₂ (Mt)', 'country': 'Pays'},
                color='co2', # Colorer par valeur d'émission
                color_continuous_scale='reds',
                title=None # Titre dans subheader
            )

            # Pas besoin de la helper car c'est un bar plot horizontal, mise en page spécifique
            fig_top15_bar_co2.update_layout(
                 template='plotly_white',
                 height=600,
                 margin=dict(l=150, r=50, t=50, b=50),
                 xaxis_title='Émissions de CO₂ (Mt)', # Déplacé ici pour plus de clarté
                 yaxis_title='Pays' # Déplacé ici
             )
            # Les titres des axes sont déjà dans labels et layout, pas besoin de update_xaxes/yaxes ici

            st.plotly_chart(fig_top15_bar_co2, use_container_width=True)
        else:
            st.warning("Aucune donnée suffisante (après 1950) pour calculer le top 15 des pays émetteurs.")
    else:
        st.warning("Aucune donnée disponible dans la période 1950-2023 pour calculer le top 15 des pays émetteurs.")
else:
    st.warning("Les données CO2 ne sont pas chargées. Impossible de calculer le top 15.")


st.caption("Source : https://ourworldindata.org/")
st.markdown("""
On voit bien que les US et la Chine sont de loin les deux plus gros émetteurs de CO2 sur ces dernières années (en moyenne sur la période 1950-2023). <br>
""", unsafe_allow_html=True)


# Douzième graphique: Boxplots sur la liste des top 15 émetteurs de CO2
st.subheader("Distribution annuelle des émissions de CO2 pour les Top 15 pays émetteurs")
st.write("Ces boxplots montrent la distribution annuelle des émissions de CO2 pour les 15 pays les plus émetteurs (basé sur la moyenne 1950-2023), depuis 1850.")

# Utiliser la liste top_15_mean_emitters_co2 calculée précédemment
if 'top_15_mean_emitters_co2' in locals() and not top_15_mean_emitters_co2.empty:
    top15_countries_list_co2 = top_15_mean_emitters_co2['country'].tolist()

    # Filtrer le dataframe CO2 complet pour ces pays depuis 1850
    country_top_boxplot_data_co2 = df_co2_countries.loc[
        (df_co2_countries['country'].isin(top15_countries_list_co2)) &
        (df_co2_countries['year'] >= 1850)
    ].copy()

    if not country_top_boxplot_data_co2.empty:
        fig_boxplot_co2 = px.box(
            country_top_boxplot_data_co2, x="country", y="co2", hover_data=["year"],
            # title="Boxplots des émissions de CO2 depuis 1850 - Top 15 pays émetteurs" # Titre dans subheader
        )

        apply_common_plotly_layout_updates(
            fig_boxplot_co2,
            title=None, # Titre dans subheader
            xaxis_title="Pays",
            yaxis_title="Émissions CO2 (Mt)",
            # Les boxplots n'utilisent généralement pas rangeslider/rangeselector
            # margin=dict(l=50, r=50, t=50, b=50) # Marges ajustées par helper
        )

        st.plotly_chart(fig_boxplot_co2, use_container_width=True)
    else:
         st.warning("Aucune donnée suffisante depuis 1850 pour les top 15 pays afin d'afficher les boxplots.")
else:
    st.warning("Le top 15 des pays émetteurs n'a pas été calculé. Impossible d'afficher les boxplots.")

st.caption("Source : https://ourworldindata.org/")
st.markdown("""
Les boxplots nous permettent ici de juger la distribution annuelle des émissions pour ces pays. <br>
Ils mettent en évidence que les États-Unis sont de gros émetteurs, et depuis longtemps, tandis qu'on voit que la distribution pour la Chine a beaucoup <br>
évolué, avec une forte augmentation des émissions annuelles au fil du temps. <br>
""", unsafe_allow_html=True)


# Treizième graphique: Émissions de CO2 des top 15 émetteurs au fil du temps (Line plot)
st.subheader("Émissions de CO2 des Top 15 pays émetteurs au fil du temps (depuis 1850)")
st.write("Ce graphique montre l'évolution annuelle des émissions totales de CO2 pour les 15 pays les plus émetteurs (basé sur la moyenne 1950-2023), depuis 1850.")

# Utiliser le même DataFrame filtré pour les boxplots
if 'country_top_boxplot_data_co2' in locals() and not country_top_boxplot_data_co2.empty:
    fig_top15_line_co2 = px.line(
        country_top_boxplot_data_co2,
        x='year',
        y='co2',
        color='country',
        # title='Émissions de CO₂ des Top 15 pays émetteurs (1850-2023)', # Titre dans subheader
        labels={'year': 'Année', 'co2': 'Émissions de CO₂ (Mt)'}
    )

    apply_common_plotly_layout_updates(
        fig_top15_line_co2,
        title=None, # Titre dans subheader
        xaxis_title='Année',
        yaxis_title='Émissions de CO₂ (Mt)',
        # xaxis_is_date=False # Année comme entier ici, pas besoin de config date complexe
        template='plotly_white' # Template spécifique à px.line
    )

    st.plotly_chart(fig_top15_line_co2, use_container_width=True)
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
