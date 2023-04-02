# Import Libraries
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from math import sqrt
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set Page configuration
# Read more at https://docs.streamlit.io/1.6.0/library/api-reference/utilities/st.set_page_config
st.set_page_config(page_title='Football Player Price Prediction', page_icon='⚽', layout='wide')

# +
# Set title and picture for app
# Create 2 columns
col1, col2 = st.columns([3,1])

with col1:
    st.title('Football Player Price Prediction')
    
with col2:
    st.image('https://a4.espncdn.com/combiner/i?img=%2Fphoto%2F2022%2F1109%2Fsoc_fc_rank_16x9.jpg')


# -

# Load data and model
@st.cache_data()
def load_data():
    players = pd.read_csv('FIFA_23_Players_Data.csv')
    
    # Prep data
    # Fill missing values in 'value_eur' with 0
    players['value_eur'].fillna(0, inplace=True)
    
    # Remove accents from names
    players['short_name'] = players['short_name'].apply(unidecode)
    players['long_name'] = players['long_name'].apply(unidecode)

    # Separate dataset into goalkeepers and outfield players (defenders, midfielders & strikers)
    goalkeepers = players[players['player_positions']=='GK'].copy()
    goalkeepers.reset_index(drop=True,inplace=True)

    outfieldplayers = players[players['player_positions']!='GK'].copy()
    outfieldplayers.reset_index(drop=True,inplace=True)
    
    # Get columns for prediction model
    model_data_goalkeepers = goalkeepers[['value_eur','age','goalkeeping_diving','goalkeeping_handling','goalkeeping_kicking',
                              'goalkeeping_positioning','goalkeeping_reflexes','goalkeeping_speed']].copy()

    model_data_outfieldplayers = outfieldplayers[['value_eur','age','pace','shooting','passing',
                                              'dribbling','defending','physic']].copy()
    
    # GK Random Forest Regression
    X = model_data_goalkeepers.drop(['value_eur'], axis=1)
    y = model_data_goalkeepers['value_eur']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_gk_model = RandomForestRegressor(n_estimators=100, max_depth=None,
                                    min_samples_split=5, min_samples_leaf=5, random_state=42)
    rf_gk_model.fit(X_train, y_train)

    # GK Create column for predictions
    goalkeepers['Predicted Value'] = rf_gk_model.predict(X)
    
    # Outfield players Random Forest Regression
    X = model_data_outfieldplayers.drop(['value_eur'], axis=1)
    y = model_data_outfieldplayers['value_eur']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_op_model = RandomForestRegressor(n_estimators=100, max_depth=None,
                                    min_samples_split=5, min_samples_leaf=5, random_state=42)
    rf_op_model.fit(X_train, y_train)

    # Outfield players Create column for predictions
    outfieldplayers['Predicted Value'] = rf_op_model.predict(X)
    
    # Create column for difference between actual and predicted value
    goalkeepers['Over/Under Value'] = goalkeepers['value_eur'] - goalkeepers['Predicted Value']
    outfieldplayers['Over/Under Value'] = outfieldplayers['value_eur'] - outfieldplayers['Predicted Value']
    
    # Create column for overvalued / undervalued / fair value labels
    goalkeepers['Overvalued/Undervalued'] = np.where(goalkeepers['Over/Under Value']>0, 'Overvalued',
                                                     np.where(goalkeepers['Over/Under Value']==0, 'Fair value', 'Undervalued'))
    outfieldplayers['Overvalued/Undervalued'] = np.where(outfieldplayers['Over/Under Value']>0, 'Overvalued',
                                                 np.where(outfieldplayers['Over/Under Value']==0, 'Fair value', 'Undervalued'))
    
    # Concatenate GK and Outfield players datasets
    players_final = pd.concat([goalkeepers,outfieldplayers],axis=0)
    players_final.sort_values(by=['overall','potential'], ascending=False, inplace=True)
    players_final.reset_index(drop=True, inplace=True)
    
    return rf_gk_model, rf_op_model, goalkeepers, outfieldplayers, players_final


rf_gk_model, rf_op_model, goalkeepers, outfieldplayers, players_final = load_data()

# Create 3 tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Intro", "Goalkeepers", "Outfield Players", "Player Profiles", "Price Prediction"])

# Intro tab1
with tab1:
    st.subheader('Dashboards are filtered for players above 80 Overall Rating to focus on top players.')

# Goalkeepers Tableau tab2
with tab2:
    def main():
        html_temp = """
        <div class='tableauPlaceholder' id='viz1680441144546' style='position: static'>
            <noscript>
                <a href='#'>
                    <img alt='Goalkeepers' src='https://public.tableau.com/static/images/Go/Goalkeepers_16804303938650/Goalkeepers/1_rss.png' style='border: none'/>
                </a>
            </noscript>
            <object class='tableauViz' style='display:none;'>
                <param name='host_url' value='https://public.tableau.com/' />
                <param name='embed_code_version' value='3' />
                <param name='site_root' value='' />
                <param name='name' value='Goalkeepers_16804303938650/Goalkeepers' />
                <param name='tabs' value='no' />
                <param name='toolbar' value='yes' />
                <param name='static_image' value='https://public.tableau.com/static/images/Go/Goalkeepers_16804303938650/Goalkeepers/1.png' />
                <param name='animate_transition' value='yes' />
                <param name='display_static_image' value='yes' />
                <param name='display_spinner' value='yes' />
                <param name='display_overlay' value='yes' />
                <param name='display_count' value='yes' />
                <param name='language' value='en-US' />
                <param name='filter' value='publish=yes' />
            </object>
        </div>

        <script type='text/javascript' src='https://public.tableau.com/javascripts/api/viz_v1.js'></script>
        <script type='text/javascript'>
            var divElement = document.getElementById('viz1680441144546');
            var vizElement = divElement.getElementsByTagName('object')[0];
            if (divElement.offsetWidth > 800) {
                vizElement.style.width = '100%';
                vizElement.style.height = (divElement.offsetWidth * 0.5) + 'px';
            } else if (divElement.offsetWidth > 500) {
                vizElement.style.width = '100%';
                vizElement.style.height = (divElement.offsetWidth * 0.5) + 'px';
            } else {
                vizElement.style.width = '100%';
                vizElement.style.height = '2000px';
            }
        </script>
        """
        components.html(html_temp, width=None, height=1000, scrolling=False)
    if __name__ == "__main__":    
        main()

# Outfield Players Tableau tab3
with tab3:
    def main():
        html_temp = """
        <div class='tableauPlaceholder' id='viz1680441224329' style='position: static'>
          <noscript>
            <a href='#'>
              <img alt='Outfield Players ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ou&#47;OutfieldPlayers&#47;OutfieldPlayers&#47;1_rss.png' style='border: none' />
            </a>
          </noscript>
          <object class='tableauViz' style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
            <param name='embed_code_version' value='3' />
            <param name='site_root' value='' />
            <param name='name' value='OutfieldPlayers&#47;OutfieldPlayers' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ou&#47;OutfieldPlayers&#47;OutfieldPlayers&#47;1.png' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='en-US' />
            <param name='filter' value='publish=yes' />
          </object>
          <script type='text/javascript'>
            var divElement = document.getElementById('viz1680441224329');
            var vizElement = divElement.getElementsByTagName('object')[0];
            if (divElement.offsetWidth > 800) {
              vizElement.style.width = '100%';
              vizElement.style.height = (divElement.offsetWidth * 0.5) + 'px';
            } else if (divElement.offsetWidth > 500) {
              vizElement.style.width = '100%';
              vizElement.style.height = (divElement.offsetWidth * 0.75) + 'px';
            } else {
              vizElement.style.width = '100%';
              vizElement.style.height = '2077px';
            }
            var scriptElement = document.createElement('script');
            scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
            vizElement.parentNode.insertBefore(scriptElement, vizElement);
          </script>
        </div>"""
        components.html(html_temp, width=None, height=1000, scrolling=False)
    if __name__ == "__main__":    
        main()

# Player Profiles section
with tab4:
    st.subheader('Player Profiles')

    namelist = players_final['long_name'].tolist()
    selected_players = st.multiselect('Select Players', namelist)
    
    player_table = players_final[players_final['long_name'].isin(selected_players)]
    for i in range(len(player_table)):
        name = player_table.iloc[i]['long_name']
        picture = player_table.iloc[i]['player_face_url']
        actual_value = player_table.iloc[i]['value_eur'].astype(int)
        predicted_value = player_table.iloc[i]['Predicted Value'].astype(int)
        position = player_table.iloc[i]['player_positions']
        club = player_table.iloc[i]['club_name']
        age = player_table.iloc[i]['age']
        overunder = player_table.iloc[i]['Overvalued/Undervalued']
        overundervalue = player_table.iloc[i]['Over/Under Value'].astype(int)
        pace = player_table.iloc[i]['pace'].astype(int)
        shooting = player_table.iloc[i]['shooting'].astype(int)
        passing = player_table.iloc[i]['passing'].astype(int)
        dribbling = player_table.iloc[i]['dribbling'].astype(int)
        defending = player_table.iloc[i]['defending'].astype(int)
        physic = player_table.iloc[i]['physic'].astype(int)
        goalkeeping_diving = player_table.iloc[i]['goalkeeping_diving'].astype(int)
        goalkeeping_handling = player_table.iloc[i]['goalkeeping_handling'].astype(int)
        goalkeeping_kicking = player_table.iloc[i]['goalkeeping_kicking'].astype(int)
        goalkeeping_positioning = player_table.iloc[i]['goalkeeping_positioning'].astype(int)
        goalkeeping_reflexes = player_table.iloc[i]['goalkeeping_reflexes'].astype(int)
        goalkeeping_speed = player_table.iloc[i]['goalkeeping_speed'].astype(int)

        # Create 3 columns for profile
        col1, col2, col3, col4 = st.columns(4)

        # st.number_input(label, min_value=None, max_value=None, value= )
        with col1:
            st.image(picture)
        with col2:
            st.write(f'Name: {name}')
            st.write(f'Age: {age}')
            st.write(f'Position: {position}')
            st.write(f'Club: {club}')
        with col3:
            st.write(f'Actual Price: € {actual_value:,}')
            st.write(f'Predicted Price: € {predicted_value:,}')
            if overundervalue>0:
                st.write(f'{overunder} by: :red[{overundervalue:,}]')
            elif overundervalue<0:
                st.write(f'{overunder} by: :green[{overundervalue:,}]')
        with col4:
            if position == 'GK':
                st.write(f'Diving: {goalkeeping_diving}')
                st.write(f'Handling: {goalkeeping_handling}')
                st.write(f'Kicking: {goalkeeping_kicking}')
                st.write(f'Reflexes: {goalkeeping_reflexes}')
                st.write(f'Speed: {goalkeeping_speed}')
                st.write(f'Positioning: {goalkeeping_positioning}')
            else:
                st.write(f'Pace: {pace}')
                st.write(f'Shooting: {shooting}')
                st.write(f'Passing: {passing}')
                st.write(f'Dribbling: {dribbling}')
                st.write(f'Defending: {defending}')
                st.write(f'Physicality: {physic}')                
        st.markdown("""---""")

# GK section
with tab5:
    # Set input widgets
    st.header('Goalkeeper Price Prediction')
    st.subheader('Select Attributes')

    # Create 6 columns for sliders
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # st.number_input(label, min_value=None, max_value=None, value= )
    with col1:
        diving = st.slider('Diving', 1, 99, 50)
        age = st.slider('Age', 15, 50, 25)
    with col2:
        handling = st.slider('Handling', 1, 99, 50)
    with col3:
        kicking = st.slider('Kicking', 1, 99, 50)
    with col4:
        reflexes = st.slider('Reflexes', 1, 99, 50)
    with col5:
        speed = st.slider('Speed', 1, 99, 50)
    with col6:
        positioning = st.slider('Positioning', 1, 99, 50)
    
    # Generate prediction based on user selected attributes
    y_pred_gk = rf_gk_model.predict([[age, diving, handling, kicking, positioning, reflexes, speed]]).astype(int)
    st.metric('Predicted Price for Goalkeeper:', f'€ {y_pred_gk[0]:,}')
    st.markdown("""---""")

# Outfield Players section
with tab5:
    # Set input widgets
    st.header('Outfield Player Price Prediction')
    st.subheader('Select Attributes')

    # Create 6 columns for sliders
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # st.number_input(label, min_value=None, max_value=None, value= )
    with col1:
        pace = st.slider('Pace', 1, 99, 50)
        age = st.slider('Age', 15, 50, 25, key='age2')
    with col2:
        shooting = st.slider('Shooting', 1, 99, 50)
    with col3:
        passing = st.slider('Passing', 1, 99, 50)
    with col4:
        dribbling = st.slider('Dribbling', 1, 99, 50)
    with col5:
        defending = st.slider('Defending', 1, 99, 50)
    with col6:
        physicality = st.slider('Physicality', 1, 99, 50)

    # Outfield players
    y_pred_op = rf_op_model.predict([[age, pace, shooting, passing, dribbling, defending, physicality]]).astype(int)
    st.metric('Predicted Price for Outfield Player:', f'€ {y_pred_op[0]:,}')
    
    # Centralise text
    css='''
    [data-testid="metric-container"] {
        width: fit-content;
        margin: auto;
    }

    [data-testid="metric-container"] > div {
        width: fit-content;
        margin: auto;
    }

    [data-testid="metric-container"] label {
        width: fit-content;
        margin: auto;
    }
    '''
    st.markdown(f'<style>{css}</style>',unsafe_allow_html=True)
