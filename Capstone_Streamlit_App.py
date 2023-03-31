# Package for removing accents from names
# pip install unidecode

# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
# from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set Page configuration
# Read more at https://docs.streamlit.io/1.6.0/library/api-reference/utilities/st.set_page_config
st.set_page_config(page_title='Football Player Price Prediction ⚽🏃💵', page_icon='⚽', layout='centered')

# +
# import streamlit.components.v1 as components
# def main():
#     html_temp = “”"your embed code here”“”
#     components.html(html_temp, height=1000)
# if __name__ == “__main__“:
#     main()

# +
# Set background
# def set_bg_hack_url():
#     '''
#     A function to set background image from a url.
#     '''
#     image_url = "https://wallpaper-mania.com/wp-content/uploads/2018/09/High_resolution_wallpaper_background_ID_77702048137.jpg"
#     st.markdown(
#         f"""
#         <style>
#             .stApp {{
#                 background-image: url('{image_url}');
#                 background-size: cover;
#             }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# set_bg_hack_url()
# -

# Set title of the app
st.title('Football Player Price Prediction ⚽🏃💵')


# Load data
@st.cache_data()
def load_data():
    players = pd.read_csv('FIFA_23_Players_Data.csv')
    
    # Prep data
    # Fill missing values in 'value_eur' with 0
    players['value_eur'].fillna(0, inplace=True)
    
    # Remove accents from names
#     players['short_name'] = players['short_name'].apply(unidecode)
#     players['long_name'] = players['long_name'].apply(unidecode)

    # Separate dataset into goalkeepers and outfield players (defenders, midfielders & strikers)
    goalkeepers = players[players['player_positions']=='GK'].copy()
    goalkeepers.reset_index(drop=True,inplace=True)

    outfieldplayers = players[players['player_positions']!='GK'].copy()
    outfieldplayers.reset_index(drop=True,inplace=True)
    
    # Get columns for prediction model
    model_data_goalkeepers = goalkeepers[['value_eur','goalkeeping_diving','goalkeeping_handling','goalkeeping_kicking',
                                  'goalkeeping_positioning','goalkeeping_reflexes','goalkeeping_speed']].copy()

    model_data_outfieldplayers = outfieldplayers[['value_eur','pace','shooting','passing',
                                              'dribbling','defending','physic']].copy()
    
    # GK Random Forest Regression
    X = model_data_goalkeepers.drop(['value_eur'], axis=1)
    y = model_data_goalkeepers['value_eur']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_gk_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_gk_model.fit(X_train, y_train)

    # GK Create column for predictions
    goalkeepers['Predicted Value'] = rf_gk_model.predict(X)
    
    # Outfield players Random Forest Regression
    X = model_data_outfieldplayers.drop(['value_eur'], axis=1)
    y = model_data_outfieldplayers['value_eur']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_op_model = RandomForestRegressor(n_estimators=100, random_state=42)
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

# +
# tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
# tab1.write("this is tab 1")


# # Most Undervalued Players
# st.subheader('Most Undervalued Players')

# st.dataframe(goalkeepers.sort_values(by='Over/Under Value').head())

# +
# GK section
# Set input widgets
st.header('Goalkeeper Price Prediction')
st.subheader('Select Attributes')

# Create 6 columns for sliders
col1, col2, col3, col4, col5, col6 = st.columns(6)

# st.number_input(label, min_value=None, max_value=None, value= )
diving = col1.slider('Diving', 0, 99, 50)
handling = col2.slider('Handling', 0, 99, 50)
kicking = col3.slider('Kicking', 0, 99, 50)
positioning = col4.slider('Positioning', 0, 99, 50)
reflexes = col5.slider('Reflexes', 0, 99, 50)
speed = col6.slider('Speed', 0, 99, 50)

# Generate prediction based on user selected attributes
y_pred_gk = rf_gk_model.predict([[diving, handling, kicking, positioning, reflexes, speed]]).astype(int)

st.metric('Predicted Price for Goalkeeper:', f'€ {y_pred_gk[0]:,}')

# +
# Outfield Players section
# Set input widgets
st.header('Outfield Player Price Prediction (Defender/Midfielder/Striker)')
st.subheader('Select Attributes')

# Create 6 columns for sliders
col1, col2, col3, col4, col5, col6 = st.columns(6)

# st.number_input(label, min_value=None, max_value=None, value= )
pace = col1.slider('Pace', 0, 99, 50)
shooting = col2.slider('Shooting', 0, 99, 50)
passing = col3.slider('Passing', 0, 99, 50)
dribbling = col4.slider('Dribbling', 0, 99, 50)
defending = col5.slider('Defending', 0, 99, 50)
physicality = col6.slider('Physicality', 0, 99, 50)

# Outfield players
y_pred_op = rf_op_model.predict([[pace, shooting, passing, dribbling, defending, physicality]]).astype(int)

st.metric('Predicted Price for Outfield Player:', f'€ {y_pred_op[0]:,}')

# +
# Display EDA
# st.subheader('Exploratory Data Analysis')
# st.write('The data is grouped by the class and the variable mean is computed for each class.')
# groupby_species_mean = df.groupby('Species').mean()
# st.write(groupby_species_mean)
# st.bar_chart(groupby_species_mean.T)

# +
# Player Profiles section
st.subheader('Player Profiles')

namelist = players_final['long_name'].tolist()
selected_players = st.multiselect('Select Players', namelist)
try:
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
        
        # Create 2 columns for profile
        col1, col2, col3 = st.columns([1, 2, 2])

        # st.number_input(label, min_value=None, max_value=None, value= )
        col1.image(picture)
        with col2:
            st.write(f'Name: {name}')
            st.write(f'Age: {age}')
            st.write(f'Position: {position}')
            st.write(f'Club: {club}')
        with col3:
            st.write(f'Actual Price: € {actual_value:,}')
            st.write(f'Predicted Price: € {predicted_value:,}')
            st.metric(f'{overunder} by', value=None, delta=f'{overundervalue:,}', delta_color = 'inverse')
        st.markdown("""---""")
            
except:
    pass
