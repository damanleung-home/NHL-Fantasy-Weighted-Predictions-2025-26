import streamlit as st
import pandas as pd
import os
import json
from scipy.stats import zscore
import warnings

# Suppress all warnings for a cleaner UI
warnings.filterwarnings('ignore')

# --- Helper Functions ---
def load_data_and_weights(predictions_directory):
    """
    Loads all necessary data and weight files.
    """
    skater_file = os.path.join(predictions_directory, 'cleaned_combined_analyst_skaters.csv')
    goalie_file = os.path.join(predictions_directory, 'cleaned_combined_analyst_goalies.csv')
    skater_weights_file = os.path.join(predictions_directory, 'skater_weights.json')
    goalie_weights_file = os.path.join(predictions_directory, 'goalie_weights.json')

    if not all([os.path.exists(skater_file), os.path.exists(goalie_file), os.path.exists(skater_weights_file), os.path.exists(goalie_weights_file)]):
        st.error("One or more required data or weight files are missing. Please ensure all files are in the correct directory.")
        return None, None, None, None
    
    skater_df = pd.read_csv(skater_file)
    goalie_df = pd.read_csv(goalie_file)
    with open(skater_weights_file, 'r') as f:
        skater_weights = json.load(f)
    with open(goalie_weights_file, 'r') as f:
        goalie_weights = json.load(f)

    return skater_df, goalie_df, skater_weights, goalie_weights

def calculate_weighted_prediction(df, weights_config, position):
    """
    Calculates the final weighted prediction for each stat based on analyst weights.
    """
    categories = ['GP', 'G', 'A', '+/-', 'PIM', 'PPP', 'SHP', 'SOG', 'FOW', 'HIT', 'BLK'] if position == 'skater' else ['GP', 'W', 'GAA', 'SV', 'SV%', 'SO']
    analysts = list(weights_config['analyst_weights'][categories[0]].keys())
    
    all_predictions = pd.DataFrame()
    all_predictions['PLAYER'] = df['PLAYER']
    all_predictions['POS'] = df['POS']
    all_predictions['PlayerID'] = df['PlayerID']
    
    for category in categories:
        category_analyst_weights = [weights_config['analyst_weights'][category][analyst] for analyst in analysts]
        predictions_to_average = df[[f'{category}_{analyst}' for analyst in analysts]].copy()
        predictions_to_average = predictions_to_average.T.fillna(predictions_to_average.mean(axis=1)).T
        
        all_predictions[f'Weighted_Predicted_{category}'] = (predictions_to_average * category_analyst_weights).sum(axis=1) / sum(category_analyst_weights)

    return all_predictions

def calculate_zscore_ranking(df, stat_weights_config, position, scarcity_enabled=False):
    """
    Calculates a final ranking using weighted Z-scores.
    """
    categories = list(stat_weights_config.keys())
    
    df['Final_Ranking_Score'] = 0.0
    for category in categories:
        is_negative_stat = (position == 'goalie' and category in ['GAA'])
        
        predicted_col = f'Weighted_Predicted_{category}'
        df[f'Z_score_{category}'] = zscore(df[predicted_col])
        
        # Apply negative sign for negative stats
        if is_negative_stat:
            df[f'Z_score_{category}'] *= -1

        df['Final_Ranking_Score'] += df[f'Z_score_{category}'] * stat_weights_config[category]

    if scarcity_enabled:
        position_zscores = df.groupby('POS')['Final_Ranking_Score'].transform('mean')
        overall_avg_zscore = df['Final_Ranking_Score'].mean()
        scarcity_bonus = overall_avg_zscore - position_zscores
        df['Final_Ranking_Score'] += scarcity_bonus
        
    return df.sort_values(by='Final_Ranking_Score', ascending=False)

# --- Streamlit UI ---
st.title("Fantasy Hockey Prediction Tool")
st.markdown("---")

predictions_directory = '.'

# Load data and weights into session state
if 'skater_df_raw' not in st.session_state:
    st.session_state.skater_df_raw, st.session_state.goalie_df_raw, st.session_state.skater_weights_config, st.session_state.goalie_weights_config = load_data_and_weights(predictions_directory)

skater_df_raw, goalie_df_raw, skater_weights_config, goalie_weights_config = st.session_state.skater_df_raw, st.session_state.goalie_df_raw, st.session_state.skater_weights_config, st.session_state.goalie_weights_config

if skater_df_raw is not None and goalie_df_raw is not None:
    # Dictionary to map analyst acronyms to full names
    analyst_names = {
        "AnG": "Apples & Ginos", "DFO": "Dailyfaceoff", "DtZ": "DatsyukToZetterberg",
        "LX": "LineupExperts", "Cull": "Scott Cullen", "SL": "Steve Laidlaw",
        "YF": "Yahoo Fantrax", "KUB": "Kubota", "BFH": "Bangers Fantasy Hockey",
        "i1": "Dom Luszczyszyn"
    }
    
    # Callback function to reset weights
    def reset_weights(key, default_value):
        st.session_state[key] = default_value
    with st.container(border=True):
        st.header("Customize Skater Rankings")
        
        skater_df = skater_df_raw[skater_df_raw['season'] == '2025-2026'].copy()
        
        #with st.expander("Analyst Weights"):
        with st.container(border=True):
            st.subheader("Analyst Weights")
            for category in skater_weights_config['analyst_weights'].keys():
                with st.expander(f"**{category}**"):
                #st.write(f"**{category}**")
                    for analyst, weight in skater_weights_config['analyst_weights'][category].items():
                        col1, col2 = st.columns([0.8, 0.2])
                        with col1:
                            key = f"skater_analyst_{category}_{analyst}"
                            if key not in st.session_state:
                                st.session_state[key] = float(weight)
                            st.slider(f"**{analyst_names.get(analyst, analyst)} Weight**", 0.0, 100.0, st.session_state[key], key=key)
                        with col2:
                            st.button("Reset", key=f"reset_skater_analyst_{category}_{analyst}", on_click=reset_weights, args=(key, float(weight)))

        #st.subheader("Stat Weights")
        st.markdown("---")
        with st.container(border=True):
            st.subheader("Stat Weights")
            with st.expander("Stat Weights"):
                for stat, weight in skater_weights_config['stat_weights'].items():
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        key = f"skater_stat_{stat}"
                        if key not in st.session_state:
                            st.session_state[key] = float(weight)
                        st.slider(f"**{stat} Weight**", 0.0, 2.0, st.session_state[key], key=key)
                    with col2:
                        st.button("Reset", key=f"reset_skater_stat_{stat}", on_click=reset_weights, args=(key, float(weight)))

                for category in skater_weights_config['analyst_weights'].keys():
                    for analyst in skater_weights_config['analyst_weights'][category].keys():
                        skater_weights_config['analyst_weights'][category][analyst] = st.session_state[f"skater_analyst_{category}_{analyst}"]
                        
                for stat in skater_weights_config['stat_weights'].keys():
                    skater_weights_config['stat_weights'][stat] = st.session_state[f"skater_stat_{stat}"]

    st.markdown("---")
    with st.container(border=True):
        st.header("Customize Goalie Rankings")
        
        goalie_df = goalie_df_raw[goalie_df_raw['season'] == '2025-2026'].copy()

        with st.container(border=True):
            st.subheader("Analyst Weights")
            for category in goalie_weights_config['analyst_weights'].keys():
                #st.write(f"**{category}**")
                with st.expander(f"**{category}**"):                
                    for analyst, weight in goalie_weights_config['analyst_weights'][category].items():
                        col1, col2 = st.columns([0.8, 0.2])
                        with col1:
                            key = f"goalie_analyst_{category}_{analyst}"
                            if key not in st.session_state:
                                st.session_state[key] = float(weight)
                            st.slider(f"**{analyst_names.get(analyst, analyst)} Weight**", 0.0, 100.0, st.session_state[key], key=key)
                        with col2:
                            st.button("Reset", key=f"reset_goalie_analyst_{category}_{analyst}", on_click=reset_weights, args=(key, float(weight)))

        #st.subheader("Stat Weights")
        st.markdown("---")
        with st.container(border=True):
            st.subheader("Stat Weights")
            with st.expander("Stat Weights"):
                for stat, weight in goalie_weights_config['stat_weights'].items():
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        key = f"goalie_stat_{stat}"
                        if key not in st.session_state:
                            st.session_state[key] = float(weight)
                        st.slider(f"**{stat} Weight**", 0.0, 2.0, st.session_state[key], key=key)
                    with col2:
                        st.button("Reset", key=f"reset_goalie_stat_{stat}", on_click=reset_weights, args=(key, float(weight)))
                
                for category in goalie_weights_config['analyst_weights'].keys():
                    for analyst in goalie_weights_config['analyst_weights'][category].keys():
                        goalie_weights_config['analyst_weights'][category][analyst] = st.session_state[f"goalie_analyst_{category}_{analyst}"]

                for stat in goalie_weights_config['stat_weights'].keys():
                    goalie_weights_config['stat_weights'][stat] = st.session_state[f"goalie_stat_{stat}"]
    
    st.markdown("---")
    
    st.header("Final Ranking Options")
    
    # Store the input in session state for later use
    st.session_state.keeper_list_input = st.text_area(
        "Enter Kept Players (one per line, e.g., Connor McDavid, David Pastrnak)", 
        key="keeper_text_area", 
        height=100
    )
    
    scarcity_enabled = st.checkbox("Enable Position Scarcity", value=False)
    
    if st.button("Generate Final Ranking List"):
        with st.spinner("Generating rankings..."):
            
            # 1. Clean the keeper list
            # The input is now read directly from session state.
            keeper_input = st.session_state.keeper_list_input
            kept_players_lower = {name.strip().lower() for name in keeper_input.split('\n') if name.strip()}
            
            # 2. Skater Prediction and Ranking
            skater_predictions = calculate_weighted_prediction(skater_df, skater_weights_config, 'skater')
            skater_ranked = calculate_zscore_ranking(skater_predictions, skater_weights_config['stat_weights'], 'skater', scarcity_enabled)
            
            # 3. Goalie Prediction and Ranking
            goalie_predictions = calculate_weighted_prediction(goalie_df, goalie_weights_config, 'goalie')
            goalie_ranked = calculate_zscore_ranking(goalie_predictions, goalie_weights_config['stat_weights'], 'goalie', scarcity_enabled)
            
            # 4. Combine and Filter
            final_ranking = pd.concat([skater_ranked, goalie_ranked], ignore_index=True)

            # Create a lowercase version of the player name column for filtering
            final_ranking['PLAYER_LOWER'] = final_ranking['PLAYER'].str.lower()
            
            # Filter out keepers by checking if the lowercase player name is in the set of kept names
            final_ranking = final_ranking[~final_ranking['PLAYER_LOWER'].isin(kept_players_lower)].copy()
            
            # Drop the temporary lowercase column
            final_ranking.drop(columns=['PLAYER_LOWER'], inplace=True)
            
            # 5. Final Sort
            final_ranking = final_ranking.sort_values(by='Final_Ranking_Score', ascending=False)
            
            st.dataframe(final_ranking)
