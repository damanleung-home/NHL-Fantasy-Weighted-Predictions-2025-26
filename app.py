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
    categories = ['GP', 'G', 'A', '+/-', 'PIM', 'PPP', 'SHP', 'SOG', 'FOW', 'HIT', 'BLK'] if position == 'skater' else ['GP', 'W', 'L', 'GAA', 'SV', 'SV%', 'SO']
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
        is_negative_stat = (position == 'goalie' and category in ['L', 'GAA'])
        
        predicted_col = f'Weighted_Predicted_{category}'
        if is_negative_stat:
            df[f'Z_score_{category}'] = zscore(df[predicted_col]) * -1
        else:
            df[f'Z_score_{category}'] = zscore(df[predicted_col])

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

    # Callback function to manage percentage sliders
    def update_analyst_weights_percent(category, analyst_changed, new_value):
        current_weights = st.session_state[f"analyst_weights_{category}"]
        
        all_analysts = list(current_weights.keys())
        other_analysts = [analyst for analyst in all_analysts if analyst != analyst_changed]
        
        old_sum_others = sum(current_weights[analyst] for analyst in other_analysts)
        
        if old_sum_others > 0:
            remaining_pool = 100 - new_value
            for analyst in other_analysts:
                old_weight = current_weights[analyst]
                current_weights[analyst] = (old_weight / old_sum_others) * remaining_pool
        else:
            remaining_pool = 100 - new_value
            if len(other_analysts) > 0:
                for analyst in other_analysts:
                    current_weights[analyst] = remaining_pool / len(other_analysts)
        
        current_weights[analyst_changed] = new_value
        st.session_state[f"analyst_weights_{category}"] = current_weights

    # Initialize session state for weights if not present
    if 'skater_analyst_weights' not in st.session_state:
        st.session_state['skater_analyst_weights'] = {cat: {a: w * 100 for a, w in skater_weights_config['analyst_weights'][cat].items()} for cat in skater_weights_config['analyst_weights'].keys()}
    if 'goalie_analyst_weights' not in st.session_state:
        st.session_state['goalie_analyst_weights'] = {cat: {a: w * 100 for a, w in goalie_weights_config['analyst_weights'][cat].items()} for cat in goalie_weights_config['analyst_weights'].keys()}

    st.header("Customize Skater Rankings")
    
    skater_df = skater_df_raw[skater_df_raw['season'] == '2025-2026'].copy()
    
    with st.expander("Analyst Weights (Percentage-Based)"):
        skater_analyst_weights_ui = st.session_state.skater_analyst_weights
        for category in skater_analyst_weights_ui.keys():
            st.write(f"**{category}**")
            cols = st.columns(len(analyst_names))
            for i, (analyst, weight) in enumerate(skater_analyst_weights_ui[category].items()):
                with cols[i]:
                    st.slider(f"{analyst_names.get(analyst, analyst)}", 0.0, 100.0, weight, key=f"skater_analyst_{category}_{analyst}", on_change=update_analyst_weights_percent, args=(category, analyst, st.session_state[f"skater_analyst_{category}_{analyst}"]))

    st.subheader("Stat Weights")
    skater_stat_weights_ui = {}
    for stat, weight in skater_weights_config['stat_weights'].items():
        skater_stat_weights_ui[stat] = st.slider(f"{stat} Weight", 0.0, 5.0, float(weight), key=f"skater_stat_{stat}")
    
    for category in skater_weights_config['analyst_weights'].keys():
        for analyst in skater_weights_config['analyst_weights'][category].keys():
            skater_weights_config['analyst_weights'][category][analyst] = skater_analyst_weights_ui[(category, analyst)]
            
    for stat in skater_weights_config['stat_weights'].keys():
        skater_weights_config['stat_weights'][stat] = skater_stat_weights_ui[stat]

    st.markdown("---")

    st.header("Customize Goalie Rankings")
    
    goalie_df = goalie_df_raw[goalie_df_raw['season'] == '2025-2026'].copy()

    st.subheader("Analyst Weights")
    goalie_analyst_weights_ui = {}
    for category in goalie_weights_config['analyst_weights'].keys():
        st.write(f"**{category}**")
        for analyst, weight in goalie_analyst_weights_ui[category].items():
            col1, col2 = st.columns(len(analyst_names))
            with col1:
                key = f"goalie_analyst_{category}_{analyst}"
                if key not in st.session_state:
                    st.session_state[key] = float(weight)
                goalie_analyst_weights_ui[(category, analyst)] = st.slider(f"{analyst_names.get(analyst, analyst)} Weight", 0.0, 100.0, weight, key=key)
            with col2:
                st.button("Reset", key=f"reset_goalie_analyst_{category}_{analyst}", on_click=reset_weights, args=(key, float(weight)))

    st.subheader("Stat Weights")
    goalie_stat_weights_ui = {}
    for stat, weight in goalie_weights_config['stat_weights'].items():
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            key = f"goalie_stat_{stat}"
            if key not in st.session_state:
                st.session_state[key] = float(weight)
            goalie_stat_weights_ui[stat] = st.slider(f"{stat} Weight", 0.0, 5.0, st.session_state[key], key=key)
        with col2:
            st.button("Reset", key=f"reset_goalie_stat_{stat}", on_click=reset_weights, args=(key, float(weight)))
    
    for category in goalie_weights_config['analyst_weights'].keys():
        for analyst in goalie_weights_config['analyst_weights'][category].keys():
            goalie_weights_config['analyst_weights'][category][analyst] = goalie_analyst_weights_ui[(category, analyst)]

    for stat in goalie_weights_config['stat_weights'].keys():
        goalie_weights_config['stat_weights'][stat] = goalie_stat_weights_ui[stat]
    
    st.markdown("---")
    
    st.header("Final Ranking Options")
    scarcity_enabled = st.checkbox("Enable Position Scarcity", value=False)
    
    if st.button("Generate Final Ranking List"):
        with st.spinner("Generating rankings..."):
            skater_predictions = calculate_weighted_prediction(skater_df, skater_weights_config, 'skater')
            skater_ranked = calculate_zscore_ranking(skater_predictions, skater_weights_config['stat_weights'], 'skater', scarcity_enabled)
            
            goalie_predictions = calculate_weighted_prediction(goalie_df, goalie_weights_config, 'goalie')
            goalie_ranked = calculate_zscore_ranking(goalie_predictions, goalie_weights_config['stat_weights'], 'goalie', scarcity_enabled)
            
            final_ranking = pd.concat([skater_ranked, goalie_ranked], ignore_index=True)
            final_ranking = final_ranking.sort_values(by='Final_Ranking_Score', ascending=False)
            
            st.dataframe(final_ranking)
