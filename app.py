import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

st.set_page_config(
    page_title="HK Squash App",
    page_icon="🇭🇰",
    layout="wide"
)

today = pd.Timestamp(datetime.now().date())

divisions = {
    "1": 379,
    "2": 380,
    "3": 381,
    "4": 382,
    "5": 383,
    "7": 384,
    "8": 385,
    "10": 386,
    "11": 387,
    "12B": 388,
    "12A": 389,
    "13": 390,
    "14": 391,
    "15": 392,
    "16": 394,
    "17A": 395,
    "17B": 396,
    "18": 397,
    "19": 398,
    "M1": 399,
    "M2": 400,
    "M3": 401,
    "L1": 402,
    "L2": 403,
    "L3": 404
    }


def load_overall_home_away_data(division):
    with open(f"home_away_data/{division}_overall_scores.csv") as f:
        scores = f.read().split(',')
    return scores


def load_csvs(division):
    try:
        final_table = pd.read_csv(f"simulated_tables/{division}_proj_final_table.csv")
        fixtures = pd.read_csv(f"simulated_fixtures/{division}_proj_fixtures.csv")
        home_away_df = pd.read_csv(f"home_away_data/{division}_team_average_scores.csv")
        team_win_breakdown = pd.read_csv(f"team_win_percentage_breakdown/{division}_team_win_percentage_breakdown.csv")
        awaiting_results = pd.read_csv(f"awaiting_results/{division}_awaiting_results.csv")
        return final_table, fixtures, home_away_df, team_win_breakdown, awaiting_results
    except FileNotFoundError:
        st.error(f"Data not found for division {division}. Please run simulations first.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def main():
    st.title("HK Squash League App")

    # Use columns to control the width of the dropdown boxes
    col1, col2 = st.columns([1, 5])  # Adjust the ratio as needed

    with col1:
        division = st.selectbox("Select a Division:", list(divisions.keys()))

    sections = [
        "Home/Away Splits",
        "Rubber Win Percentage",
        "Projections"
    ]

    with col1:
        sections_box = st.selectbox("Select a section:", sections)

    if st.button("Load Data"):
        simulated_table, simulated_fixtures, home_away_df, team_win_breakdown, awaiting_results = load_csvs(division)

        if sections_box == "Home/Away Splits":
            # Load and display overall scores
            overall_home_away = load_overall_home_away_data(division)

            # Line break
            st.write('<br>', unsafe_allow_html=True)
            st.write("Overall split:")

            # Sizes for the pie chart
            sizes = [float(overall_home_away[0]), float(overall_home_away[1])]

            # Update labels and colors
            labels = ['Home', 'Away']
            colors = ['#ff9999', '#66b3ff']

            # Set font properties to Calibri
            prop = fm.FontProperties(family='Calibri')

            # Create columns using st.columns
            # Adjust the fractions to control the width of each column
            col1, col2 = st.columns([1, 1])

            # Plotting the chart in the first column
            with col1:

                # Create a pie chart with larger font size for labels
                fig, ax = plt.subplots(figsize=(8, 6))
                pie, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='',
                                               textprops={'fontsize': 16, 'fontproperties': prop})

                # Draw a white circle in the middle to create the donut shape
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                fig = plt.gcf()
                fig.gca().add_artist(centre_circle)

                # Place the absolute value texts inside their respective segments
                for i, (slice, value) in enumerate(zip(pie, sizes)):
                    angle = (slice.theta2 + slice.theta1) / 2
                    x, y = slice.r * np.cos(np.deg2rad(angle)), slice.r * np.sin(np.deg2rad(angle))
                    ax.text(x * 0.85, y * 0.85, f'{value:.2f}', ha='center', va='center', fontproperties=prop, fontsize=14,
                            color='black')

                # Add text in the center of the circle
                plt.text(0, 0, f"Home Win\n{float(overall_home_away[2]) * 100:.1f}%", ha='center', va='center',
                         fontproperties=prop,
                         fontsize=14)

                # Add title
                plt.title(f"Average Home/Away Rubbers Won in Division {division}", fontproperties=prop, size=16)

                # Ensure the pie chart is a circle
                ax.axis('equal')
                plt.tight_layout()

                # Display the plot in the Streamlit app
                st.pyplot(fig)

            # Use the second column for other content or leave it empty
            with col2:
                st.write("")  # You can add other content here if needed

            # Line break
            st.write('<br>', unsafe_allow_html=True)
            st.write("Split by team:")

            # Show home/away split by team
            # Rename columns appropriately
            home_away_df = home_away_df.rename(columns={
                'home_away_diff': 'Difference',
                'Home': 'Home Venue',  # Renaming existing 'Home' column to 'Home Venue'
                'Average Home Score': 'Home',
                'Average Away Score': 'Away'
            })

            # Function to format and round the numbers for HTML rendering
            def format_float(val):
                if isinstance(val, float):
                    return f'{val:.2f}'
                return val

            # Apply the format to the 'Home', 'Away', and 'Difference' columns
            columns_to_format = ['Home', 'Away', 'Difference']
            for col in columns_to_format:
                if col in home_away_df.columns:
                    home_away_df[col] = home_away_df[col].apply(format_float)

            # Apply a color gradient using 'Blues' colormap to 'Home' and 'Away' columns
            colormap_blues = 'Blues'
            styled_home_away_df = home_away_df.style.background_gradient(cmap=colormap_blues, subset=['Home', 'Away']) \
                .set_properties(subset=['Home', 'Away'], **{'text-align': 'right'})

            # Apply a color gradient using 'OrRd' colormap to 'Difference' column
            colormap_orrd = 'OrRd'
            if 'Difference' in home_away_df.columns:
                styled_home_away_df = styled_home_away_df.background_gradient(cmap=colormap_orrd, subset=['Difference']) \
                    .set_properties(subset=['Difference'], **{'text-align': 'right'})

            # Hide the index
            styled_home_away_df = styled_home_away_df.hide(axis='index')

            # Display the styled DataFrame in Streamlit
            st.write(styled_home_away_df.to_html(escape=False), unsafe_allow_html=True)

            # Line break
            st.write('<br>', unsafe_allow_html=True)

            # Note
            st.write("Note:  \nMatches where the home team and away team share a home venue are ignored in the calculation")

        elif sections_box == "Rubber Win Percentage":

            # Round values in team_win_breakdown DataFrame
            numeric_cols_team_win_breakdown = team_win_breakdown.select_dtypes(include=['float', 'int']).columns
            team_win_breakdown[numeric_cols_team_win_breakdown] = team_win_breakdown[
                numeric_cols_team_win_breakdown].applymap(lambda x: f'{x:.1f}')

            # Apply a color gradient using 'Blues' colormap to all numeric columns except 'avg_win_perc'
            colormap_blues = 'Blues'
            cols_for_blues_gradient = [col for col in numeric_cols_team_win_breakdown if col != 'avg_win_perc']
            styled_team_win_breakdown = team_win_breakdown.style.background_gradient(cmap=colormap_blues,
                                                                                     subset=cols_for_blues_gradient) \
                .set_properties(subset=cols_for_blues_gradient, **{'text-align': 'right'})

            # Apply a color gradient using 'Oranges' colormap to 'avg_win_perc' column
            colormap_oranges = 'OrRd'
            if 'avg_win_perc' in team_win_breakdown.columns:
                styled_team_win_breakdown = styled_team_win_breakdown.background_gradient(cmap=colormap_oranges,
                                                                                          subset=['avg_win_perc']) \
                    .set_properties(subset=['avg_win_perc'], **{'text-align': 'right'})

            # Hide the index
            styled_team_win_breakdown = styled_team_win_breakdown.hide(axis='index')

            # Line break and display DataFrame in Streamlit
            st.write('<br>', unsafe_allow_html=True)
            st.write("Team win percentage by rubber:")
            st.write(styled_team_win_breakdown.to_html(escape=False), unsafe_allow_html=True)

        elif sections_box == "Projections":

            if len(awaiting_results) > 0:
                # Line break
                st.write('<br>', unsafe_allow_html=True)
                st.write("Still awaiting these results:")
                st.dataframe(awaiting_results, hide_index=True)

            # Convert the "Match Week" column to integers
            simulated_fixtures['Match Week'] = simulated_fixtures['Match Week'].astype(int)

            # Adjust the "Date" column format
            simulated_fixtures['Date'] = pd.to_datetime(simulated_fixtures['Date']).dt.date

            # Round values in simulated_fixtures DataFrame except for "Match Week"
            numeric_cols_simulated_fixtures = simulated_fixtures.select_dtypes(include=['float', 'int']).columns.drop(
                'Match Week')
            simulated_fixtures[numeric_cols_simulated_fixtures] = simulated_fixtures[
                numeric_cols_simulated_fixtures].applymap(lambda x: f'{x:.2f}')

            # Create dataframe to show next round of fixtures
            next_round_of_fixtures = simulated_fixtures[simulated_fixtures["Date"] > today.date() + pd.Timedelta(days=-1)].head(
                len(simulated_table) // 2)

            # Apply a color gradient using a colormap to numeric columns except for "Matchweek"
            colormap = 'Blues'
            styled_next_round_of_fixtures = next_round_of_fixtures.style.background_gradient(cmap=colormap,
                                                                                             subset=numeric_cols_simulated_fixtures) \
                .set_properties(subset=numeric_cols_simulated_fixtures, **{'text-align': 'right'}) \
                .hide(axis='index')

            # Display the styled DataFrame in Streamlit
            st.write("Next Round of Fixtures:")
            st.write(styled_next_round_of_fixtures.to_html(escape=False), unsafe_allow_html=True)

            if not simulated_table.empty:
                # Line break
                st.write('<br>', unsafe_allow_html=True)
                st.write("Simulated Final Table:")

                # Convert 'Played' column to integers
                if 'Played' in simulated_table.columns:
                    simulated_table['Played'] = simulated_table['Played'].astype(int)

                # Round values in simulated_table DataFrame except for 'Played'
                numeric_cols_simulated_table = simulated_table.select_dtypes(include=['float', 'int']).columns
                cols_to_round = numeric_cols_simulated_table.drop('Played')
                simulated_table[cols_to_round] = simulated_table[cols_to_round].applymap(lambda x: f'{x:.1f}')

                # Columns to exclude from gradient formatting
                cols_to_exclude = {'Played', 'Won', 'Lost', 'Points', 'Playoffs'}
                cols_for_blues_gradient = [col for col in cols_to_round if col not in cols_to_exclude]

                # Apply a color gradient using 'Blues' colormap to selected numeric columns
                colormap_blues = 'Blues'
                styled_simulated_table = simulated_table.style.background_gradient(cmap=colormap_blues,
                                                                                   subset=cols_for_blues_gradient) \
                    .set_properties(subset=cols_for_blues_gradient, **{'text-align': 'right'})

                # Apply a color gradient using 'OrRd' colormap to 'Playoffs' column
                colormap_orrd = 'OrRd'
                if 'Playoffs' in simulated_table.columns:
                    styled_simulated_table = styled_simulated_table.background_gradient(cmap=colormap_orrd,
                                                                                        subset=['Playoffs']) \
                        .set_properties(subset=['Playoffs'], **{'text-align': 'right'})

                # Hide the index
                styled_simulated_table = styled_simulated_table.hide(axis='index')

                # Display the styled DataFrame in Streamlit
                st.write(styled_simulated_table.to_html(escape=False), unsafe_allow_html=True)

if __name__ == "__main__":
    main()