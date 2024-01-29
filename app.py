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
        overall_home_away = pd.read_csv(f"home_away_data/{division}_overall_scores.csv", header=None)
        final_table = pd.read_csv(f"simulated_tables/{division}_proj_final_table.csv")
        fixtures = pd.read_csv(f"simulated_fixtures/{division}_proj_fixtures.csv")
        home_away_df = pd.read_csv(f"home_away_data/{division}_team_average_scores.csv")
        team_win_breakdown_overall = pd.read_csv(
            f"team_win_percentage_breakdown/Overall/{division}_team_win_percentage_breakdown.csv")
        team_win_breakdown_home = pd.read_csv(
            f"team_win_percentage_breakdown/Home/{division}_team_win_percentage_breakdown_home.csv")
        team_win_breakdown_away = pd.read_csv(
            f"team_win_percentage_breakdown/Away/{division}_team_win_percentage_breakdown_away.csv")
        team_win_breakdown_delta = pd.read_csv(
            f"team_win_percentage_breakdown/Delta/{division}_team_win_percentage_breakdown_delta.csv")
        awaiting_results = pd.read_csv(f"awaiting_results/{division}_awaiting_results.csv")
        detailed_league_table = pd.read_csv(f"detailed_league_tables/{division}_detailed_league_table.csv")
        return (final_table, fixtures, home_away_df, team_win_breakdown_overall, team_win_breakdown_home,
                team_win_breakdown_away, team_win_breakdown_delta, awaiting_results, detailed_league_table,
                overall_home_away
                )
    except FileNotFoundError as e:
        st.error(f"Data not found for division {division}. Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), \
            pd.DataFrame(), pd.DataFrame()


def main():
    st.title("HK Squash League App")

    with st.sidebar:
        division = st.selectbox("**Select a Division:**", list(divisions.keys()))

        sections = [
            "Detailed Division Table",
            "Home/Away Splits",
            "Rubber Win Percentage",
            "Projections"
        ]

        sections_box = st.selectbox("**Select a section:**", sections)

        about = st.expander("**About**")
        about.write(
            """The aim of this application is to take publicly available data from 
            [hksquash.org.hk](https://www.hksquash.org.hk/public/index.php/leagues/index/league/Squash/pages_id/25.html)
            and provide insights to players and convenors involved in the Hong Kong Squash League.
            \nThis application is not affiliated with the Squash Association of Hong Kong, China.""")

        contact = st.expander("**Contact**")
        contact.write("For any queries, email bpalitherland@gmail.com")

    if 'data_loaded' not in st.session_state or st.session_state['current_division'] != division:
        # Load data and store in session state
        st.session_state['data'] = load_csvs(division)
        st.session_state['data_loaded'] = True
        st.session_state['current_division'] = division

    # Retrieve data from session state
    simulated_table, simulated_fixtures, home_away_df, team_win_breakdown_overall, team_win_breakdown_home, \
        team_win_breakdown_away, team_win_breakdown_delta, awaiting_results, \
        detailed_league_table, overall_home_away = st.session_state['data']

    simulation_date = pd.to_datetime(overall_home_away.iloc[0, 4]).strftime('%Y-%m-%d')
    date = pd.to_datetime(overall_home_away.iloc[0, 3]).strftime('%Y-%m-%d')

    if sections_box == "Detailed Division Table":
        # Header
        st.header("Detailed Division Table")
        st.write(f"**Last updated:** {date}")

        if len(awaiting_results) > 0:
            # Line break
            st.write('<br>', unsafe_allow_html=True)
            st.subheader("Still awaiting these results:")
            styled_awaiting_results = awaiting_results.style.hide(axis='index')
            st.write(styled_awaiting_results.to_html(escape=False), unsafe_allow_html=True)
            st.write('<br>', unsafe_allow_html=True)

        # Line break
        st.write('<br>', unsafe_allow_html=True)

        # Display the styled DataFrame in Streamlit
        # st.write(detailed_league_table.to_html(escape=False), unsafe_allow_html=True)

        # Apply styles to the DataFrame
        styled_df = detailed_league_table.style.set_properties(**{'text-align': 'right'}).hide(axis='index')
        styled_df = styled_df.set_properties(subset=['Team'], **{'text-align': 'left'})

        # Convert styled DataFrame to HTML
        html = styled_df.to_html(escape=False)

        # Display in Streamlit
        st.write(html, unsafe_allow_html=True)

        # Note
        st.write('<br>', unsafe_allow_html=True)
        st.write("**Note:**  \nCR stands for Conceded Rubber.  \nWO stands for Walkover. Teams are penalized \
                 one point for each walkover given.")

    elif sections_box == "Home/Away Splits":

        # Header
        st.header("Home/Away Splits")
        # Load and display overall scores
        overall_home_away = load_overall_home_away_data(division)

        # Line break
        st.write('<br>', unsafe_allow_html=True)
        st.subheader("Overall split:")

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
        st.subheader("Split by team:")

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
        #columns_to_format = ['Home', 'Away', 'Difference']
        #for col in columns_to_format:
        #    if col in home_away_df.columns:
        #        home_away_df[col] = home_away_df[col].apply(format_float)

        # Ensure that the 'Home' and 'Away' columns are numeric
        home_away_df[['Home', 'Away', "Difference"]] = (home_away_df[['Home', 'Away', "Difference"]]
                                                        .apply(pd.to_numeric, errors='coerce'))

        # Determine the range for the colormap
        vmin = home_away_df[['Home', 'Away']].min().min()
        vmax = home_away_df[['Home', 'Away']].max().max()

        # Apply a color gradient using 'Blues' colormap to 'Home' and 'Away' columns
        colormap_blues = 'Blues'
        styled_home_away_df = (home_away_df.style.background_gradient(
            cmap=colormap_blues,
            vmin=vmin,
            vmax=vmax,
            subset=['Home', 'Away']
        ).set_properties(subset=['Home', 'Away'], **{'text-align': 'right'})
                               .format("{:.2f}", subset=['Home', 'Away', 'Difference']))

        # Apply a color gradient using 'OrRd' colormap to 'Difference' column
        colormap_orrd = 'OrRd'
        if 'Difference' in home_away_df.columns:
            styled_home_away_df = (styled_home_away_df.background_gradient(
                cmap=colormap_orrd,
                subset=['Difference']
            ).set_properties(subset=['Difference'], **{'text-align': 'right'}))

        # Hide the index
        styled_home_away_df = styled_home_away_df.hide(axis='index')

        # Display the styled DataFrame in Streamlit
        st.write(styled_home_away_df.to_html(escape=False), unsafe_allow_html=True)

        # Line break
        st.write('<br>', unsafe_allow_html=True)

        # Note
        st.write("**Note:**  \nMatches where the home team and away team share a \
        home venue are ignored in the calculation")

    elif sections_box == "Rubber Win Percentage":

        # Header
        st.header("Rubber Win Percentage")

        # Function to apply common formatting
        def format_dataframe(df):
            if df is not None and not df.empty:

                # Rename avg_win_perc column
                df = df.rename(columns={"avg_win_perc": "Average"})
                # Select only numeric columns for vmin and vmax calculation
                numeric_cols_raw = [col for col in df.columns if 'Win' in col]

                # Convert these columns to numeric type, handling non-numeric values
                for col in numeric_cols_raw:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Determine the range for the colormap
                vmin = df[numeric_cols_raw].min().min()
                vmax = df[numeric_cols_raw].max().max()

                # Format numeric columns for display
                def format_float(x):
                    try:
                        return f'{float(x):.1f}'
                    except ValueError:
                        return x

                df[numeric_cols_raw + ["Average"]] = df[numeric_cols_raw + ["Average"]].map(format_float)

                # Check if "Total Rubbers" is in the DataFrame and format it as integer
                if 'Total Rubbers' in df.columns:
                    df['Total Rubbers'] = df['Total Rubbers'].astype(int)

                colormap_blues = 'Blues'
                cols_for_blues_gradient = numeric_cols_raw
                styled_df = df.style.background_gradient(
                    cmap=colormap_blues,
                    vmin=vmin,
                    vmax=vmax,
                    subset=cols_for_blues_gradient
                ).set_properties(subset=cols_for_blues_gradient, **{'text-align': 'right'})

                # Set right alignment for "Total Rubbers"
                if 'Total Rubbers' in df.columns:
                    styled_df = styled_df.set_properties(subset=['Total Rubbers'], **{'text-align': 'right'})

                colormap_oranges = 'OrRd'
                if 'Average' in df.columns:
                    styled_df = styled_df.background_gradient(
                        cmap=colormap_oranges,
                        subset=['Average']
                    ).set_properties(subset=['Average'], **{'text-align': 'right'})

                styled_df = styled_df.hide(axis='index')
                return styled_df
            else:
                return "DataFrame is empty or not loaded."


        def format_dataframe_delta(df):
            if df is not None and not df.empty:

                # Rename avg_win_perc column
                df = df.rename(columns={"avg_win_perc": "Average"})
                # Select only numeric columns for vmin and vmax calculation
                numeric_cols_raw = [col for col in df.columns if 'Win' in col]

                # Convert these columns to numeric type, handling non-numeric values
                for col in numeric_cols_raw:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Determine the range for the colormap
                vmin = df[numeric_cols_raw].min().min()
                vmax = df[numeric_cols_raw].max().max()

                # Format numeric columns for display
                def format_float(x):
                    try:
                        return f'{float(x):.1f}'
                    except ValueError:
                        return x

                df[numeric_cols_raw + ["Average"]] = df[numeric_cols_raw + ["Average"]].map(format_float)

                # Check if "Total Rubbers" is in the DataFrame and format it as integer
                if 'Total Rubbers' in df.columns:
                    df['Total Rubbers'] = df['Total Rubbers'].astype(int)

                colormap_blues = 'RdYlBu'
                cols_for_blues_gradient = numeric_cols_raw
                styled_df = df.style.background_gradient(
                    cmap=colormap_blues,
                    vmin=vmin,
                    vmax=vmax,
                    subset=cols_for_blues_gradient
                ).set_properties(subset=cols_for_blues_gradient, **{'text-align': 'right'})

                # Set right alignment for "Total Rubbers"
                if 'Total Rubbers' in df.columns:
                    styled_df = styled_df.set_properties(subset=['Total Rubbers'], **{'text-align': 'right'})

                colormap_oranges = 'RdYlBu'
                if 'Average' in df.columns:
                    styled_df = styled_df.background_gradient(
                        cmap=colormap_oranges,
                        subset=['Average']
                    ).set_properties(subset=['Average'], **{'text-align': 'right'})

                styled_df = styled_df.hide(axis='index')
                return styled_df
            else:
                return "DataFrame is empty or not loaded."

        # Radio button for user to choose the DataFrame
        option = st.radio("Select Team Win Breakdown View:", ['Overall', 'Home', 'Away', 'H/A Delta'], horizontal=True)

        st.write('<br>', unsafe_allow_html=True)
        st.subheader("Team win percentage by rubber:")

        # Apply formatting and display the selected DataFrame
        dataframes = {
            'Overall': team_win_breakdown_overall,
            'Home': team_win_breakdown_home,
            'Away': team_win_breakdown_away,
            'H/A Delta': team_win_breakdown_delta
        }

        selected_df = dataframes.get(option)
        if selected_df is not None:
            if selected_df is team_win_breakdown_delta:
                st.write(format_dataframe_delta(selected_df).to_html(escape=False), unsafe_allow_html=True)
            else:
                st.write(format_dataframe(selected_df).to_html(escape=False), unsafe_allow_html=True)
        else:
            st.error("Selected DataFrame is not available.")

        # Note
        st.write('<br>', unsafe_allow_html=True)
        st.write(
            "**Note:**  \nOnly rubbers that were played are included. Conceded Rubbers \
            and Walkovers are ignored.  \nMatches where the home team and away team share \
             a home venue are ignored in the Home and Away tables.")

    elif sections_box == "Projections":

        # Load and display overall scores
        st.header("Projections")
        st.write(f"**Date of last simulation:** {simulation_date}")

        if len(awaiting_results) > 0:
            # Line break
            st.write('<br>', unsafe_allow_html=True)
            st.subheader("Still awaiting these results:")
            styled_awaiting_results = awaiting_results.style.hide(axis='index')
            st.write(styled_awaiting_results.to_html(escape=False), unsafe_allow_html=True)
            st.write('<br>', unsafe_allow_html=True)
            st.write('<br>', unsafe_allow_html=True)

        # Convert the "Match Week" column to integers
        simulated_fixtures['Match Week'] = simulated_fixtures['Match Week'].astype(int)

        # Adjust the "Date" column format
        simulated_fixtures['Date'] = pd.to_datetime(simulated_fixtures['Date']).dt.date

        # Rename columns
        simulated_fixtures = simulated_fixtures.rename(columns={
            "Avg Simulated Home Points": "Proj. Home Pts",
            "Avg Simulated Away Points": "Proj. Away Pts"
        })

        # Round values in simulated_fixtures DataFrame except for "Match Week"
        numeric_cols_simulated_fixtures = simulated_fixtures.select_dtypes(include=['float', 'int']).columns.drop(
            'Match Week')
        simulated_fixtures[numeric_cols_simulated_fixtures] = simulated_fixtures[
            numeric_cols_simulated_fixtures].map(lambda x: f'{x:.2f}')

        # Ensure the columns are numeric for vmin and vmax calculation
        simulated_fixtures_numeric = simulated_fixtures.copy()
        simulated_fixtures_numeric[numeric_cols_simulated_fixtures] = simulated_fixtures_numeric[
            numeric_cols_simulated_fixtures].apply(pd.to_numeric, errors='coerce')

        # Get the range of match weeks
        min_week = simulated_fixtures_numeric['Match Week'].min()
        max_week = simulated_fixtures_numeric['Match Week'].max()

        # Create a slider for match weeks
        col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed
        with col1:
            selected_week = st.slider("**Select Match Week:**", min_week, max_week, value=min_week, step=1)

        # Filter the fixtures based on the selected match week
        filtered_fixtures = simulated_fixtures_numeric[simulated_fixtures_numeric["Match Week"] == selected_week]

        # Determine the range for the colormap
        vmin = filtered_fixtures[["Proj. Home Pts", "Proj. Away Pts"]].min().min()
        vmax = filtered_fixtures[["Proj. Home Pts", "Proj. Away Pts"]].max().max()

        # Apply styling to the filtered DataFrame
        styled_filtered_fixtures = (
            filtered_fixtures.style.background_gradient(
                cmap='Blues',
                vmin=vmin,
                vmax=vmax,
                subset=numeric_cols_simulated_fixtures)
            .set_properties(subset=numeric_cols_simulated_fixtures, **{'text-align': 'right'})
            .format("{:.2f}", subset=["Proj. Home Pts", "Proj. Away Pts"])
            .hide(axis='index'))

        # Display the styled DataFrame in Streamlit
        # st.write('<br>', unsafe_allow_html=True)
        st.subheader(f"Projected Fixtures for Match Week {selected_week}:")
        st.write(styled_filtered_fixtures.to_html(escape=False), unsafe_allow_html=True)

        if not simulated_table.empty:
            # Line break and subheader
            st.write('<br>', unsafe_allow_html=True)
            st.subheader("Projected Final Table:")

            # Convert 'Played' column to integers
            if 'Played' in simulated_table.columns:
                simulated_table['Played'] = simulated_table['Played'].astype(int)

            # Round values in simulated_table DataFrame except for 'Played'
            numeric_cols_simulated_table = simulated_table.select_dtypes(include=['float', 'int']).columns
            cols_to_round = numeric_cols_simulated_table.drop('Played')

            # Columns to exclude from gradient formatting
            cols_to_exclude = {'Played', 'Won', 'Lost', 'Points', 'Playoffs'}
            cols_for_blues_gradient = [col for col in cols_to_round if col not in cols_to_exclude]

            # Determine the range for the colormap
            vmin = simulated_table[cols_for_blues_gradient].min().min()
            vmax = simulated_table[cols_for_blues_gradient].max().max()

            # Apply a color gradient using 'Blues' colormap to selected numeric columns
            styled_simulated_table = simulated_table.style.background_gradient(
                cmap='Blues',
                vmin=vmin,
                vmax=vmax,
                subset=cols_for_blues_gradient
            ).set_properties(subset=cols_for_blues_gradient + ["Played", "Won", "Lost"], **{'text-align': 'right'})

            # Apply a color gradient using 'OrRd' colormap to 'Playoffs' column
            if 'Playoffs' in simulated_table.columns:
                styled_simulated_table = styled_simulated_table.background_gradient(
                    cmap='OrRd', subset=['Playoffs']
                ).set_properties(subset=['Playoffs'], **{'text-align': 'right'})

            # Apply bar chart formatting to the 'Points' column
            styled_simulated_table = styled_simulated_table.bar(
                subset=['Points'], color='#87CEEB'
            )

            # Round all numeric columns
            styled_simulated_table = styled_simulated_table.format("{:.1f}", subset=cols_to_round)

            # Apply custom formatting for zero values in cols_for_blues_gradient
            styled_simulated_table = styled_simulated_table.format(
                lambda x: f"<span style='color: #f7fbff;'>{x:.1f}</span>" if x == 0 else f"{x:.1f}",
                subset=cols_for_blues_gradient
            )

            # Hide the index
            styled_simulated_table = styled_simulated_table.hide(axis='index')

            # Display the styled DataFrame in Streamlit
            st.write(styled_simulated_table.to_html(escape=False), unsafe_allow_html=True)

            # Note
            st.write('<br>', unsafe_allow_html=True)
            st.write("**Note:**  \nThe projected final table is the average result of simulating the remaining \
                     fixtures 5,000 times.  \nFixtures are simulated using teams' average rubber win percentage, \
                     factoring in home advantage.")


def generate_styled_html(df, numeric_cols, blues_cols, orrd_cols):
    styled_df = df.copy()
    styled_df[numeric_cols] = styled_df[numeric_cols].map(lambda x: f'{x:.2f}')

    # Apply 'Blues' gradient
    styled_df = styled_df.style.background_gradient(cmap='Blues', subset=blues_cols) \
        .set_properties(subset=blues_cols, **{'text-align': 'right'})

    # Apply 'OrRd' gradient
    if orrd_cols:
        styled_df = styled_df.background_gradient(cmap='OrRd', subset=orrd_cols) \
            .set_properties(subset=orrd_cols, **{'text-align': 'right'})

    styled_df = styled_df.hide(axis='index')
    return styled_df.to_html(escape=False)

if __name__ == "__main__":
    main()
