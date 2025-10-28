"""
Validator for player details data.
"""

import pandas as pd
from .base import BaseValidator, ValidationResult


class PlayersValidator(BaseValidator):
    """Validates player details data."""
    
    REQUIRED_COLUMNS = {'Order', 'Player', 'HKS No.', 'Ranking', 'Points', 'Team'}
    EXPECTED_MIN_PLAYERS = 20
    EXPECTED_MAX_PLAYERS = 200
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate players DataFrame."""
        result = ValidationResult(
            validator_name='PlayersValidator',
            data_type='players',
            is_valid=True
        )
        
        # Store metadata
        result.metadata['league_id'] = self.league_id
        result.metadata['year'] = self.year
        if self.division:
            result.metadata['division'] = self.division
        
        # Check not empty
        if not self._check_not_empty(df, result):
            return result
        
        result.metadata['row_count'] = len(df)
        
        # Check required columns
        if not self._check_required_columns(df, self.REQUIRED_COLUMNS, result):
            return result
        
        # Check no null values in critical columns
        critical_columns = {'Player', 'Team'}
        self._check_no_null_values(df, critical_columns, result)
        
        # Check data types
        expected_types = {
            'Order': 'int',
            'Player': 'str',
            'HKS No.': 'int',
            'Ranking': 'int',
            'Points': 'float',
            'Team': 'str',
        }
        self._check_data_types(df, expected_types, result)
        
        # Check row count is reasonable
        self._check_row_count_range(df, self.EXPECTED_MIN_PLAYERS,
                                    self.EXPECTED_MAX_PLAYERS, result)
        
        # Check value ranges
        if 'Order' in df.columns:
            # Allow up to 15 players per team (some teams have more than 10)
            self._check_value_range(df, 'Order', 1, 15, result)
            
            # Check Order distribution per team
            if 'Team' in df.columns:
                for team in df['Team'].unique():
                    team_players = df[df['Team'] == team]
                    orders = team_players['Order'].sort_values().tolist()
                    expected_orders = list(range(1, len(team_players) + 1))
                    
                    if orders != expected_orders:
                        result.add_warning('Order',
                            f'Team "{team}" has non-sequential orders: {orders}')
        
        if 'HKS No.' in df.columns:
            # Check for missing HKS numbers
            missing_hks = df['HKS No.'].isna().sum()
            if missing_hks > 0:
                result.add_warning('HKS No.',
                    f'{missing_hks} players missing HKS number')
            
            # Check for reasonable HKS number range
            valid_hks = df['HKS No.'].dropna()
            if len(valid_hks) > 0:
                self._check_value_range(df, 'HKS No.', 1, 99999, result)
                
                # Check for duplicate HKS numbers
                duplicates = df['HKS No.'].dropna().duplicated().sum()
                if duplicates > 0:
                    result.add_error('HKS No.',
                        f'{duplicates} duplicate HKS numbers (same player on multiple teams?)')
        
        if 'Ranking' in df.columns:
            missing_ranking = df['Ranking'].isna().sum()
            if missing_ranking > 0:
                result.add_info('Ranking',
                    f'{missing_ranking} players have no ranking')
            
            # Allow 0 for unranked players, and check for reasonable range
            valid_rankings = df['Ranking'].dropna()
            if len(valid_rankings) > 0:
                self._check_value_range(df, 'Ranking', 0, 10000, result)
        
        if 'Points' in df.columns:
            missing_points = df['Points'].isna().sum()
            if missing_points > 0:
                result.add_info('Points',
                    f'{missing_points} players have no points')
            
            valid_points = df['Points'].dropna()
            if len(valid_points) > 0:
                # Allow up to 15 points (some players have higher points)
                self._check_value_range(df, 'Points', 0.0, 15.0, result)
        
        # Check for duplicate players
        if 'Player' in df.columns:
            duplicates = df['Player'].duplicated().sum()
            if duplicates > 0:
                result.add_warning('Player',
                    f'{duplicates} duplicate player names')
                
                # Show which players are duplicated
                dup_players = df[df['Player'].duplicated(keep=False)]['Player'].unique()
                if len(dup_players) <= 5:  # Only show if not too many
                    result.add_info('Player',
                        f'Duplicated players: {dup_players.tolist()}')
        
        # Team distribution
        if 'Team' in df.columns:
            team_counts = df['Team'].value_counts()
            result.add_info('Team',
                f'{team_counts.nunique()} unique teams, avg {team_counts.mean():.1f} players/team')
            
            # Check for teams with unusual player counts
            small_teams = team_counts[team_counts < 3]
            if len(small_teams) > 0:
                result.add_warning('Team',
                    f'{len(small_teams)} teams have < 3 players')
            
            large_teams = team_counts[team_counts > 8]
            if len(large_teams) > 0:
                result.add_warning('Team',
                    f'{len(large_teams)} teams have > 8 players')
        
        # Check Player-HKS No. uniqueness
        if 'Player' in df.columns and 'HKS No.' in df.columns:
            # Group by player and check if they have multiple HKS numbers
            player_hks = df.dropna(subset=['HKS No.']).groupby('Player')['HKS No.'].nunique()
            multiple_hks = player_hks[player_hks > 1]
            
            if len(multiple_hks) > 0:
                result.add_error('Player-HKS',
                    f'{len(multiple_hks)} players have multiple HKS numbers')
        
        self._log_result(result)
        return result
