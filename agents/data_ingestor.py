"""
DataIngestorAgent
Role: Fetch, validate, and normalize data from multiple NHL stat sources
Inputs: Date range, team identifiers, season parameters
Outputs: Normalized DataFrames for teams, goalies, schedule, odds
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import time
import json


@dataclass
class DataConfig:
    """Configuration for data ingestion"""
    seasons: List[str] = None  # e.g., ['2024-25', '2023-24', '2022-23']
    cache_ttl: int = 3600      # Cache TTL in seconds
    request_delay: float = 1.0  # Delay between requests (rate limiting)
    user_agent: str = "Mozilla/5.0 (NHL Prediction Model)"

    def __post_init__(self):
        if self.seasons is None:
            self.seasons = ['2024-25', '2023-24', '2022-23']


class DataIngestor:
    """
    Multi-source NHL data ingestion agent.

    Sources:
    - Natural Stat Trick: Advanced metrics (xG, Corsi, scoring chances)
    - MoneyPuck: Goalie stats, GSAX, 60-min metrics
    - Hockey-Reference: Historical data, standings
    - The Odds API: Live betting lines
    """

    # Team name mappings (different sources use different abbreviations)
    TEAM_MAPPING = {
        'ANA': ['ANA', 'Anaheim', 'Ducks'],
        'ARI': ['ARI', 'Arizona', 'Coyotes'],
        'BOS': ['BOS', 'Boston', 'Bruins'],
        'BUF': ['BUF', 'Buffalo', 'Sabres'],
        'CGY': ['CGY', 'CAL', 'Calgary', 'Flames'],
        'CAR': ['CAR', 'Carolina', 'Hurricanes'],
        'CHI': ['CHI', 'Chicago', 'Blackhawks'],
        'COL': ['COL', 'Colorado', 'Avalanche'],
        'CBJ': ['CBJ', 'Columbus', 'Blue Jackets'],
        'DAL': ['DAL', 'Dallas', 'Stars'],
        'DET': ['DET', 'Detroit', 'Red Wings'],
        'EDM': ['EDM', 'Edmonton', 'Oilers'],
        'FLA': ['FLA', 'Florida', 'Panthers'],
        'LAK': ['LAK', 'L.A', 'LA', 'Los Angeles', 'Kings'],
        'MIN': ['MIN', 'Minnesota', 'Wild'],
        'MTL': ['MTL', 'MON', 'Montreal', 'Canadiens'],
        'NSH': ['NSH', 'Nashville', 'Predators'],
        'NJD': ['NJD', 'N.J', 'NJ', 'New Jersey', 'Devils'],
        'NYI': ['NYI', 'NY Islanders', 'Islanders'],
        'NYR': ['NYR', 'NY Rangers', 'Rangers'],
        'OTT': ['OTT', 'Ottawa', 'Senators'],
        'PHI': ['PHI', 'Philadelphia', 'Flyers'],
        'PIT': ['PIT', 'Pittsburgh', 'Penguins'],
        'SJS': ['SJS', 'S.J', 'SJ', 'San Jose', 'Sharks'],
        'SEA': ['SEA', 'Seattle', 'Kraken'],
        'STL': ['STL', 'St. Louis', 'St Louis', 'Blues'],
        'TBL': ['TBL', 'T.B', 'TB', 'Tampa Bay', 'Lightning'],
        'TOR': ['TOR', 'Toronto', 'Maple Leafs'],
        'UTA': ['UTA', 'Utah', 'Utah HC'],
        'VAN': ['VAN', 'Vancouver', 'Canucks'],
        'VGK': ['VGK', 'VEG', 'Vegas', 'Golden Knights'],
        'WSH': ['WSH', 'WAS', 'Washington', 'Capitals'],
        'WPG': ['WPG', 'Winnipeg', 'Jets'],
    }

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self._cache = {}
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': self.config.user_agent})

    def normalize_team_name(self, name: str) -> str:
        """Normalize team name to standard 3-letter code."""
        name = name.strip()
        for code, aliases in self.TEAM_MAPPING.items():
            if name in aliases or name.upper() == code:
                return code
        return name  # Return as-is if no mapping found

    def _fetch_with_cache(self, url: str, parser: str = 'json') -> Optional[Dict]:
        """Fetch URL with caching and rate limiting."""
        cache_key = url
        now = time.time()

        # Check cache
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if now - cached_time < self.config.cache_ttl:
                return cached_data

        # Rate limiting
        time.sleep(self.config.request_delay)

        try:
            response = self._session.get(url, timeout=30)
            response.raise_for_status()

            if parser == 'json':
                data = response.json()
            else:
                data = response.text

            self._cache[cache_key] = (now, data)
            return data

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    # =========================================================================
    # Natural Stat Trick
    # =========================================================================

    def fetch_nst_team_stats(self, season: str = '2024-25') -> pd.DataFrame:
        """
        Fetch team statistics from Natural Stat Trick.

        Returns DataFrame with columns:
        - team, gp, gf, ga, gf_60, ga_60, xgf, xga, cf_pct, scf_pct, etc.
        """
        # NST URL format for team stats
        season_id = season.replace('-', '')[:4]  # '2024-25' -> '2024'
        url = f"https://www.naturalstattrick.com/teamtable.php?fromseason={season_id}{int(season_id)+1}&thruseason={season_id}{int(season_id)+1}&stype=2&sit=5v5&score=all&rate=y&team=all&loc=B&gpf=410&fd=&td="

        html = self._fetch_with_cache(url, parser='html')
        if not html:
            return self._get_sample_team_stats()

        try:
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', {'id': 'teams'})
            if table:
                df = pd.read_html(str(table))[0]
                df['team'] = df['Team'].apply(self.normalize_team_name)
                return df
        except Exception as e:
            print(f"Error parsing NST data: {e}")

        return self._get_sample_team_stats()

    def _get_sample_team_stats(self) -> pd.DataFrame:
        """Return sample team stats for testing/demo purposes."""
        teams = list(self.TEAM_MAPPING.keys())
        np.random.seed(42)

        data = []
        for team in teams:
            data.append({
                'team': team,
                'gp': np.random.randint(30, 50),
                'gf_60': round(np.random.uniform(2.5, 3.8), 2),
                'ga_60': round(np.random.uniform(2.4, 3.5), 2),
                'xgf_60': round(np.random.uniform(2.6, 3.5), 2),
                'xga_60': round(np.random.uniform(2.5, 3.4), 2),
                'cf_pct': round(np.random.uniform(45, 55), 1),
                'scf_pct': round(np.random.uniform(46, 54), 1),
                'pp_pct': round(np.random.uniform(15, 30), 1),
                'pk_pct': round(np.random.uniform(75, 88), 1),
                'sh_pct': round(np.random.uniform(8, 12), 1),
                'sv_pct': round(np.random.uniform(0.895, 0.920), 3),
                'fo_pct': round(np.random.uniform(47, 53), 1),
            })

        return pd.DataFrame(data)

    # =========================================================================
    # MoneyPuck
    # =========================================================================

    def fetch_moneypuck_goalie_stats(self, season: str = '2024-25') -> pd.DataFrame:
        """
        Fetch goalie statistics from MoneyPuck.

        Returns DataFrame with columns:
        - name, team, gp, sv_pct, gaa, gsax, gsax_60, es_sv_pct
        """
        # MoneyPuck API endpoint
        season_id = season.replace('-', '')[:4]
        url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season_id}/regular/goalies.csv"

        try:
            df = pd.read_csv(url)
            df['team'] = df['team'].apply(self.normalize_team_name)

            # Calculate derived stats
            if 'goalsAgainst' in df.columns and 'xGoals' in df.columns:
                df['gsax'] = df['xGoals'] - df['goalsAgainst']

            return df
        except Exception as e:
            print(f"Error fetching MoneyPuck goalie data: {e}")
            return self._get_sample_goalie_stats()

    def _get_sample_goalie_stats(self) -> pd.DataFrame:
        """Return sample goalie stats for testing/demo purposes."""
        goalies = [
            ('Connor Hellebuyck', 'WPG'), ('Igor Shesterkin', 'NYR'),
            ('Linus Ullmark', 'OTT'), ('Jake Oettinger', 'DAL'),
            ('Thatcher Demko', 'VAN'), ('Ilya Sorokin', 'NYI'),
            ('Juuse Saros', 'NSH'), ('Andrei Vasilevskiy', 'TBL'),
            ('Stuart Skinner', 'EDM'), ('Jeremy Swayman', 'BOS'),
            ('Joseph Woll', 'TOR'), ('Sergei Bobrovsky', 'FLA'),
        ]
        np.random.seed(42)

        data = []
        for name, team in goalies:
            data.append({
                'name': name,
                'team': team,
                'gp': np.random.randint(20, 45),
                'sv_pct': round(np.random.uniform(0.900, 0.925), 3),
                'gaa': round(np.random.uniform(2.2, 3.2), 2),
                'gsax': round(np.random.uniform(-5, 15), 1),
                'gsax_60': round(np.random.uniform(-0.3, 0.5), 2),
                'es_sv_pct': round(np.random.uniform(0.910, 0.935), 3),
            })

        return pd.DataFrame(data)

    # =========================================================================
    # Schedule & Games
    # =========================================================================

    def fetch_todays_games(self) -> List[Dict]:
        """
        Fetch today's NHL schedule.

        Returns list of game dicts with:
        - game_id, date, home_team, away_team, start_time
        """
        today = date.today().isoformat()
        url = f"https://statsapi.web.nhl.com/api/v1/schedule?date={today}"

        data = self._fetch_with_cache(url)
        if not data:
            return self._get_sample_schedule()

        games = []
        try:
            for game_date in data.get('dates', []):
                for game in game_date.get('games', []):
                    games.append({
                        'game_id': str(game['gamePk']),
                        'date': game_date['date'],
                        'home_team': self.normalize_team_name(
                            game['teams']['home']['team']['name']
                        ),
                        'away_team': self.normalize_team_name(
                            game['teams']['away']['team']['name']
                        ),
                        'start_time': game.get('gameDate', ''),
                        'venue': game.get('venue', {}).get('name', ''),
                    })
        except Exception as e:
            print(f"Error parsing schedule: {e}")
            return self._get_sample_schedule()

        return games

    def _get_sample_schedule(self) -> List[Dict]:
        """Return sample schedule for testing/demo purposes."""
        return [
            {
                'game_id': '2024020001',
                'date': date.today().isoformat(),
                'home_team': 'TOR',
                'away_team': 'MTL',
                'start_time': '19:00 ET',
                'venue': 'Scotiabank Arena',
            },
            {
                'game_id': '2024020002',
                'date': date.today().isoformat(),
                'home_team': 'COL',
                'away_team': 'VGK',
                'start_time': '21:00 ET',
                'venue': 'Ball Arena',
            },
        ]

    # =========================================================================
    # Odds API
    # =========================================================================

    def fetch_betting_odds(self, api_key: str = None) -> List[Dict]:
        """
        Fetch current betting odds from The Odds API.

        Requires API key (free tier available).
        Returns list of odds dicts with:
        - game_id, home_ml, away_ml, over_under, over_odds, under_odds
        """
        if not api_key:
            return self._get_sample_odds()

        url = f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/?apiKey={api_key}&regions=us&markets=h2h,totals&oddsFormat=american"

        data = self._fetch_with_cache(url)
        if not data:
            return self._get_sample_odds()

        odds_list = []
        try:
            for game in data:
                home_team = self.normalize_team_name(game['home_team'])
                away_team = self.normalize_team_name(game['away_team'])

                odds = {
                    'game_id': game['id'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_ml': None,
                    'away_ml': None,
                    'over_under': None,
                    'over_odds': -110,
                    'under_odds': -110,
                }

                for bookmaker in game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                if outcome['name'] == game['home_team']:
                                    odds['home_ml'] = outcome['price']
                                else:
                                    odds['away_ml'] = outcome['price']
                        elif market['key'] == 'totals':
                            for outcome in market['outcomes']:
                                odds['over_under'] = outcome.get('point', 6.0)
                                if outcome['name'] == 'Over':
                                    odds['over_odds'] = outcome['price']
                                else:
                                    odds['under_odds'] = outcome['price']
                    break  # Just use first bookmaker

                odds_list.append(odds)

        except Exception as e:
            print(f"Error parsing odds: {e}")
            return self._get_sample_odds()

        return odds_list

    def _get_sample_odds(self) -> List[Dict]:
        """Return sample odds for testing/demo purposes."""
        return [
            {
                'game_id': '2024020001',
                'home_team': 'TOR',
                'away_team': 'MTL',
                'home_ml': -165,
                'away_ml': 140,
                'over_under': 6.5,
                'over_odds': -115,
                'under_odds': -105,
            },
            {
                'game_id': '2024020002',
                'home_team': 'COL',
                'away_team': 'VGK',
                'home_ml': -135,
                'away_ml': 115,
                'over_under': 6.0,
                'over_odds': -110,
                'under_odds': -110,
            },
        ]

    # =========================================================================
    # Aggregated Data
    # =========================================================================

    def get_team_data(
        self,
        team: str,
        seasons: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all available data for a team across seasons.

        Returns dict mapping season to DataFrame of stats.
        """
        seasons = seasons or self.config.seasons
        team = self.normalize_team_name(team)

        team_data = {}
        for season in seasons:
            stats = self.fetch_nst_team_stats(season)
            team_stats = stats[stats['team'] == team]
            if not team_stats.empty:
                team_data[season] = team_stats.iloc[0].to_dict()

        return team_data

    def get_goalie_data(
        self,
        team: str,
        goalie_name: str = None
    ) -> Dict:
        """
        Get goalie data for a team (optionally specific goalie).

        Returns dict of goalie stats.
        """
        team = self.normalize_team_name(team)
        goalies = self.fetch_moneypuck_goalie_stats()

        team_goalies = goalies[goalies['team'] == team]

        if goalie_name:
            # Fuzzy match goalie name
            matches = team_goalies[
                team_goalies['name'].str.lower().str.contains(goalie_name.lower())
            ]
            if not matches.empty:
                return matches.iloc[0].to_dict()

        # Return starter (most games played)
        if not team_goalies.empty:
            return team_goalies.sort_values('gp', ascending=False).iloc[0].to_dict()

        return {}

    def get_full_game_data(
        self,
        home_team: str,
        away_team: str,
        home_goalie: str = None,
        away_goalie: str = None
    ) -> Dict:
        """
        Get all data needed for a game prediction.

        Returns comprehensive dict with team stats, goalie stats,
        and betting odds.
        """
        home_team = self.normalize_team_name(home_team)
        away_team = self.normalize_team_name(away_team)

        return {
            'home_team': {
                'code': home_team,
                'stats': self.get_team_data(home_team),
                'goalie': self.get_goalie_data(home_team, home_goalie),
            },
            'away_team': {
                'code': away_team,
                'stats': self.get_team_data(away_team),
                'goalie': self.get_goalie_data(away_team, away_goalie),
            },
            'odds': next(
                (o for o in self.fetch_betting_odds()
                 if o['home_team'] == home_team and o['away_team'] == away_team),
                self._get_sample_odds()[0]
            ),
        }
