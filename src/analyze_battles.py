import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

def load_game_logs(data_dir: str = "models/battleship/data") -> List[Dict]:
    """Load all game logs from the data directory"""
    game_logs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                game_logs.append(json.load(f))
    return game_logs

def analyze_battle_types(game_logs: List[Dict]) -> Dict:
    """Analyze the distribution of battle types"""
    battle_types = defaultdict(int)
    total_battles = len(game_logs)
    
    for game in game_logs:
        player_type = game.get('player_type', 'unknown')
        player_ai_type = game.get('player_ai_type', 'unknown')
        enemy_ai_type = game.get('enemy_ai_type', 'unknown')
        
        # Determine battle type
        if player_type == 'human':
            battle_type = f"Human vs {enemy_ai_type.capitalize()}"
        else:
            battle_type = f"{player_ai_type.capitalize()} vs {enemy_ai_type.capitalize()}"
        
        battle_types[battle_type] += 1
    
    # Calculate percentages
    battle_stats = {}
    for battle_type, count in battle_types.items():
        percentage = (count / total_battles) * 100
        battle_stats[battle_type] = {
            'count': count,
            'percentage': percentage
        }
    
    return battle_stats

def analyze_winning_moves(game_logs: List[Dict]) -> Dict:
    """Analyze the number of moves needed to win for each battle type"""
    winning_moves = defaultdict(lambda: {'player': [], 'enemy': []})
    
    for game in game_logs:
        player_type = game.get('player_type', 'unknown')
        player_ai_type = game.get('player_ai_type', 'unknown')
        enemy_ai_type = game.get('enemy_ai_type', 'unknown')
        total_moves = game.get('total_moves', 0)
        winner = game.get('winner', 'unknown')
        
        # Determine battle type
        if player_type == 'human':
            battle_type = f"Human vs {enemy_ai_type.capitalize()}"
        else:
            battle_type = f"{player_ai_type.capitalize()} vs {enemy_ai_type.capitalize()}"
        
        # Add moves to the appropriate list based on winner
        if winner == 'player':
            winning_moves[battle_type]['player'].append(total_moves)
        else:
            winning_moves[battle_type]['enemy'].append(total_moves)
    
    # Calculate statistics for each battle type
    move_stats = {}
    for battle_type, moves in winning_moves.items():
        move_stats[battle_type] = {
            'player': {
                'min_moves': min(moves['player']) if moves['player'] else 0,
                'max_moves': max(moves['player']) if moves['player'] else 0,
                'avg_moves': statistics.mean(moves['player']) if moves['player'] else 0,
                'median_moves': statistics.median(moves['player']) if moves['player'] else 0,
                'total_games': len(moves['player'])
            },
            'enemy': {
                'min_moves': min(moves['enemy']) if moves['enemy'] else 0,
                'max_moves': max(moves['enemy']) if moves['enemy'] else 0,
                'avg_moves': statistics.mean(moves['enemy']) if moves['enemy'] else 0,
                'median_moves': statistics.median(moves['enemy']) if moves['enemy'] else 0,
                'total_games': len(moves['enemy'])
            }
        }
    
    return move_stats

def analyze_win_rates(game_logs: List[Dict]) -> Dict:
    """Analyze win rates for each battle type"""
    wins = defaultdict(lambda: {'player': 0, 'enemy': 0})
    total_games = defaultdict(int)
    
    for game in game_logs:
        player_type = game.get('player_type', 'unknown')
        player_ai_type = game.get('player_ai_type', 'unknown')
        enemy_ai_type = game.get('enemy_ai_type', 'unknown')
        winner = game.get('winner', 'unknown')
        
        # Determine battle type
        if player_type == 'human':
            battle_type = f"Human vs {enemy_ai_type.capitalize()}"
        else:
            battle_type = f"{player_ai_type.capitalize()} vs {enemy_ai_type.capitalize()}"
        
        total_games[battle_type] += 1
        if winner == 'player':
            wins[battle_type]['player'] += 1
        else:
            wins[battle_type]['enemy'] += 1
    
    # Calculate win rates
    win_rates = {}
    for battle_type in total_games:
        total = total_games[battle_type]
        player_wins = wins[battle_type]['player']
        enemy_wins = wins[battle_type]['enemy']
        
        win_rates[battle_type] = {
            'player_wins': player_wins,
            'enemy_wins': enemy_wins,
            'total_games': total,
            'player_win_rate': (player_wins / total) * 100 if total > 0 else 0,
            'enemy_win_rate': (enemy_wins / total) * 100 if total > 0 else 0
        }
    
    return win_rates

def print_analysis(battle_stats: Dict, move_stats: Dict, win_rates: Dict):
    """Print the analysis results in a formatted way"""
    print("\n=== Battle Analysis Report ===\n")
    
    print("1. Battle Type Distribution:")
    print("-" * 50)
    total_games = sum(stats['count'] for stats in battle_stats.values())
    for battle_type, stats in battle_stats.items():
        print(f"{battle_type}:")
        print(f"  Count: {stats['count']} out of {total_games} total games")
        print(f"  Percentage of all games: {stats['percentage']:.1f}%")
        print()
    
    print("\n2. Winning Moves Analysis:")
    print("-" * 50)
    for battle_type, stats in move_stats.items():
        print(f"{battle_type}:")
        
        print("  Player wins:")
        if stats['player']['total_games'] > 0:
            print(f"    Games won: {stats['player']['total_games']}")
            print(f"    Minimum moves to win: {stats['player']['min_moves']}")
            print(f"    Maximum moves to win: {stats['player']['max_moves']}")
            print(f"    Average moves to win: {stats['player']['avg_moves']:.1f}")
            print(f"    Median moves to win: {stats['player']['median_moves']}")
        else:
            print("    No wins recorded")
            
        print("  Enemy wins:")
        if stats['enemy']['total_games'] > 0:
            print(f"    Games won: {stats['enemy']['total_games']}")
            print(f"    Minimum moves to win: {stats['enemy']['min_moves']}")
            print(f"    Maximum moves to win: {stats['enemy']['max_moves']}")
            print(f"    Average moves to win: {stats['enemy']['avg_moves']:.1f}")
            print(f"    Median moves to win: {stats['enemy']['median_moves']}")
        else:
            print("    No wins recorded")
        print()
    
    print("\n3. Win Rates:")
    print("-" * 50)
    for battle_type, stats in win_rates.items():
        print(f"{battle_type}:")
        print(f"  Total Games: {stats['total_games']}")
        print(f"  Player wins: {stats['player_wins']} ({stats['player_win_rate']:.1f}%)")
        print(f"  Enemy wins: {stats['enemy_wins']} ({stats['enemy_win_rate']:.1f}%)")
        print()

def main():
    # Load game logs
    game_logs = load_game_logs()
    
    if not game_logs:
        print("No game logs found!")
        return
    
    # Analyze battle types
    battle_stats = analyze_battle_types(game_logs)
    
    # Analyze winning moves
    move_stats = analyze_winning_moves(game_logs)
    
    # Analyze win rates
    win_rates = analyze_win_rates(game_logs)
    
    # Print results
    print_analysis(battle_stats, move_stats, win_rates)

if __name__ == "__main__":
    main() 