from utils.constants import SHIPS

class GameState:
    SETUP = "setup"
    PLAYING = "playing"
    GAME_OVER = "game_over"

    def __init__(self, is_player_turn=True):
        self.current_state = self.SETUP
        self.ships_to_place = self._initialize_ships()
        self.current_ship_index = 0
        self.horizontal = True
        self.is_player_turn = is_player_turn  # True for player's turn, False for AI's turn

    def _initialize_ships(self):
        """Initialize the list of ships to be placed"""
        ships = []
        for size, count in SHIPS.items():
            for _ in range(count):
                ships.append(size)
        return ships

    def get_current_ship_size(self):
        """Get the size of the current ship to place"""
        if self.current_ship_index < len(self.ships_to_place):
            return self.ships_to_place[self.current_ship_index]
        return None

    def next_ship(self):
        """Move to the next ship to place"""
        self.current_ship_index += 1
        if self.current_ship_index >= len(self.ships_to_place):
            # All ships placed, start playing phase
            self.current_state = self.PLAYING
            # Don't override is_player_turn here, keep the value set in constructor

    def toggle_ship_orientation(self):
        """Toggle between horizontal and vertical ship placement"""
        self.horizontal = not self.horizontal

    def switch_turn(self):
        """Switch between player and AI turns"""
        self.is_player_turn = not self.is_player_turn

    def is_setup_phase(self):
        """Check if the game is in setup phase"""
        return self.current_state == self.SETUP

    def is_playing_phase(self):
        """Check if the game is in playing phase"""
        return self.current_state == self.PLAYING

    def is_game_over(self):
        """Check if the game is over"""
        return self.current_state == self.GAME_OVER

    def set_game_over(self):
        """Set the game state to game over"""
        self.current_state = self.GAME_OVER

    def is_ship_placement_complete(self):
        """Check if all ships have been placed"""
        return self.current_ship_index >= len(self.ships_to_place)

    def start_playing_phase(self):
        """Start the playing phase"""
        self.current_state = self.PLAYING
        # Don't override is_player_turn here, keep the value set in constructor

    def is_ai_turn(self):
        """Check if it's AI's turn"""
        return self.is_playing_phase() and not self.is_player_turn 