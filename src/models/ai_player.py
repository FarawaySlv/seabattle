import random
from models.ship import Ship
from utils.constants import SHIPS, GRID_SIZE

class AIPlayer:
    def __init__(self):
        self.shot_history = set()  # Keep track of all shots
        self.hit_history = set()   # Keep track of hits
        self.last_hit = None       # Last successful hit
        self.hunt_mode = False     # Whether we're hunting around a hit
        self.hunt_direction = None # Current direction we're hunting in
        self.possible_targets = set()  # Cells that might contain ships
        self.consecutive_hits = []  # Track consecutive hits to determine ship direction
        self.ship_direction = None  # Current ship direction (dx, dy)

    def place_ships(self, board):
        """Place ships randomly on the board"""
        for size, count in SHIPS.items():
            for _ in range(count):
                while True:
                    # Try random position and orientation
                    x = random.randint(0, GRID_SIZE - 1)
                    y = random.randint(0, GRID_SIZE - 1)
                    horizontal = random.choice([True, False])
                    
                    ship = Ship(size)
                    if board.place_ship(ship, x, y, horizontal):
                        break

    def get_next_shot(self, board):
        """Get the next shot coordinates using advanced strategy"""
        # First, check if we have consecutive hits to determine ship direction
        if len(self.consecutive_hits) >= 2:
            # Get the direction of the ship
            x1, y1 = self.consecutive_hits[-2]
            x2, y2 = self.consecutive_hits[-1]
            dx, dy = x2 - x1, y2 - y1
            self.ship_direction = (dx, dy)
            
            # Try to continue in the same direction
            nx, ny = x2 + dx, y2 + dy
            if 0 <= nx < 10 and 0 <= ny < 10:
                cell_state = board.get_cell_state(nx, ny)
                if cell_state == 0 or cell_state == 1:  # Empty or ship
                    return nx, ny
            
            # If we can't continue in that direction, try the opposite direction
            nx, ny = x1 - dx, y1 - dy
            if 0 <= nx < 10 and 0 <= ny < 10:
                cell_state = board.get_cell_state(nx, ny)
                if cell_state == 0 or cell_state == 1:  # Empty or ship
                    return nx, ny

        # Look for any hit to continue hunting
        for y in range(10):
            for x in range(10):
                cell_state = board.get_cell_state(x, y)
                if cell_state == 2 or cell_state == 4:  # Found a hit or partially hit
                    # Check all four adjacent cells
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 10 and 0 <= ny < 10:
                            cell_state = board.get_cell_state(nx, ny)
                            if cell_state == 0 or cell_state == 1:  # Empty or ship
                                return nx, ny
        
        # If no hit found, use probability-based targeting
        probabilities = [[0 for _ in range(10)] for _ in range(10)]
        
        # Mark cells around misses as less likely
        for y in range(10):
            for x in range(10):
                if board.get_cell_state(x, y) == 3:  # Miss
                    for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 10 and 0 <= ny < 10:
                            probabilities[ny][nx] -= 1
        
        # Find the cell with highest probability
        max_prob = -float('inf')
        best_shots = []
        for y in range(10):
            for x in range(10):
                cell_state = board.get_cell_state(x, y)
                if cell_state == 0 or cell_state == 1:  # Empty or ship
                    if probabilities[y][x] > max_prob:
                        max_prob = probabilities[y][x]
                        best_shots = [(x, y)]
                    elif probabilities[y][x] == max_prob:
                        best_shots.append((x, y))
        
        if best_shots:
            return random.choice(best_shots)
        
        # Fallback to random shot if no good options found
        while True:
            x = random.randint(0, 9)
            y = random.randint(0, 9)
            cell_state = board.get_cell_state(x, y)
            if cell_state == 0 or cell_state == 1:  # Empty or ship
                return x, y

    def record_shot(self, x, y, hit):
        """Record the result of a shot"""
        self.shot_history.add((x, y))
        if hit:
            self.hit_history.add((x, y))
            self.last_hit = (x, y)
            self.consecutive_hits.append((x, y))
            # If we have two or more hits, update ship direction
            if len(self.consecutive_hits) >= 2:
                x1, y1 = self.consecutive_hits[-2]
                x2, y2 = self.consecutive_hits[-1]
                self.ship_direction = (x2 - x1, y2 - y1)
        else:
            if self.hunt_mode:
                # Try next direction if we missed
                self.hunt_direction = (self.hunt_direction + 1) % 4
                if self.hunt_direction == 0:  # We've tried all directions
                    self.hunt_mode = False
                    self.last_hit = None
            # Reset consecutive hits and ship direction on miss
            self.consecutive_hits = []
            self.ship_direction = None 