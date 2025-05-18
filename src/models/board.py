from utils.constants import GRID_SIZE

class Board:
    def __init__(self):
        # Initialize empty grid (0 = empty, 1 = ship, 2 = hit, 3 = miss, 4 = partially hit)
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.ships = []

    def is_valid_placement(self, ship, x, y, horizontal):
        """Check if ship placement is valid"""
        # Check if ship fits within grid
        if horizontal:
            if x + ship.size > GRID_SIZE:
                return False
            # Check ship area and surrounding cells
            for i in range(-1, ship.size + 1):
                for j in range(-1, 2):
                    check_x = x + i
                    check_y = y + j
                    if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                        if self.grid[check_y][check_x] != 0:
                            return False
        else:
            if y + ship.size > GRID_SIZE:
                return False
            # Check ship area and surrounding cells
            for i in range(-1, 2):
                for j in range(-1, ship.size + 1):
                    check_x = x + i
                    check_y = y + j
                    if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                        if self.grid[check_y][check_x] != 0:
                            return False
        return True

    def place_ship(self, ship, x, y, horizontal):
        """Place a ship on the board"""
        if not self.is_valid_placement(ship, x, y, horizontal):
            return False

        if horizontal:
            for i in range(ship.size):
                self.grid[y][x + i] = 1
        else:
            for i in range(ship.size):
                self.grid[y + i][x] = 1

        ship.place(x, y, horizontal)
        self.ships.append(ship)
        return True

    def receive_shot(self, x, y):
        """Receive a shot at the given coordinates"""
        if self.grid[y][x] == 2 or self.grid[y][x] == 3:  # Already hit or missed
            return False, "Already shot here"
            
        if self.grid[y][x] == 1:  # Hit a ship
            ship = self.get_ship_at(x, y)
            if ship:
                ship.add_hit(x, y)
                if ship.is_destroyed():
                    # Mark all ship cells as hit (2) when destroyed
                    for coord in ship.coordinates:
                        self.grid[coord[1]][coord[0]] = 2
                        # Mark surrounding cells as missed
                        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                            nx, ny = coord[0] + dx, coord[1] + dy
                            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[ny][nx] == 0:
                                self.grid[ny][nx] = 3  # Mark as miss
                    return True, "Ship destroyed!"
                else:
                    # Mark as partially hit (4) if ship is not destroyed
                    self.grid[y][x] = 4
                    return True, "Ship hit!"
        else:  # Miss
            self.grid[y][x] = 3  # Mark as miss
            return False, "Miss"

    def is_all_ships_destroyed(self):
        """Check if all ships are destroyed"""
        return all(ship.is_destroyed() for ship in self.ships)

    def get_cell_state(self, x, y):
        """Get the state of a cell (0: empty, 1: ship, 2: hit, 3: miss)"""
        return self.grid[y][x]  # Return the actual cell state directly

    def get_remaining_ships(self):
        """Get the count of remaining ships by size"""
        remaining = {size: 0 for size in [4, 3, 2, 1]}
        for ship in self.ships:
            if not ship.is_destroyed():
                remaining[ship.size] += 1
        return remaining

    def get_ship_at(self, x, y):
        """Get the ship at the given coordinates"""
        for ship in self.ships:
            if (x, y) in ship.coordinates:
                return ship
        return None 