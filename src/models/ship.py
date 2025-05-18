class Ship:
    def __init__(self, size):
        self.size = size
        self.hits = set()  # Set of hit coordinates
        self.coordinates = []  # List of ship coordinates
        self.horizontal = True
        self.x = -1
        self.y = -1

    def place(self, x, y, horizontal):
        """Place the ship on the board"""
        self.x = x
        self.y = y
        self.horizontal = horizontal
        self.coordinates = []
        
        if horizontal:
            for i in range(self.size):
                self.coordinates.append((x + i, y))
        else:
            for i in range(self.size):
                self.coordinates.append((x, y + i))

    def is_hit(self, x, y):
        """Check if the ship is hit at the given coordinates"""
        return (x, y) in self.hits

    def hit(self, x, y):
        """Record a hit on the ship"""
        if (x, y) in self.coordinates:  # Check if coordinates are part of the ship
            self.hits.add((x, y))
            return True
        return False

    def is_destroyed(self):
        """Check if the ship is completely destroyed"""
        return len(self.hits) == self.size  # Compare hits count with ship size

    def add_hit(self, x, y):
        """Record a hit on the ship"""
        if (x, y) in self.coordinates:  # Only add hit if coordinates are part of the ship
            self.hits.add((x, y)) 