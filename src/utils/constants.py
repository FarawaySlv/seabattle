# Window settings
WINDOW_WIDTH = 1200  # Increased width for right margin
WINDOW_HEIGHT = 800
WINDOW_TITLE = "Battleship Game"

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)  # For partially hit ships
GREEN = (0, 255, 0)

# Grid settings
GRID_SIZE = 10  # 10x10 grid
CELL_SIZE = 40
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 100
GRID_SPACING = 20
RIGHT_MARGIN = 50
LABEL_OFFSET = 22  # Space for coordinate labels
BUTTON_PANEL_OFFSET = 60  # Space for button panels

# Ship settings
SHIPS = {
    4: 1,  # 1 ship of size 4
    3: 2,  # 2 ships of size 3
    2: 3,  # 3 ships of size 2
    1: 4   # 4 ships of size 1
}

# Ship sizes
SHIP_SIZES = [5, 4, 3, 3, 2]  # Standard battleship ship sizes 