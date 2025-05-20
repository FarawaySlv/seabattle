# Window settings
WINDOW_WIDTH = 1200  # Increased width for right margin
WINDOW_HEIGHT = 700
WINDOW_TITLE = "Sea Battle"

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)  # For partially hit ships

# Grid settings
GRID_SIZE = 10
CELL_SIZE = 40
GRID_OFFSET_X = 100
GRID_OFFSET_Y = 90  # Increased to make room for panels and labels
GRID_SPACING = 100
RIGHT_MARGIN = 60  # New margin after second grid
LABEL_OFFSET = 22  # Space for coordinate labels
BUTTON_PANEL_OFFSET = 60  # Space for button panels

# Ship settings
SHIPS = {
    4: 1,  # 1 ship of size 4
    3: 2,  # 2 ships of size 3
    2: 3,  # 3 ships of size 2
    1: 4   # 4 ships of size 1
} 