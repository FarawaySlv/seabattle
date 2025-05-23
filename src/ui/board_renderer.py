import pygame
from utils.constants import (
    CELL_SIZE, GRID_OFFSET_X, GRID_OFFSET_Y, LABEL_OFFSET,
    WHITE, BLACK, BLUE, RED, GRAY, ORANGE
)

class BoardRenderer:
    def __init__(self, screen):
        self.screen = screen

    def get_cell_color(self, board, x, y, hide_ships=False):
        """Get the color for a cell based on its state"""
        cell_state = board.get_cell_state(x, y)
        
        if cell_state == 1 and not hide_ships:  # Ship
            return BLUE  # Intact ship
        elif cell_state == 2:  # Hit
            return RED  # Hit cell
        elif cell_state == 3:  # Miss
            return GRAY  # Missed cell
        elif cell_state == 4:  # Partially hit
            return ORANGE  # Partially hit ship
        return WHITE  # Empty cell

    def draw_grid(self, board, offset_x=GRID_OFFSET_X, offset_y=GRID_OFFSET_Y, hide_ships=False):
        """Draw the game grid"""
        # Draw vertical lines
        for x in range(11):
            pygame.draw.line(
                self.screen,
                BLACK,
                (offset_x + x * CELL_SIZE, offset_y),
                (offset_x + x * CELL_SIZE, offset_y + 10 * CELL_SIZE)
            )

        # Draw horizontal lines
        for y in range(11):
            pygame.draw.line(
                self.screen,
                BLACK,
                (offset_x, offset_y + y * CELL_SIZE),
                (offset_x + 10 * CELL_SIZE, offset_y + y * CELL_SIZE)
            )

        # Draw cell contents
        for y in range(10):
            for x in range(10):
                rect = pygame.Rect(
                    offset_x + x * CELL_SIZE + 1,
                    offset_y + y * CELL_SIZE + 1,
                    CELL_SIZE - 2,
                    CELL_SIZE - 2
                )
                
                # Get and draw cell color
                color = self.get_cell_color(board, x, y, hide_ships)
                pygame.draw.rect(self.screen, color, rect)

        # Draw surrounding cells for destroyed ships after all cells are drawn
        if not hide_ships:
            for y in range(10):
                for x in range(10):
                    ship = board.get_ship_at(x, y)
                    if ship and ship.is_destroyed():
                        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < 10 and 0 <= ny < 10:
                                # Only mark empty cells as missed
                                if board.get_cell_state(nx, ny) == 0:
                                    surround_rect = pygame.Rect(
                                        offset_x + nx * CELL_SIZE + 1,
                                        offset_y + ny * CELL_SIZE + 1,
                                        CELL_SIZE - 2,
                                        CELL_SIZE - 2
                                    )
                                    pygame.draw.rect(self.screen, GRAY, surround_rect)

    def draw_coordinates(self, x, y):
        """Draw coordinate labels (A-J, 0-9)"""
        # Use system font instead of default font
        try:
            font = pygame.font.SysFont('arial', 22)  # Reduced font size by 20%
        except:
            # Fallback to any available system font
            font = pygame.font.SysFont(None, 22)  # Reduced font size by 20%
            
        # Draw column labels (A-J)
        for i in range(10):
            text = font.render(chr(65 + i), True, (0, 0, 0))
            text_rect = text.get_rect(
                centerx=x + i * CELL_SIZE + CELL_SIZE // 2,
                top=y - LABEL_OFFSET
            )
            self.screen.blit(text, text_rect)
        
        # Draw row labels (0-9)
        for i in range(10):
            text = font.render(str(i), True, (0, 0, 0))
            text_rect = text.get_rect(
                right=x - 10,
                centery=y + i * CELL_SIZE + CELL_SIZE // 2
            )
            self.screen.blit(text, text_rect)

    def draw_board_labels(self, offset_x=GRID_OFFSET_X, offset_y=GRID_OFFSET_Y, second_grid_x=None):
        """Draw labels for both boards"""
        # Use system font instead of default font
        try:
            font = pygame.font.SysFont('arial', 36)
        except:
            # Fallback to any available system font
            font = pygame.font.SysFont(None, 36)
            
        # Player's board label
        text = font.render("Your Board", True, BLACK)
        text_rect = text.get_rect(center=(offset_x + 5 * CELL_SIZE, 60))
        self.screen.blit(text, text_rect)

        # Enemy's board label
        if second_grid_x:
            text = font.render("Enemy Board", True, BLACK)
            text_rect = text.get_rect(center=(second_grid_x + 5 * CELL_SIZE, 60))
            self.screen.blit(text, text_rect) 