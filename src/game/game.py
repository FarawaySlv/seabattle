import pygame
from utils.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, WHITE,
    GRID_OFFSET_X, GRID_OFFSET_Y, CELL_SIZE, GRID_SPACING, RIGHT_MARGIN
)
from models.board import Board
from models.ship import Ship
from models.ai_player import AIPlayer
from ui.board_renderer import BoardRenderer
from game.game_state import GameState

class Game:
    def __init__(self, ai_difficulty="medium"):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Initialize game components
        self.player_board = Board()
        self.ai_board = Board()
        self.board_renderer = BoardRenderer(self.screen)
        self.game_state = GameState()
        
        # Player type (human or AI)
        self.player_type = "human"  # Default to human player
        
        # Initialize board rectangles
        self.player_board_rect = pygame.Rect(
            GRID_OFFSET_X,
            GRID_OFFSET_Y + 100,
            10 * CELL_SIZE,
            10 * CELL_SIZE
        )
        self.ai_board_rect = pygame.Rect(
            GRID_OFFSET_X + 10 * CELL_SIZE + GRID_SPACING,
            GRID_OFFSET_Y + 100,
            10 * CELL_SIZE,
            10 * CELL_SIZE
        )
        
        # Initialize AI
        self.ai = AIPlayer(difficulty=ai_difficulty)
        self.player_ai = None  # Will be initialized if player type is AI
        
        # Initialize fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Notification system
        self.notification = ""
        self.notification_timer = 0
        self.notification_duration = 120  # frames (2 seconds at 60 FPS)
        
        # Ship placement
        self.selected_ship = None
        self.ship_orientation = True  # True for horizontal, False for vertical

        # Restart button
        self.restart_button = pygame.Rect(
            WINDOW_WIDTH // 2 - 100,
            WINDOW_HEIGHT - 80,
            200,
            40
        )

        # Random placement button
        self.random_placement_button = pygame.Rect(
            WINDOW_WIDTH // 2 - 100,
            100,
            200,
            40
        )

        # Difficulty buttons
        button_width = 100
        button_spacing = 20
        total_width = (button_width * 3) + (button_spacing * 2)
        start_x = (WINDOW_WIDTH - total_width) // 2
        
        self.easy_button = pygame.Rect(
            start_x,
            50,
            button_width,
            40
        )
        self.medium_button = pygame.Rect(
            start_x + button_width + button_spacing,
            50,
            button_width,
            40
        )
        self.hard_button = pygame.Rect(
            start_x + (button_width + button_spacing) * 2,
            50,
            button_width,
            40
        )

        # Player type buttons
        self.human_button = pygame.Rect(
            start_x - button_width - button_spacing,
            50,
            button_width,
            40
        )
        self.ai_player_button = pygame.Rect(
            start_x + total_width + button_spacing,
            50,
            button_width,
            40
        )

    def show_notification(self, message, duration=60):
        """Show a temporary notification message"""
        self.notification = message
        self.notification_timer = duration
        # Redraw to show the notification immediately
        self.draw()
        pygame.display.flip()

    def restart_game(self):
        """Restart the game"""
        self.player_board = Board()
        self.ai_board = Board()
        self.game_state = GameState()
        self.ai = AIPlayer(difficulty=self.ai.difficulty)
        self.notification = "Game restarted! Place your ships."
        self.notification_timer = self.notification_duration

    def place_ships_randomly(self):
        """Place player's ships randomly on the board"""
        self.player_board = Board()  # Clear the board
        ai = AIPlayer(difficulty="easy")  # Use AI to place ships
        ai.place_ships(self.player_board)
        self.game_state.start_playing_phase()
        self.show_notification("Ships placed randomly! Enemy fleet is ready!")
        # Place AI ships
        self.ai.place_ships(self.ai_board)

    def change_difficulty(self, difficulty):
        """Change AI difficulty and restart the game"""
        self.ai = AIPlayer(difficulty=difficulty)
        self.restart_game()
        self.show_notification(f"Difficulty changed to {difficulty.capitalize()}!")

    def change_player_type(self, player_type):
        """Change player type and restart the game"""
        self.player_type = player_type
        if player_type == "ai":
            self.player_ai = AIPlayer(difficulty=self.ai.difficulty)
        else:
            self.player_ai = None
        self.restart_game()
        self.show_notification(f"Player type changed to {player_type.capitalize()}!")

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r and self.game_state.is_setup_phase():
                    self.game_state.toggle_ship_orientation()
                    self.show_notification("Ship orientation changed!")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if self.game_state.is_game_over() and self.restart_button.collidepoint(event.pos):
                        self.restart_game()
                    elif self.game_state.is_setup_phase() and self.random_placement_button.collidepoint(event.pos):
                        self.place_ships_randomly()
                    # Check difficulty buttons
                    elif self.easy_button.collidepoint(event.pos):
                        self.change_difficulty("easy")
                    elif self.medium_button.collidepoint(event.pos):
                        self.change_difficulty("medium")
                    elif self.hard_button.collidepoint(event.pos):
                        self.change_difficulty("hard")
                    # Check player type buttons
                    elif self.human_button.collidepoint(event.pos):
                        self.change_player_type("human")
                    elif self.ai_player_button.collidepoint(event.pos):
                        self.change_player_type("ai")
                    elif not self.game_state.is_ai_turn() and self.player_type == "human":  # Only handle clicks during player's turn if human
                        self.handle_click(event.pos)
                elif event.button == 3 and not self.game_state.is_ai_turn() and self.player_type == "human":  # Right click
                    self.handle_right_click(event.pos)

    def handle_click(self, pos):
        """Handle mouse click based on game state"""
        if self.game_state.is_setup_phase():
            self.handle_setup_click(pos)
        elif self.game_state.is_playing_phase():
            self.handle_playing_click(pos)

    def handle_right_click(self, pos):
        """Handle right mouse click to remove a ship during setup phase"""
        if not self.game_state.is_setup_phase():
            return
        x = (pos[0] - GRID_OFFSET_X) // CELL_SIZE
        y = (pos[1] - GRID_OFFSET_Y) // CELL_SIZE
        if 0 <= x < 10 and 0 <= y < 10:
            current_board = self.player_board if self.game_state.is_player_turn else self.ai_board
            # Find and remove the ship at (x, y)
            for ship in current_board.ships:
                if (x, y) in ship.coordinates:
                    for sx, sy in ship.coordinates:
                        current_board.grid[sy][sx] = 0
                    current_board.ships.remove(ship)
                    # Move back in the ship placement order
                    if self.game_state.current_ship_index > 0:
                        self.game_state.current_ship_index -= 1
                    self.show_notification(f"Ship of size {ship.size} removed!")
                    break

    def handle_setup_click(self, pos):
        """Handle click during ship placement phase"""
        # Check if all ships are already placed
        if self.game_state.is_ship_placement_complete():
            return
            
        x = (pos[0] - GRID_OFFSET_X) // CELL_SIZE
        y = (pos[1] - GRID_OFFSET_Y) // CELL_SIZE
        if 0 <= x < 10 and 0 <= y < 10:
            ship_size = self.game_state.get_current_ship_size()
            if ship_size:
                ship = Ship(ship_size)
                if self.player_board.place_ship(ship, x, y, self.game_state.horizontal):
                    self.show_notification(f"Ship of size {ship_size} placed!")
                    self.game_state.next_ship()
                    
                    # Check if all ships are placed
                    if self.game_state.is_ship_placement_complete():
                        self.show_notification("Your fleet is ready for battle!")
                        # Place AI ships
                        self.ai.place_ships(self.ai_board)
                        self.show_notification("Enemy fleet is ready!")
                        self.game_state.start_playing_phase()
                else:
                    self.show_notification("Invalid placement! Ships must have 1 cell gap between them.")

    def handle_playing_click(self, pos):
        """Handle clicks during the playing phase"""
        if not self.game_state.is_player_turn or self.player_type == "ai":
            return  # Ignore clicks during AI's turn or if player is AI
            
        # Check if click is on AI's board
        if self.ai_board_rect.collidepoint(pos):
            # Convert screen coordinates to grid coordinates
            grid_x = (pos[0] - self.ai_board_rect.x) // CELL_SIZE
            grid_y = (pos[1] - self.ai_board_rect.y) // CELL_SIZE
            
            if 0 <= grid_x < 10 and 0 <= grid_y < 10:
                hit, message = self.ai_board.receive_shot(grid_x, grid_y)
                
                if hit:
                    self.show_notification(f"Hit! {message}")
                    if self.ai_board.is_all_ships_destroyed():
                        self.game_state.set_game_over()
                        self.show_notification("Game Over! You win!")
                else:
                    self.show_notification("Miss!")
                    pygame.time.wait(1000)  # Wait to show miss
                    self.game_state.switch_turn()
                    
                    # Clear notification and show AI turn
                    self.show_notification("AI's turn...")
                    pygame.time.wait(1500)  # Wait before AI's turn
                    self.handle_ai_turn()

    def handle_ai_turn(self):
        """Handle AI's turn"""
        if not self.game_state.is_playing_phase():
            return

        # Get AI's shot
        x, y = self.ai.get_next_shot(self.player_board)
        hit, message = self.player_board.receive_shot(x, y)
        
        # Record the result for AI's strategy
        self.ai.record_shot(x, y, hit)
        
        # Show AI's shot result
        if hit:
            self.show_notification(f"AI hits at {chr(65 + x)}{y + 1}! {message}")
            if self.player_board.is_all_ships_destroyed():
                self.game_state.set_game_over()
                self.show_notification("Game Over! AI wins!")
            else:
                # AI gets another turn after a hit
                pygame.time.wait(1500)  # Wait to show hit
                self.handle_ai_turn()
        else:
            self.show_notification(f"AI misses at {chr(65 + x)}{y + 1}")
            pygame.time.wait(1000)  # Wait to show miss
            self.game_state.switch_turn()
            self.show_notification("Your turn!")

    def draw(self):
        """Draw the game state"""
        self.screen.fill(WHITE)
        
        # Draw both boards
        self.board_renderer.draw_grid(
            self.player_board,
            self.player_board_rect.x,
            self.player_board_rect.y,
            hide_ships=False  # Show ships on player's board
        )
        
        self.board_renderer.draw_grid(
            self.ai_board,
            self.ai_board_rect.x,
            self.ai_board_rect.y,
            hide_ships=True  # Hide ships on enemy's board
        )
        
        # Draw coordinates and labels
        self.board_renderer.draw_coordinates(
            self.player_board_rect.x,
            self.player_board_rect.y
        )
        self.board_renderer.draw_coordinates(
            self.ai_board_rect.x,
            self.ai_board_rect.y
        )
        self.board_renderer.draw_board_labels(
            self.player_board_rect.x,
            self.player_board_rect.y,
            self.ai_board_rect.x
        )
        
        # Draw status and notifications
        self.draw_status()
        
        # Draw turn indicator below the grids only during playing phase
        if self.game_state.is_playing_phase():
            turn_color = (0, 255, 0) if self.game_state.is_player_turn else (255, 0, 0)
            turn_text = "Your turn - Click on enemy board to shoot" if self.game_state.is_player_turn and self.player_type == "human" else "AI's turn..."
            turn_surface = self.font.render(turn_text, True, turn_color)
            turn_rect = turn_surface.get_rect(
                centerx=self.screen.get_rect().centerx,
                top=self.player_board_rect.bottom + 20
            )
            self.screen.blit(turn_surface, turn_rect)
        
        # Draw notifications below turn indicator
        if self.notification:
            notification_surface = self.font.render(self.notification, True, (0, 0, 0))
            notification_rect = notification_surface.get_rect(
                centerx=self.screen.get_rect().centerx,
                top=self.player_board_rect.bottom + 60
            )
            self.screen.blit(notification_surface, notification_rect)

        # Draw restart button if game is over
        if self.game_state.is_game_over():
            pygame.draw.rect(self.screen, (200, 200, 200), self.restart_button)
            restart_text = self.font.render("Restart Game", True, (0, 0, 0))
            restart_rect = restart_text.get_rect(center=self.restart_button.center)
            self.screen.blit(restart_text, restart_rect)

        # Draw random placement button during setup phase
        if self.game_state.is_setup_phase():
            pygame.draw.rect(self.screen, (200, 200, 200), self.random_placement_button)
            random_text = self.font.render("Random Placement", True, (0, 0, 0))
            random_rect = random_text.get_rect(center=self.random_placement_button.center)
            self.screen.blit(random_text, random_rect)

        # Draw difficulty buttons
        # Easy button
        easy_color = (100, 255, 100) if self.ai.difficulty == "easy" else (200, 200, 200)
        pygame.draw.rect(self.screen, easy_color, self.easy_button)
        easy_text = self.font.render("Easy", True, (0, 0, 0))
        easy_rect = easy_text.get_rect(center=self.easy_button.center)
        self.screen.blit(easy_text, easy_rect)

        # Medium button
        medium_color = (255, 255, 100) if self.ai.difficulty == "medium" else (200, 200, 200)
        pygame.draw.rect(self.screen, medium_color, self.medium_button)
        medium_text = self.font.render("Medium", True, (0, 0, 0))
        medium_rect = medium_text.get_rect(center=self.medium_button.center)
        self.screen.blit(medium_text, medium_rect)

        # Hard button
        hard_color = (255, 100, 100) if self.ai.difficulty == "hard" else (200, 200, 200)
        pygame.draw.rect(self.screen, hard_color, self.hard_button)
        hard_text = self.font.render("Hard", True, (0, 0, 0))
        hard_rect = hard_text.get_rect(center=self.hard_button.center)
        self.screen.blit(hard_text, hard_rect)

        # Draw player type buttons
        # Human button
        human_color = (100, 100, 255) if self.player_type == "human" else (200, 200, 200)
        pygame.draw.rect(self.screen, human_color, self.human_button)
        human_text = self.font.render("Human", True, (0, 0, 0))
        human_rect = human_text.get_rect(center=self.human_button.center)
        self.screen.blit(human_text, human_rect)

        # AI Player button
        ai_player_color = (255, 100, 255) if self.player_type == "ai" else (200, 200, 200)
        pygame.draw.rect(self.screen, ai_player_color, self.ai_player_button)
        ai_player_text = self.font.render("AI", True, (0, 0, 0))
        ai_player_rect = ai_player_text.get_rect(center=self.ai_player_button.center)
        self.screen.blit(ai_player_text, ai_player_rect)
        
        pygame.display.flip()

    def draw_status(self):
        """Draw the game status"""
        if self.game_state.is_setup_phase():
            if self.game_state.is_ship_placement_complete():
                status_text = "Your fleet is ready for battle!"
            else:
                ship_size = self.game_state.get_current_ship_size()
                status_text = f"Place your ship of size {ship_size} (Right-click to remove)"
                if self.game_state.horizontal:
                    status_text += " - Horizontal (Press R to rotate)"
                else:
                    status_text += " - Vertical (Press R to rotate)"
        elif self.game_state.is_game_over():
            status_text = "Game Over! You win!" if self.game_state.is_player_turn else "Game Over! AI wins!"
        else:
            # Add remaining ships info
            board = self.player_board if self.game_state.is_player_turn else self.ai_board
            remaining = board.get_remaining_ships()
            
            if remaining:
                status_text = f"Ships remaining: {', '.join(f'{size}: {count}' for size, count in remaining.items())}"
            else:
                status_text = ""
        
        if status_text:
            status_surface = self.font.render(status_text, True, (0, 0, 0))
            status_rect = status_surface.get_rect(
                centerx=self.screen.get_rect().centerx,
                top=10  # Keep status text at top
            )
            self.screen.blit(status_surface, status_rect)

    def run(self):
        while self.running:
            self.handle_events()
            
            # Handle AI vs AI mode
            if self.player_type == "ai" and self.game_state.is_playing_phase() and not self.game_state.is_game_over():
                if self.game_state.is_player_turn:
                    # Player AI's turn
                    x, y = self.player_ai.get_next_shot(self.ai_board)
                    hit, message = self.ai_board.receive_shot(x, y)
                    self.player_ai.record_shot(x, y, hit)
                    
                    if hit:
                        self.show_notification(f"Player AI hits at {chr(65 + x)}{y + 1}! {message}")
                        if self.ai_board.is_all_ships_destroyed():
                            self.game_state.set_game_over()
                            self.show_notification("Game Over! Player AI wins!")
                    else:
                        self.show_notification(f"Player AI misses at {chr(65 + x)}{y + 1}")
                        pygame.time.wait(1000)
                        self.game_state.switch_turn()
                else:
                    # Enemy AI's turn
                    self.handle_ai_turn()
            
            self.draw()
            self.clock.tick(60)

        pygame.quit() 