import pygame
from utils.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, WHITE,
    GRID_OFFSET_X, GRID_OFFSET_Y, CELL_SIZE, GRID_SPACING, RIGHT_MARGIN
)
from models.board import Board
from models.ship import Ship
from models.ai_player import AIPlayer
from models.transformer_player import TransformerPlayer
from models.game_logger import GameLogger
from ui.board_renderer import BoardRenderer
from game.game_state import GameState

class Game:
    def __init__(self, ai_type="algorithmic"):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Initialize game components
        self.player_board = Board()
        self.ai_board = Board()
        self.board_renderer = BoardRenderer(self.screen)
        self.game_state = GameState()
        self.game_logger = GameLogger()
        
        # Player type (human or AI)
        self.player_type = "human"  # Default to human player
        self.player_ai_type = "algorithmic"  # Default to algorithmic AI
        self.enemy_ai_type = ai_type  # Type of AI opponent
        
        # Auto-restart state for AI vs AI
        self.auto_restart = False
        
        # Speed control
        self.speed_multiplier = 1  # Default speed
        self.speed_button_radius = 15
        self.speed_button_spacing = 10
        self.speed_button_x = WINDOW_WIDTH - 50  # Position from right edge
        self.speed_button_y = 50  # Start position from top
        
        # Initialize fonts
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 20)
        
        # Notification system
        self.notification = ""
        self.notification_timer = 0
        self.notification_duration = 120
        
        # Ship placement
        self.selected_ship = None
        self.ship_orientation = True

        # Panel settings
        panel_height = 60
        button_height = 35
        button_width = 110
        button_spacing = 8
        panel_padding = 15

        # Initialize board rectangles first
        self.player_board_rect = pygame.Rect(
            GRID_OFFSET_X,
            GRID_OFFSET_Y + 20 + panel_height + 10,  # Reduced top margin
            10 * CELL_SIZE,
            10 * CELL_SIZE
        )
        self.ai_board_rect = pygame.Rect(
            GRID_OFFSET_X + 10 * CELL_SIZE + GRID_SPACING,
            GRID_OFFSET_Y + 20 + panel_height + 10,  # Reduced top margin
            10 * CELL_SIZE,
            10 * CELL_SIZE
        )

        # Player side panel (left)
        self.player_panel_rect = pygame.Rect(
            self.player_board_rect.x,
            GRID_OFFSET_Y + 20,  # Reduced top margin
            self.player_board_rect.width,
            panel_height
        )

        # Player side buttons
        total_buttons_width = (button_width * 3) + (button_spacing * 2)
        start_x = self.player_panel_rect.x + (self.player_panel_rect.width - total_buttons_width) // 2
        button_y = self.player_panel_rect.y + (panel_height - button_height) // 2

        self.player_human_button = pygame.Rect(
            start_x,
            button_y,
            button_width,
            button_height
        )
        self.player_algo_button = pygame.Rect(
            start_x + button_width + button_spacing,
            button_y,
            button_width,
            button_height
        )
        self.player_trans_button = pygame.Rect(
            start_x + (button_width + button_spacing) * 2,
            button_y,
            button_width,
            button_height
        )

        # Enemy side panel (right)
        self.enemy_panel_rect = pygame.Rect(
            self.ai_board_rect.x,
            GRID_OFFSET_Y + 20,  # Reduced top margin
            self.ai_board_rect.width,
            panel_height
        )

        # Enemy side buttons
        total_buttons_width = (button_width * 2) + button_spacing
        start_x = self.enemy_panel_rect.x + (self.enemy_panel_rect.width - total_buttons_width) // 2
        button_y = self.enemy_panel_rect.y + (panel_height - button_height) // 2

        self.enemy_algo_button = pygame.Rect(
            start_x,
            button_y,
            button_width,
            button_height
        )
        self.enemy_trans_button = pygame.Rect(
            start_x + button_width + button_spacing,
            button_y,
            button_width,
            button_height
        )

        # Random placement button (square, below speed buttons)
        button_size = 60  # Square size
        self.random_placement_button = pygame.Rect(
            self.speed_button_x - button_size // 2,  # Center under speed buttons
            self.speed_button_y + (self.speed_button_radius * 2 + self.speed_button_spacing) * 3 + 10,  # Below x3 button
            button_size,
            button_size
        )

        # AI vs AI button (square, bottom right corner)
        self.ai_vs_ai_button = pygame.Rect(
            WINDOW_WIDTH - button_size - 20,  # 20 pixels from right edge
            WINDOW_HEIGHT - button_size - 20,  # 20 pixels from bottom
            button_size,
            button_size
        )

        # Restart button
        self.restart_button = pygame.Rect(
            WINDOW_WIDTH // 2 - 100,
            WINDOW_HEIGHT - 50,  # Moved up from 80
            200,
            35  # Reduced height
        )

        # Start button
        self.start_button = pygame.Rect(
            WINDOW_WIDTH // 2 - 100,
            WINDOW_HEIGHT - 50,  # Moved up from 80
            200,
            35  # Reduced height
        )

        # Initialize AI
        self.ai = self._create_ai(ai_type)
        self.player_ai = None  # Will be initialized if player type is AI

    def _create_ai(self, ai_type: str):
        """Create an AI player of the specified type"""
        if ai_type == "transformer":
            return TransformerPlayer()
        else:
            return AIPlayer()

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
        self.ai = self._create_ai(self.enemy_ai_type)
        self.notification = "Game restarted! Place your ships."
        self.notification_timer = self.notification_duration

    def place_ships_randomly(self):
        """Place player's ships randomly on the board"""
        self.player_board = Board()  # Clear the board
        ai = AIPlayer()  # Use AI to place ships
        ai.place_ships(self.player_board)        
        self.show_notification("Ships placed randomly! Enemy fleet is ready!")
        # Place AI ships
        self.ai.place_ships(self.ai_board)
        # Log initial board states
        self.game_logger.log_initial_board(self.player_board.grid, is_player=True)
        self.game_logger.log_initial_board(self.ai_board.grid, is_player=False)
        # Update game state to reflect all ships are placed
        self.game_state.current_ship_index = len(self.game_state.ships_to_place)

    def change_ai_type(self, ai_type: str):
        """Change AI type and restart the game"""
        self.enemy_ai_type = ai_type
        self.ai = self._create_ai(ai_type)
        if self.player_type == "ai":
            self.player_ai = self._create_ai(self.player_ai_type)
        self.restart_game()
        self.show_notification(f"AI type changed to {ai_type.capitalize()}!")

    def change_player_type(self, player_type: str):
        """Change player type and restart the game"""
        self.player_type = player_type
        if player_type == "ai":
            self.player_ai = self._create_ai(self.player_ai_type)
            # Place ships for both AIs
            self.player_ai.place_ships(self.player_board)
            self.ai.place_ships(self.ai_board)
            # Log initial board states
            self.game_logger.log_initial_board(self.player_board.grid, is_player=True)
            self.game_logger.log_initial_board(self.ai_board.grid, is_player=False)
            self.game_logger.set_player_type("ai")
        else:
            self.player_ai = None
            self.game_logger.set_player_type("human")
        self.restart_game()
        self.show_notification(f"Player type changed to {player_type.capitalize()}!")

    def change_player_ai_type(self, ai_type: str):
        """Change player AI type and restart the game"""
        self.player_ai_type = ai_type
        if self.player_type == "ai":
            self.player_ai = self._create_ai(ai_type)
            # Place ships for both AIs
            self.player_ai.place_ships(self.player_board)
            self.ai.place_ships(self.ai_board)
            # Log initial board states
            self.game_logger.log_initial_board(self.player_board.grid, is_player=True)
            self.game_logger.log_initial_board(self.ai_board.grid, is_player=False)
        self.restart_game()
        self.show_notification(f"Player AI type changed to {ai_type.capitalize()}!")

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
                    # Check speed buttons
                    for i in range(3):
                        button_center = (self.speed_button_x, self.speed_button_y + i * (self.speed_button_radius * 2 + self.speed_button_spacing))
                        if ((event.pos[0] - button_center[0]) ** 2 + (event.pos[1] - button_center[1]) ** 2) <= self.speed_button_radius ** 2:
                            self.speed_multiplier = i + 1
                            self.show_notification(f"Game speed set to {self.speed_multiplier}x")
                            break
                    
                    if self.game_state.is_game_over() and self.restart_button.collidepoint(event.pos):
                        self.restart_game()
                    elif self.game_state.is_setup_phase() and self.random_placement_button.collidepoint(event.pos):
                        self.place_ships_randomly()
                    elif self.game_state.is_setup_phase() and self.game_state.is_ship_placement_complete() and self.start_button.collidepoint(event.pos):
                        self.game_state.start_playing_phase()
                        self.show_notification("Game started! Your turn!")
                    elif self.ai_vs_ai_button.collidepoint(event.pos):
                        # Toggle auto-restart state
                        self.auto_restart = not self.auto_restart
                        if self.auto_restart:
                            # Set both players to algorithmic AI
                            self.change_player_type("ai")
                            self.change_player_ai_type("algorithmic")
                            self.change_ai_type("algorithmic")
                            # Set speed to x3
                            self.speed_multiplier = 3
                            # Place ships randomly and start game
                            self.place_ships_randomly()
                            self.game_state.start_playing_phase()
                            self.show_notification("AI vs AI battle started with auto-restart!")
                        else:
                            self.show_notification("Auto-restart disabled")
                    # Check player side buttons
                    elif self.player_human_button.collidepoint(event.pos):
                        self.change_player_type("human")
                    elif self.player_algo_button.collidepoint(event.pos):
                        self.change_player_type("ai")
                        self.change_player_ai_type("algorithmic")
                    elif self.player_trans_button.collidepoint(event.pos):
                        self.change_player_type("ai")
                        self.change_player_ai_type("transformer")
                    # Check enemy side buttons
                    elif self.enemy_algo_button.collidepoint(event.pos):
                        self.change_ai_type("algorithmic")
                    elif self.enemy_trans_button.collidepoint(event.pos):
                        self.change_ai_type("transformer")
                    elif not self.game_state.is_ai_turn() and self.player_type == "human":
                        self.handle_click(event.pos)
                elif event.button == 3 and not self.game_state.is_ai_turn() and self.player_type == "human":
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
                
                # Log the move
                self.game_logger.log_move(grid_x, grid_y, hit)
                
                if hit:
                    self.show_notification(f"Hit! {message}")
                    if self.ai_board.is_all_ships_destroyed():
                        self.game_state.set_game_over()
                        self.show_notification("Game Over! You win!")
                        self.game_logger.log_game_end("player")
                else:
                    self.show_notification("Miss!")
                    pygame.time.wait(1000 // self.speed_multiplier)  # Adjusted wait time
                    self.game_state.switch_turn()
                    
                    # Clear notification and show AI turn
                    self.show_notification("AI's turn...")
                    pygame.time.wait(1500 // self.speed_multiplier)  # Adjusted wait time
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
        
        # Log the move
        self.game_logger.log_move(x, y, hit)
        
        # Show AI's shot result
        if hit:
            self.show_notification(f"AI hits at {chr(65 + x)}{y + 1}! {message}")
            if self.player_board.is_all_ships_destroyed():
                self.game_state.set_game_over()
                self.show_notification("Game Over! AI wins!")
                self.game_logger.log_game_end("ai")
                # Auto-restart if enabled
                if self.auto_restart:
                    self.restart_game()
                    self.place_ships_randomly()
                    self.game_state.start_playing_phase()
            else:
                # AI gets another turn after a hit
                pygame.time.wait(500 // self.speed_multiplier)  # Reduced base wait time and apply speed
                self.handle_ai_turn()
        else:
            self.show_notification(f"AI misses at {chr(65 + x)}{y + 1}")
            pygame.time.wait(333 // self.speed_multiplier)  # Reduced base wait time and apply speed
            self.game_state.switch_turn()
            self.show_notification("Your turn!")

    def draw(self):
        """Draw the game state"""
        self.screen.fill(WHITE)
        
        # Draw status and notifications first
        self.draw_status()
        
        # Draw both boards
        self.board_renderer.draw_grid(
            self.player_board,
            self.player_board_rect.x,
            self.player_board_rect.y,
            hide_ships=False
        )
        
        self.board_renderer.draw_grid(
            self.ai_board,
            self.ai_board_rect.x,
            self.ai_board_rect.y,
            hide_ships=True
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
        
        # Draw speed control buttons
        for i in range(3):
            button_center = (self.speed_button_x, self.speed_button_y + i * (self.speed_button_radius * 2 + self.speed_button_spacing))
            color = (100, 100, 255) if self.speed_multiplier == i + 1 else (200, 200, 200)
            pygame.draw.circle(self.screen, color, button_center, self.speed_button_radius)
            pygame.draw.circle(self.screen, (150, 150, 150), button_center, self.speed_button_radius, 2)
            speed_text = self.small_font.render(f"x{i+1}", True, (0, 0, 0))
            speed_rect = speed_text.get_rect(center=button_center)
            self.screen.blit(speed_text, speed_rect)

        # Draw turn indicator below the grids only during playing phase
        if self.game_state.is_playing_phase():
            turn_color = (0, 255, 0) if self.game_state.is_player_turn else (255, 0, 0)
            turn_text = "Your turn - Click on enemy board to shoot" if self.game_state.is_player_turn and self.player_type == "human" else "AI's turn..."
            turn_surface = self.small_font.render(turn_text, True, turn_color)
            turn_rect = turn_surface.get_rect(
                centerx=self.screen.get_rect().centerx,
                top=self.player_board_rect.bottom + 20
            )
            self.screen.blit(turn_surface, turn_rect)
        
        # Draw notifications below turn indicator
        if self.notification:
            notification_surface = self.small_font.render(self.notification, True, (0, 0, 0))
            notification_rect = notification_surface.get_rect(
                centerx=self.screen.get_rect().centerx,
                top=self.player_board_rect.bottom + 60
            )
            self.screen.blit(notification_surface, notification_rect)

        # Draw panels
        # Player panel
        pygame.draw.rect(self.screen, (240, 240, 240), self.player_panel_rect)
        pygame.draw.rect(self.screen, (200, 200, 200), self.player_panel_rect, 2)

        # Player buttons
        # Human button
        human_color = (100, 100, 255) if self.player_type == "human" else (200, 200, 200)
        pygame.draw.rect(self.screen, human_color, self.player_human_button)
        pygame.draw.rect(self.screen, (150, 150, 150), self.player_human_button, 2)
        human_text = self.small_font.render("Human", True, (0, 0, 0))  # Using small font
        human_rect = human_text.get_rect(center=self.player_human_button.center)
        self.screen.blit(human_text, human_rect)

        # Player Algo button
        player_algo_color = (100, 100, 255) if self.player_type == "ai" and self.player_ai_type == "algorithmic" else (200, 200, 200)
        pygame.draw.rect(self.screen, player_algo_color, self.player_algo_button)
        pygame.draw.rect(self.screen, (150, 150, 150), self.player_algo_button, 2)
        player_algo_text = self.small_font.render("Algo AI", True, (0, 0, 0))  # Using small font
        player_algo_rect = player_algo_text.get_rect(center=self.player_algo_button.center)
        self.screen.blit(player_algo_text, player_algo_rect)

        # Player Trans button
        player_trans_color = (255, 100, 255) if self.player_type == "ai" and self.player_ai_type == "transformer" else (200, 200, 200)
        pygame.draw.rect(self.screen, player_trans_color, self.player_trans_button)
        pygame.draw.rect(self.screen, (150, 150, 150), self.player_trans_button, 2)
        player_trans_text = self.small_font.render("Trans AI", True, (0, 0, 0))  # Using small font
        player_trans_rect = player_trans_text.get_rect(center=self.player_trans_button.center)
        self.screen.blit(player_trans_text, player_trans_rect)

        # Enemy panel
        pygame.draw.rect(self.screen, (240, 240, 240), self.enemy_panel_rect)
        pygame.draw.rect(self.screen, (200, 200, 200), self.enemy_panel_rect, 2)

        # Enemy buttons
        # Enemy Algo button
        enemy_algo_color = (100, 100, 255) if self.enemy_ai_type == "algorithmic" else (200, 200, 200)
        pygame.draw.rect(self.screen, enemy_algo_color, self.enemy_algo_button)
        pygame.draw.rect(self.screen, (150, 150, 150), self.enemy_algo_button, 2)
        enemy_algo_text = self.small_font.render("Algo AI", True, (0, 0, 0))  # Using small font
        enemy_algo_rect = enemy_algo_text.get_rect(center=self.enemy_algo_button.center)
        self.screen.blit(enemy_algo_text, enemy_algo_rect)

        # Enemy Trans button
        enemy_trans_color = (255, 100, 255) if self.enemy_ai_type == "transformer" else (200, 200, 200)
        pygame.draw.rect(self.screen, enemy_trans_color, self.enemy_trans_button)
        pygame.draw.rect(self.screen, (150, 150, 150), self.enemy_trans_button, 2)
        enemy_trans_text = self.small_font.render("Trans AI", True, (0, 0, 0))  # Using small font
        enemy_trans_rect = enemy_trans_text.get_rect(center=self.enemy_trans_button.center)
        self.screen.blit(enemy_trans_text, enemy_trans_rect)

        # Draw random placement button during setup phase
        if self.game_state.is_setup_phase():
            # Draw button background
            pygame.draw.rect(self.screen, (200, 200, 200), self.random_placement_button)
            pygame.draw.rect(self.screen, (150, 150, 150), self.random_placement_button, 2)
            
            # Draw wrapped text
            text = "Random\nPlace"
            lines = text.split('\n')
            line_height = self.small_font.get_height()
            total_height = line_height * len(lines)
            start_y = self.random_placement_button.y + (self.random_placement_button.height - total_height) // 2
            
            for i, line in enumerate(lines):
                text_surface = self.small_font.render(line, True, (0, 0, 0))
                text_rect = text_surface.get_rect(
                    centerx=self.random_placement_button.centerx,
                    top=start_y + i * line_height
                )
                self.screen.blit(text_surface, text_rect)

        # Draw restart button if game is over
        if self.game_state.is_game_over():
            pygame.draw.rect(self.screen, (200, 200, 200), self.restart_button)
            pygame.draw.rect(self.screen, (150, 150, 150), self.restart_button, 2)
            restart_text = self.small_font.render("Restart Game", True, (0, 0, 0))  # Using small font
            restart_rect = restart_text.get_rect(center=self.restart_button.center)
            self.screen.blit(restart_text, restart_rect)

        # Draw start button
        if self.game_state.is_setup_phase() and self.game_state.is_ship_placement_complete():
            pygame.draw.rect(self.screen, (0, 200, 0), self.start_button)  # Changed to green
            pygame.draw.rect(self.screen, (150, 150, 150), self.start_button, 2)
            start_text = self.small_font.render("Start Game", True, (0, 0, 0))  # Using small font
            start_rect = start_text.get_rect(center=self.start_button.center)
            self.screen.blit(start_text, start_rect)

        # Draw AI vs AI button
        button_color = (0, 200, 0) if self.auto_restart else (200, 200, 200)
        pygame.draw.rect(self.screen, button_color, self.ai_vs_ai_button)
        pygame.draw.rect(self.screen, (150, 150, 150), self.ai_vs_ai_button, 2)
        ai_vs_ai_text = self.small_font.render("AI vs AI", True, (0, 0, 0))
        ai_vs_ai_rect = ai_vs_ai_text.get_rect(center=self.ai_vs_ai_button.center)
        self.screen.blit(ai_vs_ai_text, ai_vs_ai_rect)
        
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
            status_surface = self.small_font.render(status_text, True, (0, 0, 0))
            status_rect = status_surface.get_rect(
                centerx=self.screen.get_rect().centerx,
                top=10  # Position above the grids
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
                    
                    # Log the move
                    self.game_logger.log_move(x, y, hit)
                    
                    if hit:
                        self.show_notification(f"Player AI hits at {chr(65 + x)}{y + 1}! {message}")
                        if self.ai_board.is_all_ships_destroyed():
                            self.game_state.set_game_over()
                            self.show_notification("Game Over! Player AI wins!")
                            self.game_logger.log_game_end("player_ai")
                            # Auto-restart if enabled
                            if self.auto_restart:
                                self.restart_game()
                                self.place_ships_randomly()
                                self.game_state.start_playing_phase()
                    else:
                        self.show_notification(f"Player AI misses at {chr(65 + x)}{y + 1}")
                        pygame.time.wait(333 // self.speed_multiplier)  # Adjusted wait time
                        self.game_state.switch_turn()
                else:
                    # Enemy AI's turn
                    self.handle_ai_turn()
            
            self.draw()
            self.clock.tick(60)

        pygame.quit() 