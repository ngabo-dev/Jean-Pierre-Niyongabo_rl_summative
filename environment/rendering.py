import pygame
import numpy as np

class Renderer:
    """
    Pygame-based renderer for the FluentFusion RL environment.
    Visualizes the agent navigating the lesson grid.
    """

    def __init__(self, grid_size=10):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = 50
        self.screen_size = (grid_size + 1) * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("FluentFusion Language Learning RL Environment")

        self.colors = {
            'background': (255, 255, 255),
            'grid': (200, 200, 200),
            'agent': (0, 0, 255),
            'goal': (0, 255, 0),
            'lesson': (255, 255, 0)
        }

        self.clock = pygame.time.Clock()
        self.fps = 30

    def render(self, state):
        """
        Render the current state of the environment.
        state: [x, y, proficiency]
        """
        self.screen.fill(self.colors['background'])

        # Draw grid lines
        for i in range(self.grid_size + 2):
            # Vertical lines
            pygame.draw.line(self.screen, self.colors['grid'],
                           (i * self.cell_size, 0),
                           (i * self.cell_size, self.screen_size), 1)
            # Horizontal lines
            pygame.draw.line(self.screen, self.colors['grid'],
                           (0, i * self.cell_size),
                           (self.screen_size, i * self.cell_size), 1)

        # Draw lesson positions (yellow circles)
        for pos in [(i, j) for i in range(self.grid_size + 1) for j in range(self.grid_size + 1)]:
            center = (pos[0] * self.cell_size + self.cell_size // 2,
                     pos[1] * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.screen, self.colors['lesson'], center, 10)

        # Draw goal (green square)
        goal_center = (self.grid_size * self.cell_size + self.cell_size // 2,
                      self.grid_size * self.cell_size + self.cell_size // 2)
        pygame.draw.rect(self.screen, self.colors['goal'],
                        (goal_center[0] - 15, goal_center[1] - 15, 30, 30))

        # Draw agent (blue circle)
        x, y, prof = state
        agent_center = (int(x) * self.cell_size + self.cell_size // 2,
                       int(y) * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, self.colors['agent'], agent_center, 20)

        # Display proficiency
        font = pygame.font.SysFont(None, 24)
        prof_text = font.render(f"Proficiency: {prof:.1f}", True, (0, 0, 0))
        self.screen.blit(prof_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.fps)

        # Handle events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self):
        pygame.quit()