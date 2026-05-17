import pygame
import torch
import sys
# Import your exact class from your file
from model import MyNerualNetwork 

# 1. Instantiate your exact model architecture
input_nodes = 10
hidden_nodes = 16
output_nodes = 1

model = MyNerualNetwork(input_size=input_nodes, hidden_size=hidden_nodes, output_size=output_nodes)

# 2. Extract layer sizes automatically from your code
# This reads your structural setup: [10, 16, 16, 1]
layer_sizes = [
    model.layer1.in_features,
    model.layer1.out_features,
    model.layer2.out_features,
    model.output_layer.out_features
]

# 3. Initialize Pygame Window
pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Your Custom Neural Network Architecture")
clock = pygame.time.Clock()

# Distribute layers evenly across the screen width
layer_x = [int(WIDTH / (len(layer_sizes) + 1) * (i + 1)) for i in range(len(layer_sizes))]

# --- Main Window Loop ---
running = True
while running:
    screen.fill((20, 24, 33)) # Dark slate background
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 1. Draw Synapses (Lines connecting the layers)
    for l in range(len(layer_sizes) - 1):
        for i in range(layer_sizes[l]):       # Current layer nodes
            for j in range(layer_sizes[l+1]):   # Next layer nodes
                # Calculate start (x1, y1) and end (x2, y2) coordinates
                x1 = layer_x[l]
                y1 = int(HEIGHT / (layer_sizes[l] + 1) * (i + 1))
                x2 = layer_x[l+1]
                y2 = int(HEIGHT / (layer_sizes[l+1] + 1) * (j + 1))
                
                # Draw a subtle connection line
                pygame.draw.line(screen, (60, 75, 100), (x1, y1), (x2, y2), 1)

    # 2. Draw Neurons (Circles representing nodes)
    for l, size in enumerate(layer_sizes):
        for i in range(size):
            x = layer_x[l]
            y = int(HEIGHT / (size + 1) * (i + 1))
            
            # Color code layers: Input = Green, Hidden = Blue, Output = Red
            if l == 0:
                color = (46, 204, 113)
            elif l == len(layer_sizes) - 1:
                color = (231, 76, 60)
            else:
                color = (52, 152, 219)
                
            # Draw the node border and inner fill
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 12)
            pygame.draw.circle(screen, color, (x, y), 10)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()