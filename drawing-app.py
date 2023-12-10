# Description: user to draw small digits, so user can draw their own digits for the ANN handwritten digits prediction script
# Using pygame to draw the interface, then feed it into the neural network

import pygame
 
pygame.init()
screen = pygame.display.set_mode((500,500))
pygame.display.set_caption("My first game")
clock = pygame.time.Clock()
 
loop = True

while loop:
    try:
        #pygame.mouse.set_visible(False)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                loop = False
    
        px, py = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed() == (1,0,0):
            pygame.draw.rect(screen, (255,255,255), (px-15,py-15,30,30))
 
        pygame.display.update()
        clock.tick(1000)
    except Exception as e:
        print(e)
        pygame.quit()
        
pygame.quit()