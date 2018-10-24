import pygame
from pygame import Rect

pygame.init()

win = pygame.display.set_mode((500,500))
pygame.display.set_caption("Sphero AI Simulator")

class Sphero(object):
    def __init__(self):       
        self.x = 50
        self.y = 50
        self.width = 40
        self.height = 40
        self.vel = 5
        self.col = (255,255,255)
        self.sprite = pygame.Rect(self.x,self.y,self.width,self.height)

    def update(self,keys):
        if keys[pygame.K_RIGHT]:
            self. x += self.vel

        self.sprite = pygame.Rect(self.x,self.y,self.width,self.height)
        

run = True

sphero = Sphero()


while run:
    pygame.time.delay(10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    sphero.update(keys)
    
    win.fill((0,0,0))
    pygame.draw.rect(win,sphero.col,sphero.sprite)
    pygame.display.update()


pygame.quit()