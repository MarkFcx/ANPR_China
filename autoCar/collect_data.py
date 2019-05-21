# python collect_training_data.py


import cv2
import pygame
from pygame.locals import *
import os


class CollectTrainingData(object):

    def __init__(self):

        pygame.init()

    def collect(self):

        saved_1_frame = 0
        saved_2_frame = 0
        saved_3_frame = 0

        camera = cv2.VideoCapture(0)

        while True:

            (grabbed, image) = camera.read()

            if not grabbed:
                break

            cv2.imshow('image', image)

            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    key_input = pygame.key.get_pressed()

                    if key_input[pygame.K_UP]:
                        print("cup")
                        cv2.imwrite('training_images/cup/{:>05}.jpg'.format(saved_1_frame), image)
                        saved_1_frame += 1

                    elif key_input[pygame.K_LEFT]:
                        print("molihua")
                        cv2.imwrite('training_images/molihua/{:>05}.jpg'.format(saved_2_frame), image)
                        saved_2_frame += 1


                    elif key_input[pygame.K_RIGHT]:
                        print("weiC")
                        cv2.imwrite('training_images/weiC/{:>05}.jpg'.format(saved_3_frame), image)
                        saved_3_frame += 1


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("[INFO] Saved cup frame: ", saved_1_frame)
        print("[INFO] Saved molihua frame: ", saved_2_frame)
        print("[INFO] Saved weiC frame: ", saved_3_frame)



if __name__ == '__main__':

    ctd = CollectTrainingData()
    ctd.collect()
