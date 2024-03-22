import board
import busio
import adafruit_drv2605
import math
import numpy as np

def innitMotorVars():
    i2c = busio.I2C(board.SCL, board.SDA)
    drv = adafruit_drv2605.DRV2605(i2c)
    drv.realtime_value = 0
    drv.mode = adafruit_drv2605.MODE_REALTIME
    return i2c, drv

def motorOutput(depthMapArray, i2c, drv, a=2.9, c=4):
    
    #Gets the smallest number from the depth map array
    minDistance = np.amin(depthMapArray)

    #Calculates the intensity we want to have of the motor
    intensity = (1/(1+math.e**(a*(minDistance/1000) - c)))

    #Changes the intensity of the motor. (127 is 100% intensity)
    drv.realtime_value = 127*intensity

    return

    




