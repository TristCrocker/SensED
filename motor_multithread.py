import board  # CircuitPython board library
import busio  # CircuitPython busio library
import adafruit_pca9685  # Adafruit PCA9685 library
import numpy as np
from threading import Thread
import time
import math

class PCA9685_Controller:
    """
    Class to control multiple Adafruit PCA9685 motor controllers
    """

    

    def __init__(self, i2c_addresses=[0x40, 0x41, 0x42, 0x43], a=2.9, c=4):
        """
        Initializes the class and creates PCA9685 instances.

        Args:
            i2c_addresses (list, optional): A list of I2C addresses for 
                                            each PCA9685 controller. 
                                            Defaults to [0x40, 0x41, 0x42, 0x43].
            
            a (float, optional): The first variable for the intensity calculation, Default: 2.9.

            c (float, optional): The second variable for the intensity calculation, Default: 4.
        """

        self.a = a
        self.c = c

        # An dictionary for calibrating motor locations with physical locations (2 Tuples (phys_x, phys_y) : (controller_address, motor_channel))
        self.cal = {
        (0,0) : (0x40, 0), (0,1) : (0x40, 8), (0,2) : (0x41, 0), (0,3) : (0x41, 8), (0,4) : (0x42, 0), (0,5) : (0x42, 8), (0,6) : (0x43, 0), (0,7) : (0x43, 8), 
        (1,0) : (0x40, 1), (1,1) : (0x40, 9), (1,2) : (0x41, 1), (1,3) : (0x41, 9), (1,4) : (0x42, 1), (1,5) : (0x42, 9), (1,6) : (0x43, 1), (1,7) : (0x43, 9), 
        (2,0) : (0x40, 2), (2,1) : (0x40, 10), (2,2) : (0x41, 2), (2,3) : (0x41, 10), (2,4) : (0x42, 2), (2,5) : (0x42, 10), (2,6) : (0x43, 2), (2,7) : (0x43, 10), 
        (3,0) : (0x40, 3), (3,1) : (0x40, 11), (3,2) : (0x41, 3), (3,3) : (0x41, 11), (3,4) : (0x42, 3), (3,5) : (0x42, 11), (3,6) : (0x43, 3), (3,7) : (0x43, 11), 
        (4,0) : (0x40, 4), (4,1) : (0x40, 12), (4,2) : (0x41, 4), (4,3) : (0x41, 12), (4,4) : (0x42, 4), (4,5) : (0x42, 12), (4,6) : (0x43, 4), (4,7) : (0x43, 12), 
        (5,0) : (0x40, 5), (5,1) : (0x40, 13), (5,2) : (0x41, 5), (5,3) : (0x41, 13), (5,4) : (0x42, 5), (5,5) : (0x42, 13), (5,6) : (0x43, 5), (5,7) : (0x43, 13), 
        (6,0) : (0x40, 6), (6,1) : (0x40, 14), (6,2) : (0x41, 6), (6,3) : (0x41, 14), (6,4) : (0x42, 6), (6,5) : (0x42, 14), (6,6) : (0x43, 6), (6,7) : (0x43, 14), 
        (7,0) : (0x40, 7), (7,1) : (0x40, 15), (7,2) : (0x41, 7), (7,3) : (0x41, 15), (7,4) : (0x42, 7), (7,5) : (0x42, 15), (7,6) : (0x43, 7), (7,7) : (0x43, 15), 
        }

        self.i2c = busio.I2C(board.SCL, board.SDA)  # Create the I2C bus
        self.pca9685_instances = {}

        # Create an instance for each address
        for address in i2c_addresses:
            pca9685 = adafruit_pca9685.PCA9685(self.i2c, address=address)
            pca9685.frequency = 60  # Example frequency; adjust as needed 
            self.pca9685_instances[address] = pca9685

        self.intensity_pattern_array = [[(0, 0) for col in range(8)] for row in range(8)]

        self.threads = []

        self.terminate = False

    
    def start_motors(self, intensity_pattern_array=None):
        """
        Controls motors based on an intensity/pattern array using multithreading.

        Args:
            intensity_pattern_array (np.ndarray): An 8x8 NumPy array where each 
                                                  element is a tuple of (intensity, pattern_id).
                                                  - intensity (float): Value between 0.0 and 1.0.
                                                  - pattern_id (int):  Determines the vibration pattern.
        """

        if (intensity_pattern_array != None):
            self.intensity_pattern_array = intensity_pattern_array

        def motor_worker(controller_id, motor_id):
            """Worker function to control a single motor"""
            if (controller_id[0] not in self.pca9685_instances):
                    return
            
            controller = self.pca9685_instances[controller_id[0]]
            channel = controller_id[1]  # Adjust this if using a different channel

            while True:

                if (self.terminate == True):
                    controller.channels[channel].duty_cycle = 0
                    break
                
                distance, pattern_id = self.intensity_pattern_array[motor_id[0]][motor_id[1]]

                #Calculates the intensity we want to have of the motor
                intensity = (1/(1+math.e**(self.a*(distance/1000) - self.c)))

                duty_cycle = int(0xFFFF * intensity)  

                if pattern_id == 1:  # Example patterns
                    controller.channels[channel].duty_cycle = duty_cycle
                elif pattern_id == 2:
                    controller.channels[channel].duty_cycle = duty_cycle
                    time.sleep(0.2)
                    controller.channels[channel].duty_cycle = 0
                    time.sleep(0.2)
                # ... Add more patterns ... 

        
        for row_index, row in enumerate(intensity_pattern_array):
            for col_index, data in enumerate(row):
                controller_id = self.cal[(row_index, col_index)]
                t = Thread(target=motor_worker, 
                                            args=(controller_id, (row_index, col_index)))
                self.threads.append(t)

        for t in self.threads:
            t.start()


    def update_intensities(self, intensity_pattern_array):
        self.intensity_pattern_array = intensity_pattern_array


    def stop_motors(self):
        self.terminate = True

        for t in self.threads:
            t.join()

    def singleSleaveControl(self, intensity_pattern_array, shielMode=False):
        '''Merges the array into a 8x4 by taking the max of the pairs of arrays in rows Also takes the largest pattern ID as the pattern'''
        '''Shield mode (as suggested by Tim) takes the original array and transposes it so that the 8 motors display the image horizontally rather than vertically
        If the user holds their arm like a shield it should display in accordance to that'''
        
        if (shielMode):
            intensity_pattern_array = np.transpose(intensity_pattern_array)

        # Separate the intensity values from the pattern IDs
        intensities = intensity_pattern_array[:, :, 0]

        # Reshape the intensities array into four 2x8 subarrays
        subarrays = intensities.reshape(4, 2, 8)

        # Calculate the average intensity for each subarray
        averages = np.max(subarrays, axis=1)
        merged_array = averages.reshape(8, 4)

        # Replace the intensity values in the merged array with tuples
        for row in range(8):
            for col in range(4):
                if (intensity_pattern_array[row, col, 1] > intensity_pattern_array[row, col+1, 1]):
                    pattern = intensity_pattern_array[row, col, 1]
                else:
                    pattern = intensity_pattern_array[row, col+1, 1]

                merged_array[row, col] = (merged_array[row, col], pattern)

        
        self.control_motors(merged_array)