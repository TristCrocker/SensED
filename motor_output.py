import board  # CircuitPython board library
import busio  # CircuitPython busio library
import adafruit_pca9685  # Adafruit PCA9685 library
import numpy as np
import multiprocessing 
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
        (0,0) : (0x40, 0), (0,1) : (0x40, 1), (0,2) : (0x40, 2), (0,3) : (0x40, 3), (0,4) : (0x40, 4), (0,5) : (0x40, 5), (0,6) : (0x40, 6), (0,7) : (0x40, 7), 
        (1,0) : (0x40, 8), (1,1) : (0x40, 9), (1,2) : (0x40, 10), (1,3) : (0x40, 11), (1,4) : (0x40, 12), (1,5) : (0x40, 13), (1,6) : (0x40, 14), (1,7) : (0x40, 15), 
        (2,0) : (0x41, 0), (2,1) : (0x41, 1), (2,2) : (0x41, 2), (2,3) : (0x41, 3), (2,4) : (0x41, 4), (2,5) : (0x41, 5), (2,6) : (0x41, 6), (2,7) : (0x41, 7), 
        (3,0) : (0x41, 8), (3,1) : (0x41, 9), (3,2) : (0x41, 10), (3,3) : (0x41, 11), (3,4) : (0x41, 12), (3,5) : (0x41, 13), (3,6) : (0x41, 14), (3,7) : (0x41, 15), 
        (4,0) : (0x42, 0), (4,1) : (0x42, 1), (4,2) : (0x42, 2), (4,3) : (0x42, 3), (4,4) : (0x42, 4), (4,5) : (0x42, 5), (4,6) : (0x42, 6), (4,7) : (0x42, 7), 
        (5,0) : (0x42, 8), (5,1) : (0x42, 9), (5,2) : (0x42, 10), (5,3) : (0x42, 11), (5,4) : (0x42, 12), (5,5) : (0x42, 13), (5,6) : (0x42, 14), (5,7) : (0x42, 15), 
        (6,0) : (0x43, 0), (6,1) : (0x43, 1), (6,2) : (0x43, 2), (6,3) : (0x43, 3), (6,4) : (0x43, 4), (6,5) : (0x43, 5), (6,6) : (0x43, 6), (6,7) : (0x43, 7), 
        (7,0) : (0x43, 8), (7,1) : (0x43, 9), (7,2) : (0x43, 10), (7,3) : (0x43, 11), (7,4) : (0x43, 12), (7,5) : (0x43, 13), (7,6) : (0x43, 14), (7,7) : (0x43, 15), 
        }

        self.i2c = busio.I2C(board.SCL, board.SDA)  # Create the I2C bus
        self.pca9685_instances = {}

        # Create an instance for each address
        for address in i2c_addresses:
            pca9685 = adafruit_pca9685.PCA9685(self.i2c, address=address)
            pca9685.frequency = 60  # Example frequency; adjust as needed 
            self.pca9685_instances[address] = pca9685

        self.replacement_signal = multiprocessing.Event()

    
    def control_motors(self, intensity_pattern_array):
        """
        Controls motors based on an intensity/pattern array using multiprocessing.

        Args:
            intensity_pattern_array (np.ndarray): An 8x8 NumPy array where each 
                                                  element is a tuple of (intensity, pattern_id).
                                                  - intensity (float): Value between 0.0 and 1.0.
                                                  - pattern_id (int):  Determines the vibration pattern.
        """

        self.replacement_signal.set()
        # Wait for processes to complete
        for p in self.processes:
            p.join() 

        def motor_worker(controller_id, motor_data):
            """Worker function to control a single motor"""
            controller = self.pca9685_instances[controller_id[0]]
            channel = controller_id[1]  # Adjust this if using a different channel
            distance, pattern_id = motor_data

            #Calculates the intensity we want to have of the motor
            intensity = (1/(1+math.e**(self.a*(distance/1000) - self.c)))

            duty_cycle = int(0xFFFF * intensity)  

            if pattern_id == 1:  # Example patterns
                controller.channels[channel].duty_cycle = duty_cycle
            elif pattern_id == 2:
                self.replacement_signal = multiprocessing.Event()
                while not self.replacement_signal.is_set():
                    controller.channels[channel].duty_cycle = duty_cycle
                    time.sleep(0.2)
                    controller.channels[channel].duty_cycle = 0
                    time.sleep(0.2)
            # ... Add more patterns ... 

        # Use multiprocessing
        
        self.processes = []
        for row_index, row in enumerate(intensity_pattern_array):
            for col_index, data in enumerate(row):
                controller_id = self.cal[(row_index, col_index)]
                p = multiprocessing.Process(target=motor_worker, 
                                            args=(controller_id, data))
                self.processes.append(p)
                p.start()

        