from collections import deque

class controller:
    base_voltage = 6
    
    angle_coeff = ()
    lateral_coeff = ()

    Ka = 0
    Kl = 0

    target_angle = 0
    target_lat = 0

    int_lat =  deque(maxlen=50)
    int_angle = deque(maxlen=50)
    
    prev_angle_error = 0
    prev_lateral_error = 0
    
    def __init__(self, angular_constants, lateral_constants, angular_gain_coeff, lateral_gain_coeff):
        self.angle_coeff = angular_constants
        self.lateral_coeff = lateral_constants
        self.Ka = angular_gain_coeff
        self.Kl = lateral_gain_coeff

    def gain(self, traj_angle, traj_lateral, verbose=False):
        angular_error = traj_angle - self.target_angle
        lateral_error = traj_lateral - self.target_lat

        angular_d_term = 0 if self.prev_angle_error == 0 else self.prev_angle_error - angular_error
        lateral_d_term = 0 if self.prev_lateral_error == 0 else self.prev_lateral_error - lateral_error

        angular_gain = angular_error*self.angle_coeff[0] + sum(self.int_angle)*self.angle_coeff[1] + angular_d_term*self.angle_coeff[2]
        lateral_gain = lateral_error*self.lateral_coeff[0] + sum(self.int_lat)*self.lateral_coeff[1] + lateral_d_term*self.lateral_coeff[2]

        self.prev_angle_error = angular_error
        self.prev_lateral_error = lateral_error

        total_gain = angular_gain*self.Ka + lateral_gain*self.Kl

        if verbose:
            print("CONTROLLER COEFSS:\nANGULAR:", self.angle_coeff,"\nLATERAL:", self.lateral_coeff)
            print("Angular error:", angular_error, "\nLateral error:", lateral_error)
            print("Angular gain:", angular_gain, "\nLateral gain:", lateral_gain)
            print("Total gain:", total_gain)
        
        return total_gain

    
