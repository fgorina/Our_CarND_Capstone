import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                    wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement

        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        # PID Constants - Experimental values
        kp = 0.3
        ki = 0.1
        kd = 0.005

        # Minimum and maximum throttle values
        mn = 0.0
        mx = 0.5

        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5  # For the LP filter. 1/(2pi*tau) = cutoff frequency
        ts = 0.02   # 1/rate = sample time
        self.vel_lpf = LowPassFilter(tau, ts)

	self.vehicle_mass = vehicle_mass   # Used to compute torque
        self.fuel_capacity = fuel_capacity # If we want to be perfect with the mass
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs

        if not dbw_enabled:  # Manual Drive. Reset PID son I term doesn't integrate
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_vel = self.vel_lpf.filt(current_vel)    # Remove too fast transients in speed

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        #rospy.logwarn("Steer:%.2f"%steering + " Speed:%.2f"%linear_vel + " Real Speed:%.2f"%current_vel)
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400 # N*m. Stop and fix the car in place braking hard

        elif throttle < .2 and vel_error < 0: # too fast, must brake
            throttle = 0
            decel = max(vel_error , self.decel_limit)
            brake = abs(decel) * (self.vehicle_mass+self.fuel_capacity*GAS_DENSITY)*self.wheel_radius # Torque N*s
 
        return throttle, brake, steering
