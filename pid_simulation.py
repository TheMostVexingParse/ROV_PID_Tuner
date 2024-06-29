# Author: Ozan Özbek
# Date: 2024-06-29
# Description: This Python program simulates the dynamics of a Remotely Operated Vehicle (ROV)
# using a PID controller to stabilize its roll/pitch angle. It models the effects of friction,
# drag, and rotational dynamics. 
# Version: 1.0

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
DEG_TO_RAD = np.pi / 180.0  # Conversion factor from degrees to radians
MIN_FORCE = -40.0  # Minimum force that can be applied by the motor (N)
MAX_FORCE = 40.0  # Maximum force that can be applied by the motor (N)

# Physical constants
WATER_DENSITY = 997.0  # Density of water (kg/m^3)
DRAG_COEFFICIENT = 0.8  # Drag coefficient (dimensionless)
REFERENCE_AREA = 0.13  # Reference area for drag force calculation (m^2)
DAMPING_COEFFICIENT = 0.1  # Damping coefficient for frictional torque (N*m*s/rad)
REAL_RADIUS = 0.203  # Actual radius of the ROV (m)

# ROV parameters
MASS = 7.8  # Mass of the ROV (kg)
MOTOR_RADIUS = 0.15  # Radius of the motor (m)

# Simulation parameters
setpoint = 0.0  # Desired roll angle (rad)
sampling_interval = 0.1  # Sampling interval for the simulation (s)
simulation_time = 8.0  # Total simulation time (s)

# Initial conditions
initial_roll = 40.0 * DEG_TO_RAD  # Initial roll angle (rad)

# PID controller gains
Kp = 311.8  # Proportional gain
Ki = 0  # Integral gain
Kd = 202.8  # Derivative gain

class ROV:
    def __init__(self, mass, motor_radius, initial_roll, rho, C_d, A, b):
        """Initialize the ROV with given parameters."""
        self.mass = mass
        self.motor_radius = motor_radius
        self.roll = initial_roll
        self.angular_velocity = 0
        self.rho = rho
        self.C_d = C_d
        self.A = A
        self.b = b

    def apply_force(self, force, delta_time):
        """
        Apply a force to the ROV and update its state.
        
        Parameters:
        force (float): The force to apply (N).
        delta_time (float): The time interval over which to apply the force (s).
        """
        # Limit the force within the specified range
        force = np.clip(force, MIN_FORCE, MAX_FORCE)
        
        # Calculate the torque applied by the motor
        torque = force * self.motor_radius
        
        # Calculate the velocity of the ROV
        velocity = self.angular_velocity * REAL_RADIUS
        
        # Calculate the drag force
        drag_force = 0.5 * self.rho * self.C_d * self.A * velocity ** 2
        drag_force = -drag_force if velocity > 0 else drag_force
        
        # Calculate the net force and torque
        net_force = force - drag_force
        frictional_torque = self.b * self.angular_velocity
        net_torque = torque - frictional_torque
        
        # Update angular acceleration, velocity, and roll angle
        angular_acceleration = net_torque / self.mass
        self.angular_velocity += angular_acceleration * delta_time
        self.roll += self.angular_velocity * delta_time
        self.roll = self.wrap_angle(self.roll)

    def wrap_angle(self, angle):
        """
        Wrap the angle within the range -π to π.
        
        Parameters:
        angle (float): The angle to wrap (rad).
        
        Returns:
        float: The wrapped angle (rad).
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def get_roll(self):
        """
        Get the current roll angle of the ROV.
        
        Returns:
        float: The current roll angle (rad).
        """
        return self.roll

class PID:
    def __init__(self, kp, ki, kd):
        """Initialize the PID controller with given gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def calculate(self, setpoint, current_value, delta_time):
        """
        Calculate the control output using PID algorithm.
        
        Parameters:
        setpoint (float): The desired setpoint value.
        current_value (float): The current measured value.
        delta_time (float): The time interval over which to calculate (s).
        
        Returns:
        float: The control output.
        """
        error = setpoint - current_value
        self.integral += error * delta_time
        derivative = (error - self.previous_error) / delta_time
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, MIN_FORCE, MAX_FORCE)

# Create instances of ROV and PID controller
rov = ROV(MASS, MOTOR_RADIUS, initial_roll, WATER_DENSITY, DRAG_COEFFICIENT, REFERENCE_AREA, DAMPING_COEFFICIENT)
pid = PID(Kp, Ki, Kd)

# Initialize data storage lists
time_data = []
roll_data = []
force_data = []
angular_velocity_data = []

# Lambda functions to map forces to PWM outputs
map_l_pwm = lambda x: 1500 + (abs(x) ** 0.5) * (1 if x >= 0 else -1) * 500 / MAX_FORCE ** 0.5
map_r_pwm = lambda x: 1500 - (abs(x) ** 0.5) * (1 if x >= 0 else -1) * 500 / MAX_FORCE ** 0.5

# Simulation loop
time = 0
while time <= simulation_time:
    current_roll = rov.get_roll()
    force = pid.calculate(setpoint, current_roll, sampling_interval)
    rov.apply_force(force, sampling_interval)

    # Store simulation data
    time_data.append(time)
    roll_data.append(current_roll)
    force_data.append(force)
    angular_velocity_data.append(rov.angular_velocity)

    time += sampling_interval

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

def update_plot(frame):
    ax1.clear()
    ax1.plot(time_data[:frame], roll_data[:frame], label='Roll (rad)')
    ax1.set_ylabel('Roll (rad)')
    ax1.legend()

    ax2.clear()
    ax2.plot(time_data[:frame], force_data[:frame], label='Left Motor Force (N)', color='orange')
    ax2.plot(time_data[:frame], list(map(lambda x: -x, force_data[:frame])), label='Right Motor Force (N)', color='purple')
    ax2.set_ylabel('Force (N)')
    ax2.legend()

    ax3.clear()
    ax3.plot(time_data[:frame], list(map(map_l_pwm, force_data[:frame])), label='Left Motor PWM Output (ms)', color='red')
    ax3.plot(time_data[:frame], list(map(map_r_pwm, force_data[:frame])), label='Right Motor PWM Output (ms)', color='blue')
    ax3.set_ylabel('PWM Output (ms)')
    ax3.legend()

    ax4.clear()
    ax4.plot(time_data[:frame], angular_velocity_data[:frame], label='Angular Velocity (rad/s)', color='green')
    ax4.set_ylabel('Angular Velocity (rad/s)')
    ax4.legend()

    plt.xlabel('Time (s)')

ani = animation.FuncAnimation(fig, update_plot, frames=len(time_data), interval=100)
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
plt.show()
