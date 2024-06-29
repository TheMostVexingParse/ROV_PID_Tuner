/*
Author: Ozan Özbek
Date: 2024-06-29
Description: This C++ program simulates the dynamics of a Remotely Operated Vehicle (ROV)
using a PID controller to stabilize its roll/pitch angle. It models the effects of friction,
drag, and rotational dynamics, optimizing PID parameters to minimize roll angle error.
Version: 1.0

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

// Constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double DEG_TO_RAD = M_PI / 180.0;
const double MIN_FORCE = -20.0;
const double MAX_FORCE = 20.0;
const double WATER_DENSITY = 997.0; 
const double DRAG_COEFFICIENT = 0.8;                    // Your own value
const double REFERENCE_AREA = 0.13;                     // Area of the top or bottom surface. Use the average if different
const double DAMPING_COEFFICIENT = 0.1;                 // Your own value
const double REAL_RADIUS = 0.203;                       // Perpendicular distance between the roll (or pitch, depending on the use) axis of the vehicle and one of its sides
const double MASS = 7.8;                                // Mass of your vehicle in kilograms
const double MOTOR_RADIUS = 0.15;                       // Perpendicular distance between the roll (or pitch, depending on the use) axis of the vehicle and center of the engines 
const double SETPOINT = 0.0;                            // Target roll (or pitch, depending on the use)
const double SAMPLING_INTERVAL = 0.1;                   // Interval of sampling i.e. interval between each measurement of roll (or pitch, depending on the use)
const double SIMULATION_TIME = 5.0;                     // Duration of the simulation
const double INITIAL_ROLL_START = -45.0 * DEG_TO_RAD;
const double INITIAL_ROLL_END = 45.0 * DEG_TO_RAD;
const double INITIAL_ROLL_STEP = 15.0 * DEG_TO_RAD;     // Angle difference between each run, lower the better
const int KP_SAMPLES = 5000;                            // How many different values of Kp will be tested (should be greater than 0)
const int KI_SAMPLES = 2;                               // How many different values of Ki will be tested (should be greater than 0)
const int KD_SAMPLES = 5000;                            // How many different values of Kd will be tested (should be greater than 0)
const double MAX_KP_VAL = 500.0;                        // Kp will be tested between [0, MAX_KP_VAL]
const double MAX_KI_VAL = 5.0;                          // Ki will be tested between [0, MAX_KI_VAL]
const double MAX_KD_VAL = 500.0;                        // Kd will be tested between [0, MAX_KD_VAL]

// Function to wrap angle within -π to π
double wrapAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// Template function to clamp value within min and max
template<typename T>
T clamp(T value, T min_value, T max_value) {
    return std::min(std::max(value, min_value), max_value);
}

// Class representing the ROV (Remotely Operated Vehicle)
class ROV {
public:
    ROV(double mass, double motor_radius, double initial_roll, double rho, double C_d, double A, double b)
        : mass(mass), motor_radius(motor_radius), roll(initial_roll), angular_velocity(0), rho(rho), C_d(C_d), A(A), b(b) {}

    // Function to apply force and update the state of the ROV
    void applyForce(double force, double delta_time) {
        force = clamp(force, MIN_FORCE, MAX_FORCE);
        double torque = force * motor_radius; // Torque = Force * Motor Radius

        double velocity = angular_velocity * REAL_RADIUS;
        double drag_force = 0.5 * rho * C_d * A * velocity * velocity; // Drag Force = 0.5 * rho * C_d * A * velocity^2
        drag_force = (velocity > 0) ? -drag_force : drag_force; 

        double net_force = force - drag_force;
        double frictional_torque = b * angular_velocity; // Frictional Torque = Damping Coefficient * Angular Velocity
        double net_torque = torque - frictional_torque;

        double angular_acceleration = net_torque / mass; // Angular Acceleration = Net Torque / Mass
        angular_velocity += angular_acceleration * delta_time;
        roll += angular_velocity * delta_time;
        roll = wrapAngle(roll);
    }

    // Function to get the current roll of the ROV
    double getRoll() const {
        return roll;
    }

private:
    double mass;
    double motor_radius;
    double roll;
    double angular_velocity;
    double rho;
    double C_d;
    double A;
    double b;
};

// Class representing the PID controller
class PID {
public:
    PID(double kp, double ki, double kd)
        : kp(kp), ki(ki), kd(kd), previous_error(0), integral(0) {}

    // Function to calculate the control output
    double calculate(double setpoint, double current_value, double delta_time) {
        double error = setpoint - current_value;
        integral += error * delta_time;
        double derivative = (error - previous_error) / delta_time;
        previous_error = error;
        double output = kp * error + ki * integral + kd * derivative;
        return clamp(output, MIN_FORCE, MAX_FORCE); 
    }

    // Function to set new PID parameters
    void setParameters(double new_kp, double new_ki, double new_kd) {
        kp = new_kp;
        ki = new_ki;
        kd = new_kd;
    }

private:
    double kp, ki, kd;
    double previous_error;
    double integral;
};

// Function to calculate the cost for a given PID controller
double calculateCost(const PID& pid, double mass, double motor_radius, double initial_roll_start, double initial_roll_end, double initial_roll_step, double setpoint, double sampling_interval, double simulation_time) {
    double total_error = 0.0;
    int count = 0;

    // Iterate over different initial roll angles
    for (double initial_roll = initial_roll_start; initial_roll <= initial_roll_end; initial_roll += initial_roll_step) {
        if (initial_roll == 0.0) continue;
        ROV rov(mass, motor_radius, initial_roll, WATER_DENSITY, DRAG_COEFFICIENT, REFERENCE_AREA, DAMPING_COEFFICIENT);

        // Simulate the system over time
        for (double t = 0; t <= simulation_time; t += sampling_interval) {
            double current_roll = rov.getRoll();
            double force = pid.calculate(setpoint, current_roll, sampling_interval);
            rov.applyForce(force, sampling_interval);
            total_error += pow(setpoint - current_roll, 2); // Sum of squared errors
        }
        count++;
    }
    return total_error / count; // Average cost over all initial conditions
}

// Function to perform a grid search to find the best PID parameters
void gridSearch(double& best_kp, double& best_ki, double& best_kd, double mass, double motor_radius, double initial_roll_start, double initial_roll_end, double initial_roll_step, double setpoint, double sampling_interval, double simulation_time, int kp_samples, int ki_samples, int kd_samples) {
    double best_cost = std::numeric_limits<double>::infinity();

    const double kp_step = MAX_KP_VAL / kp_samples;
    const double ki_step = MAX_KI_VAL / ki_samples;
    const double kd_step = MAX_KD_VAL / kd_samples;

    // Iterate over a grid of possible PID parameters
    for (int i = 0; i < kp_samples; ++i) {
        double kp = i * kp_step;
        for (int j = 0; j < ki_samples; ++j) {
            double ki = j * ki_step;
            for (int k = 0; k < kd_samples; ++k) {
                double kd = k * kd_step;
                PID pid(kp, ki, kd);
                double cost = calculateCost(pid, mass, motor_radius, initial_roll_start, initial_roll_end, initial_roll_step, setpoint, sampling_interval, simulation_time);

                if (cost < best_cost) {
                    best_cost = cost;
                    best_kp = kp;
                    best_ki = ki;
                    best_kd = kd;
                }
            }
        }

        // Print the best parameters found so far
        std::cout << "Iteration " << i << ": \tKp=" << std::setw(2) << std::setprecision(8) << std::setw(4) << best_kp 
                  << "  \tKi=" << std::setw(2) << std::setprecision(8) << std::setw(4) << best_ki 
                  << "   \tKd=" << std::setw(2) << std::setprecision(8) << std::setw(4) << best_kd 
                  << "   \tKp/Kd=" <<  std::setw(10) << best_kp / best_kd
                  << "   \tcost=" << best_cost << std::endl;
    }

    std::cout << "Best parameters found: kp = " << best_kp << ", ki = " << best_ki << ", kd = " << best_kd << ", Cost: " << best_cost << std::endl;
}

int main() {
    double best_kp = 0.0;
    double best_ki = 0.0;
    double best_kd = 0.0;

    // Perform the grid search for the best PID parameters
    gridSearch(best_kp, best_ki, best_kd, MASS, MOTOR_RADIUS, INITIAL_ROLL_START, INITIAL_ROLL_END, INITIAL_ROLL_STEP, SETPOINT, SAMPLING_INTERVAL, SIMULATION_TIME, KP_SAMPLES, KI_SAMPLES, KD_SAMPLES);

    return 0;
}
