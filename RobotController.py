import time
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from dynamixel_sdk import PortHandler, PacketHandler
import cv2

class RobotController:
    def __init__(self, debug=False,
                 DEVICENAME="/dev/tty.usbmodem113201",
                 PROTOCOL_VERSION=1.0,
                 BAUDRATE=1000000,
                 ADDR_MX_TORQUE_ENABLE=64,
                 ADDR_MX_GOAL_POSITION=30,
                 ADDR_MX_PRESENT_POSITION=132,
                 TORQUE_ENABLE=1,
                 TORQUE_DISABLE=0,
                 motor_ids=[1, 2, 3, 4]):
        self.ADDR_MX_TORQUE_ENABLE = ADDR_MX_TORQUE_ENABLE
        self.ADDR_MX_GOAL_POSITION = ADDR_MX_GOAL_POSITION
        self.ADDR_MX_PRESENT_POSITION = ADDR_MX_PRESENT_POSITION
        self.TORQUE_ENABLE = TORQUE_ENABLE
        self.TORQUE_DISABLE = TORQUE_DISABLE
        self.motor_ids = motor_ids

        self.joint_limits = [-130, 130, -180, 0, -100, 100, -100, 100]

        self.dh_params = np.array([[50, 0, -0.5*pi, 0],
                                   [0, 93, 0, 0],
                                   [0, 93, 0, 0],
                                   [0, 50, 0, 0],
                                   ])
        
        if debug:
            print("RobotController initialized in debug mode.")
            
            plt.ion()  # Turn on interactive mode
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.link_plot, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=8, color='royalblue', label='Robot Links')
            self.ee_plot = self.ax.scatter([], [], [], color='red', s=100, label='End-Effector')
            self.ax.set_xlim(-0.2, 0.2)
            self.ax.set_ylim(-0.2, 0.2)
            self.ax.set_zlim(-0.2, 0.2)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('Robot Visualization')
            self.ax.legend()
            plt.show()

            return
        
        
        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.portHandler = PortHandler(DEVICENAME)

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()

        # Set port baudrate
        if self.portHandler.setBaudRate(BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()

        self.Motor_offsets = [152, 63, 147, 150]
        self.enable_motors()

    # ----------------------
    # Helper functions
    # ----------------------

    def Move_motor(self, angles):
        for i in range(len(angles)):
            angle = self.Motor_offsets[i]+angles[i]
            try:
                dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                        self.portHandler,
                        i+1,
                        self.ADDR_MX_GOAL_POSITION,
                        int(round(angle / 0.29))
                )
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received, stopping the loop.")
                # You can also add any cleanup code here if needed, e.g., closing port
                self.portHandler.closePort()
        

    def enable_motors(self):
        """Enable torque on all motors in motor_ids."""
        for motor_id in self.motor_ids:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler,
                                                                           motor_id,
                                                                           self.ADDR_MX_TORQUE_ENABLE,
                                                                           self.TORQUE_ENABLE)
            if dxl_comm_result != 0:
                print(self.packetHandler.getTxRxResult(dxl_comm_result))
            if dxl_error != 0:
                print(self.packetHandler.getRxPacketError(dxl_error))
            else:
                print(f"Motor {motor_id}: torque enabled")
 
    def disable_motors(self):
        """Disable torque on all motors and close port."""
        for motor_id in self.motor_ids:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler,
                                                                           motor_id,
                                                                           self.ADDR_MX_TORQUE_ENABLE,
                                                                           self.TORQUE_DISABLE)
            if dxl_comm_result != 0:
                print(self.packetHandler.getTxRxResult(dxl_comm_result))
            if dxl_error != 0:
                print(self.packetHandler.getRxPacketError(dxl_error))
            else:
                print(f"Motor {motor_id}: torque disabled")
        # close and cleanup
        try:
            self.portHandler.closePort()
            print("Port closed.")
        except Exception as e:
            print("Error closing port:", e)
        self.init_status = 0

    def set_speed(self, speeds):
        """
        Set individual motor speeds between >0 and 1 (normalized).
        speeds: iterable with same length as motor_ids
        """
        speeds = list(speeds)
        for i, motor_id in enumerate(self.motor_ids):
            v = speeds[i]
            if v > 0 and v <= 1:
                speed_value = int(v * 1023)
                dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler,
                                                                               motor_id,
                                                                               32,   # Moving Speed address for AX/MX
                                                                               speed_value)
                if dxl_comm_result != 0:
                    print(self.packetHandler.getTxRxResult(dxl_comm_result))
                if dxl_error != 0:
                    print(self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("Movement speed out of range for motor", motor_id, "enter value between (0,1]")

    def set_torque_limit(self, torques):
        """
        Set individual motor torque limits between >0 and 1 (normalized).
        """
        torques = list(torques)
        for i, motor_id in enumerate(self.motor_ids):
            t = torques[i]
            if t > 0 and t <= 1:
                torque_value = int(t * 1023)
                dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler,
                                                                               motor_id,
                                                                               34,  # Torque limit address
                                                                               torque_value)
                if dxl_comm_result != 0:
                    print(self.packetHandler.getTxRxResult(dxl_comm_result))
                if dxl_error != 0:
                    print(self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("Torque limit out of range for motor", motor_id, "enter value between (0,1]")

    def read_joint_angles(self):
        """
        Read all motor joint positions and update self.joint_angle_error.
        Returns j_a list of current joint angles (deg, offset removed).
        """
        j_a = [0.0]*len(self.motor_ids)
        for i, motor_id in enumerate(self.motor_ids):
            dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler,
                                                                                               motor_id,
                                                                                               self.ADDR_MX_PRESENT_POSITION)
            if dxl_comm_result != 0:
                print(self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print(self.packetHandler.getRxPacketError(dxl_error))
            else:
                # convert and remove offsets
                deg = self.rot_to_deg(dxl_present_position) - self.joint_offsets[i]
                j_a[i] = deg
                self.joint_angle_error[i] = deg - self.joint_angles[i]
        return j_a
    
    def read_ee_position(self):
        """Return end-effector cartesian coordinates computed from current encoder readings."""
        j_a = self.read_joint_angles()
        return self.forward(j_a)
    
    def check_limits(self, deg, motor_id):
        """Check if the given angle (deg) is within joint limits for the specified motor_id."""
        idx = self.motor_ids.index(motor_id)
        lower_limit = self.joint_limits[2*idx]
        upper_limit = self.joint_limits[2*idx + 1]
        if deg < lower_limit or deg > upper_limit:
            print(f"Angle {deg}° for motor {motor_id} exceeds limits ({lower_limit}°, {upper_limit}°).")
            return False
        return True
    
    # ----------------------
    # Kinematic functions
    # ----------------------
    
    def make_DH_matrix(self, d, a, alpha, theta):
        """Create individual DH transformation matrix."""
        T = np.array([[math.cos(theta), -math.sin(theta)*math.cos(alpha),  math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
                      [math.sin(theta),  math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
                      [0,                math.sin(alpha),                  math.cos(alpha),                 d],
                      [0,                0,                                0,                               1]])
        return T
    
    def forward(self, joint_angles):
        """
        Compute forward kinematics for given joint angles (deg).
        Returns a list of joint positions [[x0,y0,z0], [x1,y1,z1], ..., [xe,ye,ze]].
        """
        if len(joint_angles) != len(self.motor_ids):
            raise ValueError("Length of joint_angles must match number of motors.")
        
        T = np.eye(4)
        joint_positions = [T[0:3, 3].copy()]  # start with base position
        
        for i, angle in enumerate(joint_angles):
            theta = math.radians(angle)  # Convert to radians
            d, a, alpha, _ = self.dh_params[i]
            A_i = self.make_DH_matrix(d, a, alpha, theta)
            T = np.dot(T, A_i)
            joint_positions.append(T[0:3, 3].copy())  # store position of this joint/end-effector
        
        return joint_positions
    
    def inverse(self, x3, y3, z3, theta=0):
        """
        Compute inverse kinematics for given target_position [x, y, z].
        Returns joint angles [theta0, theta1, theta2, theta3] in degrees.
        Elbow-down configuration.
        """
        # Robot DH parameters
        d1 = self.dh_params[0][0]  # base height
        a2 = self.dh_params[1][1]  # link 2 length
        a3 = self.dh_params[2][1]  # link 3 length
        a4 = self.dh_params[3][1]  # link 3 length

        # Base rotation
        theta1 = math.atan2(y3, x3)

        # Planar distance to target
        x = math.sqrt(x3**2 + y3**2)
        y = z3 - d1
        
        #inverse Kinamatics
        cos_q1 = (x**2 + y**2 - a2**2 - a3**2) / (2 * a2 * a3)
        cos_q1 = max(-1, min(1, cos_q1))# Clamp to [-1, 1] to avoid numerical issues
        sin_q1 = -math.sqrt(1 - cos_q1**2)# Compute sin(q1)
        q1 = math.atan2(sin_q1, cos_q1)# Use atan2 for correct quadrant

        q0 = math.atan2(y, x) - math.atan2(a3 * math.sin(q1),a2 + a3 * math.cos(q1))

        q2 = theta - q0 - q1

        return math.degrees(theta1), math.degrees(q0), math.degrees(q1), math.degrees(q2)
    
    def inverse_orientation(self, x4, y4, z4, theta=0):
        """Compute the coordinates and orientation of the joint 4, from the end effector frame.
        returns the new x, y, z coordinates to use for inverse kinamatics function"""

        a4 = self.dh_params[3][1] # link 3 length
        theta1 = math.atan2(y4, x4)

        z_comp = a4 * math.sin(theta)
        xy_comp =  a4 * math.cos(theta)

        x_comp = xy_comp * math.cos(theta1)
        y_comp = xy_comp * math.sin(theta1)

        x3 = x4 - x_comp
        y3 = y4 - y_comp
        z3 = z4 - z_comp

        return self.inverse(x3, y3, z3, theta)
    
    def compute_quintic_coeffs(self, theta0, thetaf, vel0, velf, acc0, accf, T):
        """
        Computes quintic polynomial coefficients for trajectory:
        theta(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        # Boundary conditions vector
        b = np.array([theta0, vel0, acc0, thetaf, velf, accf])

        # Time matrix
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])

        # Solve for coefficients
        coeffs = np.linalg.solve(A, b)
        return coeffs
    
    def evaluate_quintic(self, coeffs, t):
        """
        Evaluates position, velocity, acceleration at time t
        """
        a0, a1, a2, a3, a4, a5 = coeffs
        theta = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        theta_dot = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        theta_ddot = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
        return theta, theta_dot, theta_ddot
    
    def quintic_interpolation(self, t):
        """
        Quintic interpolation function.
        Smoothly interpolates from 0 to 1 with zero velocity and acceleration at endpoints.
        """
        return 6*t**5 - 15*t**4 + 10*t**3
    
    def smoothstep(self, t):
        """Smoothstep interpolation function."""
        return 3 * t**2 - 2 * t**3

    def smootherstep(self, t):
        """Smootherstep interpolation function (higher degree polynomial)."""
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def smootheststep(self, t):
        """Smootheststep interpolation function (even higher degree polynomial)."""
        return 10 * t**6 - 15 * t**5 + 6 * t**4

    def interpolate_angles(self, theta_start, theta_end, num_steps, interpolation_func):
        """Interpolate between two sets of joint angles using the selected interpolation function.
        
        Parameters
        theta_start: List of starting joint angles [theta0, theta1, theta2, theta3].
        theta_end: List of ending joint angles [theta0, theta1, theta2, theta3].
        num_steps: The number of interpolation steps.
        interpolation_func: Function for interpolation (smoothstep, smootherstep, or smootheststep).
        
        Returns: List of interpolated joint angles at each step.
        """
        interpolated_angles = []
        
        for step in range(num_steps + 1):
            t = step / num_steps  # Normalize to [0, 1] range
            
            # Apply the selected interpolation function for each joint angle
            interpolated_theta = [
                theta_start[i] + (theta_end[i] - theta_start[i]) * interpolation_func(t)
                for i in range(4)  # Assume 4 joint angles: theta0, theta1, theta2, theta3
            ]
            
            interpolated_angles.append(interpolated_theta)
        
        return interpolated_angles

    # ----------------------
    # Debug functions
    # ----------------------

    def plot_robot(self, joint_positions):
        """
        Plot a robot in 3D given joint positions.
        
        joint_positions: list of [x,y,z] positions from base to end-effector
        """
        xs, ys, zs = zip(*joint_positions)

        # Update link data
        self.link_plot.set_data(xs, ys)
        self.link_plot.set_3d_properties(zs)

        # Update end-effector
        self.ee_plot._offsets3d = ( [xs[-1]], [ys[-1]], [zs[-1]] )

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def live_test():
    controller = RobotController(debug=True)
    try:
        target_position = [0.1, 0.1, 0.1]  # Example target position in meters
        joint_angles = controller.inverse(*target_position)
        print("Calculated joint angles (deg):", joint_angles)
        for i, angle in enumerate(joint_angles):
            if controller.check_limits(angle, controller.motor_ids[i]):
                controller.move_motor(controller.motor_ids[i], angle)
        time.sleep(2)  # Wait for movement to complete
        current_position = controller.read_ee_position()
        print("Current end-effector position:", current_position)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping the program.")
    finally:
        controller.disable_motors()


def sim_test():
    controller = RobotController(debug=True)

    T = 2.0  # Duration of the trajectory in seconds
    t = 0.0
    dt = 0.1  # Time step in seconds
    while t <= T:
        coeffs = controller.compute_quintic_coeffs(0, 90, 0, 0, 0, 0, T)
        theta, theta_dot, theta_ddot = controller.evaluate_quintic(coeffs, t)
        print(f"t={t:.2f}s: θ={theta:.2f}°, ω={theta_dot:.2f}°/s, α={theta_ddot:.2f}°/s²")
        t += dt
        joint_angles = [theta, 0, 0, 0]  # Move only first joint for demo
        positions = controller.forward(joint_angles)
        controller.plot_robot(positions)
        time.sleep(dt)

    plt.show(block=True)

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt installed
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time

def animate_arm(angles, link_length=93, end_effector_length=50, pause_time=1):
    # Create figure once
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-3 * link_length, 3 * link_length)
    ax.set_ylim(-3 * link_length, 3 * link_length)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title("3-Link Arm Animation")

    # Initialize line objects
    line1, = ax.plot([], [], 'r-o', linewidth=3, label='Link 1')
    line2, = ax.plot([], [], 'g-o', linewidth=3, label='Link 2')
    line3, = ax.plot([], [], 'b-o', linewidth=3, label='Link 3')
    # line_end_effector, = ax.plot([], [], 'k-o', linewidth=3, label='End Effector')

    ax.legend()

    for t3, t4, t5 in [(a[1], a[2], a[3]) for a in angles]:
        # Compute cumulative angles in radians, starting from the second angle (t3)
        theta2 = np.deg2rad(t3)  # First joint angle
        theta3 = np.deg2rad(t4)  # Second joint angle
        theta4 = np.deg2rad(t5) # Fourth angle is t5, added to t3 + t4

        # Compute joint positions
        x0, y0 = 0, 0  # Base (origin)
        x1 = x0 + link_length * np.cos(theta2)  # End of Link 1
        y1 = y0 + link_length * np.sin(theta2)
        x2 = x1 + link_length * np.cos(theta2 + theta3)  # End of Link 2
        y2 = y1 + link_length * np.sin(theta2 + theta3)
        x3 = x2 + end_effector_length * np.cos(theta2 + theta3 + theta4)  # End of Link 3 (end effector)
        y3 = y2 + end_effector_length * np.sin(theta2 + theta3 + theta4)

        # Update plot data for each link
        line1.set_data([x0, x1], [y0, y1])  # Link 1
        line2.set_data([x1, x2], [y1, y2])  # Link 2
        line3.set_data([x2, x3], [y2, y3])

        # print(x2,x3,y2,y3)
        # print(x3,y3)
        
        # Update end effector (make it horizontal and fixed length)
        # line_end_effector.set_data([x3, end_effector_x], [y3, end_effector_y])

        plt.pause(pause_time)  # pause to show the update

    # After the animation loop is done, call plt.show() to display the plot window
    plt.show()


def Problem2(circle_center, angle, radius):
    x = circle_center[0]
    y = circle_center[1]+radius*math.cos(angle)
    z = circle_center[2]+radius*math.sin(angle)
    return x,y,z

def Problem3():
    Controller = RobotController(debug=True)
    number = 37
    circle_center = [150,0,120]
    radius = 32

    positions = []

    increment = 360/(number-1)
    for i in range(0,number):
        x,y,z = Problem2(circle_center, math.radians(increment*(i)), radius)
        print("input",x,y,z)
        p1,p2,p3,p4 = Controller.inverse_orientation(x,y,z,math.radians(0))
        print(p1,p2,p3,p4)
        # Controller.Move_motor([p1,p2,p3,p4])
        time.sleep(0.1)
        positions.append([p1,p2,p3,p4])

    # print(positions)
    animate_arm(positions,pause_time=1/2)

def test_interpolations(x1,y1,z1,x2,y2,z2,theta=-90):
    Controller = RobotController(debug=False, DEVICENAME="COM3")
    p1,p2,p3,p4 = Controller.inverse_orientation(x1,y1,z1,math.radians(theta))
    p5,p6,p7,p8 = Controller.inverse_orientation(x2,y2,z2,math.radians(theta))

    angles = Controller.interpolate_angles([p1,p2,p3,p4], [p5,p6,p7,p8], 50, Controller.smoothstep)
    for angle_set in angles:
        Controller.Move_motor(angle_set)
        time.sleep(0.05)
    

def test(x,y,z,theta=-90):
    Controller = RobotController(debug=False, DEVICENAME="COM3")
    p1,p2,p3,p4 = Controller.inverse_orientation(x,y,z,math.radians(theta))
    Controller.Move_motor([p1,p2,p3,p4])

def calibrate():
    Controller = RobotController(debug=False, DEVICENAME="COM3")
    Controller.enable_motors()
    Controller.Move_motor([0,0,0,0])


def main():
    Problem3()
    # endeffector_rotation()
    # calibrate()
    # test_inter(100,0,0, 150,0,0)

if __name__ == "__main__":
    test(80,0,130)

    time.sleep(2)

    # Load the H template
    template = cv2.imread("H.png", cv2.IMREAD_GRAYSCALE)
    template = cv2.GaussianBlur(template, (3,3), 0)
    template_edges = cv2.Canny(template, 50, 150)
    tH, tW = template_edges.shape

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)



    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit()

    for _ in range(1):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Template matching
        res = cv2.matchTemplate(edges, template_edges, cv2.TM_CCOEFF_NORMED)
        # Get only max element of res dont use thresholding
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        pt = max_loc

        h = 30
        vx = (150*(175-h)/175)/600

        x = -(245 - pt[1])*vx + 25
        y = (300 - pt[0])*vx
        print("x:", x, "y:", y)
        test(80+x, y, 100)
        test_interpolations(80+x,y,100, 80+x,y,-20)

        cv2.rectangle(frame, (pt[0], pt[1]), (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
        cv2.putText(frame, "H", (pt[0], pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Show video feed
        cv2.imshow("H Detection (Press Q to quit)", frame)
        cv2.imwrite("current_frame.png", frame)

        # Quit on Q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
