import time
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from dynamixel_sdk import PortHandler, PacketHandler

class RobotController:
    def __init__(self, debug=False,
                 DEVICENAME="/dev/ttyUSB0",
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

        self.dh_params = np.array([[0.05, 0, -0.5*pi, 0],
                                   [0, 0.093, 0, 0],
                                   [0, 0.093, 0, 0],
                                   [0, 0.05, 0, 0],
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

        self.Motor_offsets = [146, 63, 147, 235]
        self.enable_motors()

    # ----------------------
    # Helper functions
    # ----------------------

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
    
    def inverse(self, x, y, z, theta=0):
        """Compute inverse kinematics for given target_position [x,y,z]. Returns joint angles (deg)."""
        d1 = self.dh_params[0][0]
        a2 = self.dh_params[1][1]
        a3 = self.dh_params[2][1]
        theta1 = math.atan2(y, x)
        r = math.sqrt(x**2 + y**2)
        s = z - d1
        cos3 = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
        theta3 = math.atan2(cos3, math.sqrt(1 - cos3**2))
        sin3 = math.sin(theta3)
        theta2 = math.atan2(r, s) - math.atan2(a2 + a3 * cos3, a3 * sin3)
        theta4 = theta - theta2 - theta3
        joint_angles = [math.degrees(ang) for ang in [theta1, theta2, theta3, theta4]]
        return joint_angles
    
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

def main():
    sim_test()

if __name__ == "__main__":
    main()
