"""
MyRobot - Python port of the provided MATLAB/Python hybrid.
Requires: dynamixel_sdk, numpy
 
Notes:
- The original rigidBodyTree / inverse kinematics and draw routines were MATLAB-specific.
  I left placeholders for those (create_rbt, draw_robot, inverse) — see comments in code.
- Adjust DEVICENAME, BAUDRATE, motor IDs, offsets and DH parameters to match your hardware.
"""
 
import time
import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler
 
class MyRobot:
    def __init__(self,
                 devicename='COM3',
                 baudrate=1000000,
                 motor_ids=None,
                 gripper_motor_id=4):
        # Communication / control constants
        self.ADDR_MX_TORQUE_ENABLE = 24
        self.ADDR_MX_GOAL_POSITION = 30
        self.ADDR_MX_PRESENT_POSITION = 36
        self.PROTOCOL_VERSION = 1.0
 
        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0
        self.DXL_MOVING_STATUS_THRESHOLD = 10
        self.COMM_SUCCESS = 0
        self.COMM_TX_FAIL = -1001
 
        # Port + packet handlers (initialized below)
        self.DEVICENAME = devicename
        self.BAUDRATE = baudrate
        self.portHandler = None
        self.packetHandler = None
 
        # Robot configuration
        self.motor_ids = motor_ids if motor_ids is not None else [0, 1, 2, 3]
        self.gripper_motor_id = gripper_motor_id
 
        # Denavit-Hartenberg params (same arrangement as original)
        # Each row: [a, alpha, d, theta] (angles in radians where needed)
        self.dh = np.array([
            [0.0, -np.pi/2, 0.0955, 0.0],
            [0.116, 0.0, 0.0, 0.0],
            [0.096, 0.0, 0.0, 0.0],
            [0.064, 0.0, 0.0, 0.0]
        ])
 
        # Internal state
        self.forward_transform = np.zeros((4,4))
        self.joint_angles = [0.0, 0.0, 0.0, 0.0]   # desired joint angles [deg]
        self.joint_pos = np.zeros((4,4))
        self.draw_robot_flag = False
        self.use_smooth_speed_flag = False
        self.gripper_open_flag = True
        self.rbt = None   # placeholder for rigid body tree object
        self.joint_limits = [-130, 130, -180, 0, -100, 100, -100, 100]
        self.ik = None
        self.ik_weights = [0.25, 0.25, 0.25, 1, 1, 1]
        # Offsets (degrees) - copied from your original
        self.joint_offsets = [171-5, 150+90, 150, 150]
        self.joint_angle_error = [0.0, 0.0, 0.0, 0.0]
        self.init_status = 0
        self.movement_history = []   # list of dicts for each recorded configuration
        self.motor_speed = [0.1, 0.1, 0.1, 0.1]   # normalized [0..1]
        self.motor_torque = [1.0, 1.0, 1.0, 1.0]  # normalized [0..1]
        self.pitch = 0.0
 
        # initialize communication and motors
        try:
            # initialize handlers
            self.portHandler = PortHandler(self.DEVICENAME)
            self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
 
            if not self.portHandler.openPort():
                raise RuntimeError(f"Failed to open port {self.DEVICENAME}")
 
            if not self.portHandler.setBaudRate(self.BAUDRATE):
                raise RuntimeError(f"Failed to set baudrate {self.BAUDRATE}")
 
            # set default speeds/torques and move to zero
            # NOTE: set_speed expects values in (0,1]; overwrite=True to persist
            self.set_speed([0.1, 0.1, 0.1, 0.1], overwrite_speeds=True)
            self.set_torque_limit([1,1,1,1])
            self.move_j(0,0,0,0)
            self.init_status = 1
            print("MyRobot initialized (init_status=1).")
        except Exception as e:
            print("Initialization error:", e)
            self.init_status = 0
 
    # ----------------------
    # Gripper helpers
    # ----------------------
    def open_gripper(self):
        """Open gripper (writes 0 goal position to gripper motor)."""
        if not self.gripper_open_flag:
            res, err = self.packetHandler.write2ByteTxRx(self.portHandler,
                                                         self.gripper_motor_id,
                                                         self.ADDR_MX_GOAL_POSITION,
                                                         0)
            # res contains comm result; err contains packet error in this SDK style
            if res != 0:
                print(self.packetHandler.getTxRxResult(res))
            if err != 0:
                print(self.packetHandler.getRxPacketError(err))
            self.gripper_open_flag = True
 
    def close_gripper(self):
        """Close gripper (writes 1023 goal position)."""
        if self.gripper_open_flag:
            res, err = self.packetHandler.write2ByteTxRx(self.portHandler,
                                                         self.gripper_motor_id,
                                                         self.ADDR_MX_GOAL_POSITION,
                                                         1023)
            if res != 0:
                print(self.packetHandler.getTxRxResult(res))
            if err != 0:
                print(self.packetHandler.getRxPacketError(err))
            self.gripper_open_flag = False
 
    def actuate_gripper(self):
        """Toggle gripper."""
        if self.gripper_open_flag:
            self.close_gripper()
        else:
            self.open_gripper()
 
    # ----------------------
    # Motion helpers
    # ----------------------
    def smooth_speed(self, joint_angle_deltas):
        """
        Dynamically set speeds such that larger moves are faster, smaller moves are slower,
        while trying to finish motions at the same time.
        joint_angle_deltas: iterable of angle differences [deg]
        """
        max_angle = max(abs(np.array(joint_angle_deltas)))
        # protect division by zero
        if max_angle == 0:
            return
        speed_per_deg = max_angle / 100.0
        if speed_per_deg == 0:
            return
        new_speeds = np.abs(np.array(joint_angle_deltas) / speed_per_deg) * 0.01
        # multiply by current motor_speed entries (if zero, keep previous)
        for i in range(len(self.motor_speed)):
            if new_speeds[i] == 0:
                new_speeds[i] = self.motor_speed[i]
            else:
                new_speeds[i] = new_speeds[i] * self.motor_speed[i]
        # clamp to (0,1]
        new_speeds = np.clip(new_speeds, 1e-5, 1.0)
        self.set_speed(new_speeds.tolist(), overwrite_speeds=False)
 
    def create_rbt(self):
        """
        Placeholder for creating a rigid body tree representation (MATLAB had rigidBodyTree).
        In Python you'd use e.g. roboticstoolbox or custom code.
        """
        raise NotImplementedError("create_rbt: Rigid-body tree creation not implemented in this Python port.")
 
    def set_speed(self, speeds, overwrite_speeds=False):
        """
        Set individual motor speeds between >0 and 1 (normalized).
        speeds: iterable with same length as motor_ids
        overwrite_speeds: bool, whether to overwrite internal self.motor_speed
        """
        speeds = list(speeds)
        if overwrite_speeds:
            self.motor_speed = speeds.copy()
 
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
        self.motor_torque = torques.copy()
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
 
    def get_position(self, motor_id):
        """Read current motor position (deg)."""
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler,
                                                                                            motor_id,
                                                                                            self.ADDR_MX_PRESENT_POSITION)
        if dxl_comm_result != 0:
            print(self.packetHandler.getTxRxResult(dxl_comm_result))
            return None
        if dxl_error != 0:
            print(self.packetHandler.getRxPacketError(dxl_error))
            return None
        return self.rot_to_deg(dxl_present_position)
 
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
 
    def check_limits(self, deg, motor_id):
        """Check that given deg (scalar) is inside declared limits for motor_id."""
        # mapping by index of motor_ids
        try:
            idx = self.motor_ids.index(motor_id)
        except ValueError:
            raise AssertionError("Unknown motor id")
 
        if idx == 0:
            assert abs(deg) <= 130, "Angle Limits for first Axis Reached, Min/Max: +-130°"
        elif idx == 1:
            # second axis limits [0, -180] in original; keep same semantics
            assert deg <= 0 and deg >= -180, "Angle Limits for second Axis Reached, Min/Max: [0,-180]°"
        else:
            assert abs(deg) <= 100, "Angle Limits Reached, Min/Max: +-100°"
        return deg
 
    def deg_to_rot(self, deg):
        """Convert degrees to motor units (original used 1/0.29 factor)."""
        return int(round(deg / 0.29))
 
    def rot_to_deg(self, rot):
        """Convert motor units to degrees (original used *0.29)."""
        return rot * 0.29
 
    def move_j(self, j1, j2, j3, j4, wait=True, error_threshold_deg=2.0):
        """
        Move robot joints to specified joint angles (in degrees).
        This writes goal positions and optionally waits until actual vs desired error < threshold (deg).
        """
        # check limits (may raise AssertionError)
        j1 = self.check_limits(j1, self.motor_ids[0])
        j2 = self.check_limits(j2, self.motor_ids[1])
        j3 = self.check_limits(j3, self.motor_ids[2])
        j4 = self.check_limits(j4, self.motor_ids[3])
 
        desired = [j1, j2, j3, j4]
 
        if self.use_smooth_speed_flag:
            deltas = np.array(desired) - np.array(self.joint_angles)
            self.smooth_speed(deltas)
 
        # update internal desired state
        self.joint_angles = desired.copy()
        # compute forward kinematics if needed (forward expects degrees array in code below)
        try:
            self.forward(np.array(desired))
        except Exception:
            # forward may not be implemented fully — ignore if placeholder
            pass
 
        # add offsets before sending to motors
        j1_out = j1 + self.joint_offsets[0]
        j2_out = j2 + self.joint_offsets[1]
        j3_out = j3 + self.joint_offsets[2]
        j4_out = j4 + self.joint_offsets[3]
 
        # write goal positions (2-byte)
        for motor_id, angle in zip(self.motor_ids, [j1_out, j2_out, j3_out, j4_out]):
            goal = self.deg_to_rot(angle)
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler,
                                                                           motor_id,
                                                                           self.ADDR_MX_GOAL_POSITION,
                                                                           goal)
            if dxl_comm_result != 0:
                print(self.packetHandler.getTxRxResult(dxl_comm_result))
            if dxl_error != 0:
                print(self.packetHandler.getRxPacketError(dxl_error))
 
        # optionally wait until actual angle error below threshold
        if wait:
            while True:
                current = self.read_joint_angles()
                # joint_angle_error computed in read_joint_angles (list)
                max_err = max([abs(e) for e in self.joint_angle_error])
                if max_err < error_threshold_deg:
                    break
                time.sleep(0.02)
 
    def draw_robot(self):
        """
        Placeholder for robot drawing (requires robotics plotting tools).
        """
        if self.rbt is None:
            # try to create rbt (not implemented)
            raise NotImplementedError("draw_robot requires create_rbt implementation or robotics toolbox")
 
    def forward(self, j_a):
        """
        Compute forward transform and joint positions.
        Input j_a may be in degrees; original code expected deg.
        Returns ee_cartesian_coords (4x1 homogeneous).
        """
        # ensure array
        j_a = np.array(j_a, dtype=float)
 
        # Convert degrees to radians when using trig functions where appropriate
        # The original code used cosd/sind (degree trig), so convert degrees to radians per call.
        def cosd(x): return np.cos(np.deg2rad(x))
        def sind(x): return np.sin(np.deg2rad(x))
 
        # first transform
        # Using original matrix (adapted): uses j_a(1) in degrees and dh angles in radians
        a1, alpha1, d1, _ = self.dh[0]
        j1 = j_a[0]
        T = np.array([
            [cosd(j1), -sind(j1)*np.cos(alpha1), sind(j1)*np.sin(alpha1), a1 * cosd(j1)],
            [sind(j1), cosd(j1)*np.cos(alpha1), -cosd(j1)*np.sin(alpha1), a1 * sind(j1)],
            [0.0, np.sin(alpha1), np.cos(alpha1), d1],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.forward_transform = T.copy()
        self.joint_pos[:,0] = (T @ np.array([0,0,0,1])).flatten()
        for i in range(1, len(j_a)):
            ai, alphai, di, _ = self.dh[i]
            ji = j_a[i]
            Ti = np.array([
                [cosd(ji), -sind(ji)*np.cos(alphai), sind(ji)*np.sin(alphai), ai * cosd(ji)],
                [sind(ji), cosd(ji)*np.cos(alphai), -cosd(ji)*np.sin(alphai), ai * sind(ji)],
                [0.0, np.sin(alphai), np.cos(alphai), di],
                [0.0, 0.0, 0.0, 1.0]
            ])
            T = T @ Ti
            self.joint_pos[:,i] = (T @ np.array([0,0,0,1])).flatten()
 
        self.forward_transform = T.copy()
        ee_cartesian_coords = (T @ np.array([0,0,0,1])).flatten()
        return ee_cartesian_coords
 
    def inverse(self, x, y, z, pitch_rad):
        """
        Analytic inverse kinematics from your original code.
        Input: x,y,z in meters, pitch_rad in radians.
        Returns joint angles in degrees [j1,j2,j3,j4].
        """
        # same math as original (careful: original used rad/deg mixing)
        # j1 (rad)
        j1 = np.arctan2(y, x)
 
        # use original DH parameters (self.dh)
        try:
            a4 = self.dh[3,0]
            d1 = self.dh[0,2]
            a2 = self.dh[1,0]
            a3 = self.dh[2,0]
        except Exception:
            raise RuntimeError("DH parameters not set properly")
 
        rho = np.sqrt(x**2 + y**2)
        # inside acos:
        num = ( (rho - a4 * np.cos(pitch_rad))**2 + (d1 - z)**2 - a2**2 - a3**2 )
        den = 2 * a2 * a3
        val = num / den
        # guard numerical issues
        if val < -1.0 or val > 1.0:
            raise AssertionError("Configuration Impossible (value out of acos domain)")
        j3 = np.arccos(val)
        j2 = -np.arctan2(z - d1 - a4*np.sin(pitch_rad), rho - a4*np.cos(pitch_rad)) \
             - np.arctan2(a3 + a2*np.cos(j3), a2*np.sin(j3)) + (np.pi/2 - j3)
        j4 = pitch_rad - j2 - j3
 
        j_a = np.rad2deg(np.array([j1, j2, j3, j4]))
        # save pitch
        self.pitch = pitch_rad
        return j_a.tolist()
 
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
 
    def move_c(self, x, y, z, pitch_deg):
        """Move in Cartesian space by converting to joint angles and calling move_j."""
        pitch_rad = np.deg2rad(pitch_deg)
        j_a = self.inverse(x, y, z, pitch_rad)
        # j_a are in degrees
        self.move_j(j_a[0], j_a[1], j_a[2], j_a[3])
 
    def record_configuration(self):
        """
        Record current robot configuration (joint angles, speed, torque, gripper state)
        Stores dict into movement_history list.
        """
        j_a = self.joint_angles.copy()
        torque = self.motor_torque[0] if self.motor_torque else None
        speed = self.motor_speed[0] if self.motor_speed else None
        rec = {
            'joints': j_a,
            'speed': speed,
            'torque': torque,
            'gripper_open': self.gripper_open_flag
        }
        self.movement_history.append(rec)
        print(f"Recorded Speed: {speed}, Torque: {torque}, Joint Positions: {j_a}, Gripper open: {self.gripper_open_flag}")
 
    def delete_last_recorded_configuration(self):
        """Delete last recorded configuration if exists."""
        if not self.movement_history:
            print("No last history position")
            return
        self.movement_history.pop()
 
    def play_configuration_history(self):
        """Play back recorded movement history (sequentially)."""
        for rec in self.movement_history:
            speed = rec['speed']
            torque = rec['torque']
            if speed is not None:
                self.set_speed([speed, speed, speed, speed], overwrite_speeds=True)
            if torque is not None:
                self.set_torque_limit([torque, torque, torque, torque])
            j = rec['joints']
            self.move_j(j[0], j[1], j[2], j[3])
            time.sleep(1)
            if self.gripper_open_flag != rec['gripper_open']:
                self.actuate_gripper()
                time.sleep(1)
 
# Example usage:
r = MyRobot(devicename='COM3')   # on Linux
r.enable_motors()
r.move_j(0, -100, 0, 0)
while True:
    r.move_j(0, -100, 0, 0)
    time.sleep(1)
    r.move_j(-30, -60, 0, 0)
    time.sleep(1)
r.close_gripper()
r.disable_motors()