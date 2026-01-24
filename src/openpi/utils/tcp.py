import numpy as np
import numpy.typing as npt


def rotvec2rotm(rotvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert rotation vector to rotation matrix
    """
    theta = np.linalg.norm(rotvec)
    if np.isclose(theta, 0):
        return np.eye(3)
    else:
        k = rotvec / theta
        K = np.array(
            [
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        return R

def rotm2rpy(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrix to roll-pitch-yaw angles
    """
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])


def rpy2rotm(rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert roll-pitch-yaw angles to rotation matrix
    """
    roll, pitch, yaw = rpy
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R



def rotm2rotvec(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrix to rotation vector
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(theta, 0):
        return np.zeros(3)
    else:
        k = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]
        )
        k = k / (2 * np.sin(theta))
        return theta * k



def ee2tcp(ee_pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ee_cartesian = ee_pose[:3]
    ee_rpy = ee_pose[3:]
    ee_rot_mat = rpy2rotm(ee_rpy)
    ee2tcp_rot_mat = np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ]
    )
    tcp_rot_mat = ee_rot_mat @ ee2tcp_rot_mat
    tcp_rotvec = rotm2rotvec(tcp_rot_mat)
    # opposite the rotation vector but keep the same pose
    angle_rad = np.linalg.norm(tcp_rotvec)
    vec = tcp_rotvec / angle_rad
    alternate_angle_rad = 2 * np.pi - angle_rad
    tcp_rotvec = -vec * alternate_angle_rad

    return np.concatenate([ee_cartesian, tcp_rotvec])



def tcp2ee(tcp_pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    tcp_cartesian = tcp_pose[:3]
    tcp_rotvec = tcp_pose[3:]
    tcp_rot_mat = rotvec2rotm(tcp_rotvec)
    tcp2ee_rot_mat = np.array(
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ]
    )
    ee_rot_mat = tcp_rot_mat @ tcp2ee_rot_mat
    ee_rpy = rotm2rpy(ee_rot_mat)
    return np.concatenate([tcp_cartesian, ee_rpy])


def tcp2ee_cartesian(tcp_pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return tcp_pose[:3]


def tcp2ee_rpy(tcp_pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    tcp_rotvec = tcp_pose[3:]
    tcp_rot_mat = rotvec2rotm(tcp_rotvec)
    tcp2ee_rot_mat = np.array(
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ]
    )
    ee_rot_mat = tcp_rot_mat @ tcp2ee_rot_mat
    ee_rpy = rotm2rpy(ee_rot_mat)
    return ee_rpy