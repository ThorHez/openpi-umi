import numpy as np
import scipy.spatial.transform as st


def convert_pose_mat_rep(pose_mat, base_pose_mat=None, pose_rep="abs", backward=False):
    if not backward:
        # training transform
        if pose_rep == "abs":
            return pose_mat
        elif pose_rep == "rel":
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[..., :3, 3] - base_pose_mat[:3, 3]
            rot = pose_mat[..., :3, :3] @ np.linalg.inv(base_pose_mat[:3, :3])
            out = np.copy(pose_mat)
            out[..., :3, :3] = rot
            out[..., :3, 3] = pos
            return out
        elif pose_rep == "relative":
            out = np.linalg.inv(base_pose_mat) @ pose_mat
            return out
        elif pose_rep == "delta":
            all_pos = np.concatenate(
                [base_pose_mat[None, :3, 3], pose_mat[..., :3, 3]], axis=0
            )
            out_pos = np.diff(all_pos, axis=0)

            all_rot_mat = np.concatenate(
                [base_pose_mat[None, :3, :3], pose_mat[..., :3, :3]], axis=0
            )
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            out_rot = np.matmul(curr_rot, prev_rot)

            out = np.copy(pose_mat)
            out[..., :3, :3] = out_rot
            out[..., :3, 3] = out_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    else:
        # eval transform
        if pose_rep == "abs":
            return pose_mat
        elif pose_rep == "rel":
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[..., :3, 3] + base_pose_mat[:3, 3]
            rot = pose_mat[..., :3, :3] @ base_pose_mat[:3, :3]
            out = np.copy(pose_mat)
            out[..., :3, :3] = rot
            out[..., :3, 3] = pos
            return out
        elif pose_rep == "relative":
            out = base_pose_mat @ pose_mat
            return out
        elif pose_rep == "delta":
            output_pos = np.cumsum(pose_mat[..., :3, 3], axis=0) + base_pose_mat[:3, 3]

            output_rot_mat = np.zeros_like(pose_mat[..., :3, :3])
            curr_rot = base_pose_mat[:3, :3]
            for i in range(len(pose_mat)):
                curr_rot = pose_mat[i, :3, :3] @ curr_rot
                output_rot_mat[i] = curr_rot

            out = np.copy(pose_mat)
            out[..., :3, :3] = output_rot_mat
            out[..., :3, 3] = output_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")


def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=pos.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot.as_matrix()
    mat[..., 3, 3] = 1
    return mat


def pose_to_pos_rot(pose):
    pos = pose[..., :3]
    rot = st.Rotation.from_rotvec(pose[..., 3:])
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose9d(mat):
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out



def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out


def pose9d_to_mat(d10):
    pos = d10[..., :3]
    d6 = d10[..., 3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1] + (4, 4), dtype=d10.dtype)
    out[..., :3, :3] = rotmat
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1
    return out

def mat_to_pos_rot(mat):
    pos = (mat[..., :3, 3].T / mat[..., 3, 3].T).T
    rot = st.Rotation.from_matrix(mat[..., :3, :3])
    return pos, rot


def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape + (6,), dtype=pos.dtype)
    pose[..., :3] = pos
    pose[..., 3:] = rot.as_rotvec()
    return pose


def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))


def pose6d_to_9d(pose6d):
    pose_mat = pose_to_mat(pose6d)
    obs_pose_mat = convert_pose_mat_rep(
        pose_mat, pose_rep='abs', backward=False
    )
    obs_pose_9d = mat_to_pose9d(obs_pose_mat)
    return obs_pose_9d


def pose9d_to_6d(pose9d):
    pose_mat = pose9d_to_mat(pose9d)

    action_mat = convert_pose_mat_rep(
        pose_mat,
        pose_rep='abs',
        backward=True,
    )
    action_pose = mat_to_pose(action_mat)
    return action_pose


if __name__ == "__main__":
    pose6d = np.array([[0.14 ,  0.015,  0.068,  2.452, -2.719,  0.814]])
    pose9d = pose6d_to_9d(pose6d)
    print(pose9d)
    pose6d = pose9d_to_6d(pose9d)
    print(pose6d)
