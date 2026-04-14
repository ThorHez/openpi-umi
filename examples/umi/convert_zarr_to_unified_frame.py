"""
Post-process existing HITL zarr files: transform robot1's eef data
from robot1's local frame into robot0's frame using tx_robot1_robot0.

Supports both .zarr directories and .zarr.zip archives.

Usage:
    python scripts/convert_zarr_to_unified_frame.py -i <input.zarr.zip> [-o <output.zarr.zip>] [--overwrite]

If -o is omitted the script writes to <input_stem>_unified.zarr.zip next to the input.
Pass --overwrite to modify the input file in-place (a backup is created first).
"""

import sys
import os
import shutil
import tempfile
import argparse
import time

import numpy as np
import zarr
import numcodecs

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

import numpy as np
import scipy.spatial.transform as st


def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=pos.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot.as_matrix()
    mat[..., 3, 3] = 1
    return mat


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


def pose_to_pos_rot(pose):
    pos = pose[..., :3]
    rot = st.Rotation.from_rotvec(pose[..., 3:])
    return pos, rot


def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))


def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))


def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose


TX_ROBOT1_ROBOT0 = np.array(
    [
        [0.99996206, 0.00661996, 0.00566226, -0.01676012],
        [-0.00663261, 0.99997554, 0.0022186, -0.60552492],
        [-0.00564743, -0.00225607, 0.99998151, -0.007277],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

ROBOT1_KEYS_POS = "robot1_eef_pos"
ROBOT1_KEYS_ROT = "robot1_eef_rot_axis_angle"
ROBOT1_KEYS_DEMO = "robot1_demo_start_pose"


def transform_array(pos_arr: np.ndarray, rot_arr: np.ndarray,
                     tx_robot0_robot1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batch-transform (N,3) pos + (N,3) rot arrays into robot0's frame."""
    n = pos_arr.shape[0]
    new_pos = np.empty_like(pos_arr)
    new_rot = np.empty_like(rot_arr)
    for i in range(n):
        pose_6d = np.concatenate([pos_arr[i], rot_arr[i]])
        tf = transform_pose(tx_robot0_robot1, pose_6d)
        new_pos[i] = tf[:3]
        new_rot[i] = tf[3:6]
    return new_pos, new_rot


def transform_demo_start(demo_arr: np.ndarray,
                          tx_robot0_robot1: np.ndarray) -> np.ndarray:
    """Transform (N,6) demo_start_pose array (pos+rot concatenated)."""
    n = demo_arr.shape[0]
    out = np.empty_like(demo_arr)
    for i in range(n):
        out[i] = transform_pose(tx_robot0_robot1, demo_arr[i])
    return out


def open_zarr(path: str, mode: str = "r"):
    if path.endswith(".zarr.zip"):
        # zarr v2 exposed ZipStore at top-level; zarr v3 exposes it under zarr.storage
        ZipStore = None
        if hasattr(zarr, "storage") and hasattr(zarr.storage, "ZipStore"):
            ZipStore = zarr.storage.ZipStore
        elif hasattr(zarr, "ZipStore"):
            ZipStore = zarr.ZipStore
        else:
            raise RuntimeError(
                "Could not find ZipStore in zarr. "
                "Please install a zarr version that provides ZipStore."
            )

        store = ZipStore(path, mode=mode)
        return zarr.open_group(store, mode=mode), store
    else:
        return zarr.open_group(path, mode=mode), None


def _create_array_compat(group, name: str, *, data, chunks, compressor, overwrite: bool = True):
    """Create an array on a zarr Group, compatible with zarr v2/v3."""
    if hasattr(group, "create_array"):
        # zarr v3
        if compressor is None:
            compressors = None
        else:
            # Prefer zarr v3 native codecs; fall back to numcodecs for v2.
            if hasattr(zarr, "codecs") and hasattr(zarr.codecs, "BloscCodec"):
                compressors = zarr.codecs.BloscCodec(
                    cname="lz4",
                    clevel=5,
                    shuffle=zarr.codecs.BloscShuffle.noshuffle,
                )
            else:
                compressors = compressor
        try:
            # zarr >= 3.1 supports data= for create_array
            return group.create_array(
                name,
                data=data,
                chunks=chunks,
                compressors=compressors,
                overwrite=overwrite,
            )
        except TypeError as e:
            # zarr 3.0.x: create_array requires shape/dtype and does not accept data=
            msg = str(e)
            if "unexpected keyword argument 'data'" not in msg:
                raise
            arr = group.create_array(
                name,
                shape=data.shape,
                dtype=data.dtype,
                chunks=chunks,
                compressors=compressors,
                overwrite=overwrite,
            )
            arr[...] = data
            return arr
    # zarr v2
    return group.array(
        name,
        data=data,
        chunks=chunks,
        compressor=compressor,
        overwrite=overwrite,
    )


def _zip_store_class():
    if hasattr(zarr, "storage") and hasattr(zarr.storage, "ZipStore"):
        return zarr.storage.ZipStore
    if hasattr(zarr, "ZipStore"):
        return zarr.ZipStore
    raise RuntimeError("Could not find ZipStore in zarr.")


def _local_store(path: str):
    """Return a local-directory store object for zarr v2/v3."""
    if hasattr(zarr, "storage") and hasattr(zarr.storage, "LocalStore"):
        return zarr.storage.LocalStore(path)
    if hasattr(zarr, "DirectoryStore"):
        return zarr.DirectoryStore(path)
    raise RuntimeError("Could not find a local directory store class in zarr.")


def process(input_path: str, output_path: str, tx_robot1_robot0: np.ndarray):
    tx_robot0_robot1 = np.linalg.inv(tx_robot1_robot0)

    src_root, src_store = open_zarr(input_path, mode="r")
    data = src_root["data"]
    meta = src_root["meta"]

    all_keys = list(data.keys())
    episode_ends = meta["episode_ends"][:]
    n_episodes = len(episode_ends)
    n_total = int(episode_ends[-1]) if n_episodes > 0 else 0

    has_pos = ROBOT1_KEYS_POS in all_keys
    has_rot = ROBOT1_KEYS_ROT in all_keys
    has_demo = ROBOT1_KEYS_DEMO in all_keys

    if not has_pos and not has_rot:
        print(f"[SKIP] No robot1 eef keys found in {input_path}")
        if src_store:
            src_store.close()
        return

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"  Episodes : {n_episodes}")
    print(f"  Total steps: {n_total}")
    print(f"  Keys to transform: "
          f"{'pos ' if has_pos else ''}"
          f"{'rot ' if has_rot else ''}"
          f"{'demo_start' if has_demo else ''}")

    pos_data = data[ROBOT1_KEYS_POS][:] if has_pos else None
    rot_data = data[ROBOT1_KEYS_ROT][:] if has_rot else None
    demo_data = data[ROBOT1_KEYS_DEMO][:] if has_demo else None

    t0 = time.monotonic()

    if has_pos and has_rot:
        print("  Transforming robot1 eef_pos + eef_rot_axis_angle ...")
        new_pos, new_rot = transform_array(pos_data, rot_data, tx_robot0_robot1)
    elif has_pos:
        print("  [WARN] Only pos found without rot — skipping pos-only transform")
        new_pos, new_rot = pos_data, rot_data
    else:
        new_pos, new_rot = pos_data, rot_data

    if has_demo:
        print("  Transforming robot1 demo_start_pose ...")
        new_demo = transform_demo_start(demo_data, tx_robot0_robot1)
    else:
        new_demo = None

    t_transform = time.monotonic() - t0
    print(f"  Transform done in {t_transform:.1f}s")

    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE)
    CHUNK_SIZE = 2000

    def _write_to_root(dst_root):
        dst_data = dst_root.require_group("data")
        dst_meta = dst_root.require_group("meta")

        print(f"  Writing {len(all_keys)} features ...")
        for idx, key in enumerate(all_keys, 1):
            src_arr = data[key]
            arr_data = src_arr[:]

            if key == ROBOT1_KEYS_POS and new_pos is not None:
                arr_data = new_pos
            elif key == ROBOT1_KEYS_ROT and new_rot is not None:
                arr_data = new_rot
            elif key == ROBOT1_KEYS_DEMO and new_demo is not None:
                arr_data = new_demo

            cks = (CHUNK_SIZE,) + arr_data.shape[1:]
            _create_array_compat(dst_data, key, data=arr_data, chunks=cks, compressor=compressor, overwrite=True)
            print(f"\r    [{idx}/{len(all_keys)}] {key:<40}", end="", flush=True)

        _create_array_compat(
            dst_meta,
            "episode_ends",
            data=episode_ends,
            chunks=episode_ends.shape,
            compressor=None,
            overwrite=True,
        )
        for attr_key in src_root.attrs:
            dst_root.attrs[attr_key] = src_root.attrs[attr_key]
        dst_root.attrs["unified_frame"] = "robot0"
        dst_root.attrs["tx_robot1_robot0"] = tx_robot1_robot0.tolist()
        print()

    is_zip = output_path.endswith(".zarr.zip")
    if is_zip:
        # zarr v3's copy_store path is incomplete in some versions; write directly to ZipStore instead.
        if os.path.exists(output_path):
            os.remove(output_path)
        ZipStore = _zip_store_class()
        with ZipStore(output_path, mode="w") as dst_store:
            dst_root = zarr.open_group(dst_store, mode="w")
            _write_to_root(dst_root)
    else:
        dst_root = zarr.open_group(output_path, mode="w")
        _write_to_root(dst_root)

    if src_store:
        src_store.close()

    size_mb = os.path.getsize(output_path) / 1024 ** 2 if is_zip else 0
    elapsed = time.monotonic() - t0
    print(f"  Done. {n_episodes} episodes, {n_total} steps, "
          f"{size_mb:.1f} MB, {elapsed:.1f}s total")

    # -- Verification: re-read output and compare against expected values --
    print("\n  === Verification ===")
    ver_root, ver_store = open_zarr(output_path, mode="r")
    ver_data = ver_root["data"]
    ver_meta = ver_root["meta"]

    # Build expected arrays keyed by name
    expected = {}
    for key in all_keys:
        if key == ROBOT1_KEYS_POS and new_pos is not None:
            expected[key] = new_pos
        elif key == ROBOT1_KEYS_ROT and new_rot is not None:
            expected[key] = new_rot
        elif key == ROBOT1_KEYS_DEMO and new_demo is not None:
            expected[key] = new_demo
        else:
            expected[key] = data[key][:]

    all_pass = True
    for key in all_keys:
        actual = ver_data[key][:]
        exp = expected[key]

        ok_shape = actual.shape == exp.shape
        ok_dtype = actual.dtype == exp.dtype

        if np.issubdtype(exp.dtype, np.floating):
            max_err = float(np.max(np.abs(actual - exp))) if ok_shape else float("inf")
            mean_err = float(np.mean(np.abs(actual - exp))) if ok_shape else float("inf")
            tol = 1e-5
            ok_val = ok_shape and max_err < tol
            err_str = f"max_err={max_err:.2e}, mean_err={mean_err:.2e}"
        else:
            ok_val = ok_shape and np.array_equal(actual, exp)
            err_str = "exact" if ok_val else "MISMATCH"

        status = "PASS" if (ok_shape and ok_dtype and ok_val) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"    [{status}] {key:<40} shape={ok_shape} dtype={ok_dtype} {err_str}")

    # episode_ends
    ver_ends = ver_meta["episode_ends"][:]
    ends_ok = np.array_equal(ver_ends, episode_ends)
    if not ends_ok:
        all_pass = False
    print(f"    [{'PASS' if ends_ok else 'FAIL'}] meta/episode_ends"
          f"  shape={ver_ends.shape == episode_ends.shape}")

    # attrs
    attr_unified = ver_root.attrs.get("unified_frame", None)
    attr_tx = ver_root.attrs.get("tx_robot1_robot0", None)
    attr_ok = attr_unified == "robot0" and attr_tx is not None
    if attr_tx is not None:
        attr_ok = attr_ok and np.allclose(np.array(attr_tx), tx_robot1_robot0, atol=1e-10)
    if not attr_ok:
        all_pass = False
    print(f"    [{'PASS' if attr_ok else 'FAIL'}] attrs (unified_frame, tx_robot1_robot0)")

    if ver_store:
        ver_store.close()

    if all_pass:
        print("  === All checks PASSED ===")
    else:
        print("  === Some checks FAILED ===")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Transform robot1 eef data in HITL zarr to robot0 frame.")
    parser.add_argument("-i", "--input", required=True,
                        help="Input .zarr or .zarr.zip path")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: <input>_unified.zarr.zip)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite input in-place (backup created)")
    parser.add_argument("--tx", type=str, default=None,
                        help="Custom tx_robot1_robot0 as comma-separated 16 floats (row-major 4x4)")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    if args.tx is not None:
        vals = [float(v) for v in args.tx.split(",")]
        assert len(vals) == 16, "--tx needs 16 comma-separated floats (4x4 row-major)"
        tx = np.array(vals, dtype=np.float64).reshape(4, 4)
    else:
        tx = TX_ROBOT1_ROBOT0

    if args.overwrite:
        backup = input_path + ".bak"
        print(f"Backing up {input_path} -> {backup}")
        if os.path.isdir(input_path):
            shutil.copytree(input_path, backup)
        else:
            shutil.copy2(input_path, backup)
        output_path = input_path
    elif args.output is not None:
        output_path = os.path.abspath(args.output)
    else:
        base = input_path
        if base.endswith(".zarr.zip"):
            base = base[:-len(".zarr.zip")]
        elif base.endswith(".zarr"):
            base = base[:-len(".zarr")]
        output_path = base + "_unified.zarr.zip"

    process(input_path, output_path, tx)
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
