import os
import time
import argparse
import numpy as np
import jittor as jt
from DeepM.projects.jittor_compat import torch
from scipy.spatial.transform import Rotation as R
import traceback
from DeepM.projects.jittor_compat import F as FF
from jittor_rasterizer import (
    Meshes,
    RasterizationSettings,
    MeshRasterizer,
    OrthographicCameras
)
import trimesh

# ----------------------------------------------------------------------
# 设置随机种子，确保结果可复现
# ----------------------------------------------------------------------
import random

seed_const  = 42

random.seed(seed_const)
np.random.seed(seed_const)
torch.manual_seed(seed_const)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_const)
    torch.cuda.manual_seed_all(seed_const)

# 启用确定性算法
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置环境变量，使用确定性算法
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
try:
    torch.use_deterministic_algorithms(True)
except RuntimeError:
    # 某些操作可能不支持确定性算法，忽略错误
    pass

try:
    from skimage.draw import circle_perimeter
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("---------------------------------------------------------------")
    print("[Warning] 'scikit-image' not found. Circle visualization will be skipped.")
    print("          To enable, run: pip install scikit-image")
    print("---------------------------------------------------------------")

def write_timing_numbers(txt_path, values):
    """
    覆盖写入 timing 数字，每行一个 float
    """
    with open(txt_path, "w") as f:
        for v in values:
            f.write(f"{float(v):.6f}\n")

def write_numbers(txt_path, values):
    """
    覆盖写入数字，每行一个 float/int
    """
    with open(txt_path, "w") as f:
        for v in values:
            f.write(f"{float(v):.6f}\n")

def save_obj(path, verts, faces):
    with open(path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def dilate_depth_map_1px(depth_map, max_depth_diff=None):
    """
    对 depth map 做“温和版”的 1 像素 min-dilation：
    - 只对原来无效 (inf) 的像素用 3x3 邻域的最小深度来补洞；
    - 原本有效的像素一律不改；
    - 如果指定 max_depth_diff，则只在局部 3x3 邻域足够“平滑”时才填充，
      即 (局部最大值 - 局部最小值) <= max_depth_diff。
    """
    device = depth_map.device
    dtype = depth_map.dtype
    INF = float('inf')

    # 内部用 float32 计算，更稳一点
    dm = depth_map.to(torch.float32)
    H, W = dm.shape

    # 有效像素
    valid = torch.isfinite(dm) & (dm < INF)

    # ---- 1. 3x3 邻域展开 ----
    padded = FF.pad(dm[None, None, ...], (1, 1, 1, 1), value=INF)   # [1,1,H+2,W+2]
    unfolded = FF.unfold(padded, kernel_size=3)                     # [1, 9, H*W]
    unfolded = unfolded.view(1, 9, H, W)                            # [1, 9, H, W]

    neigh_valid = torch.isfinite(unfolded)
    neigh_for_min = torch.where(neigh_valid, unfolded, INF)
    neigh_for_max = torch.where(neigh_valid, unfolded, -INF)

    # ★ 关键：squeeze 掉最前面的 batch 维度，变成 [H, W]
    local_min = neigh_for_min.min(dim=1).squeeze(0)  # [H, W]
    local_max = neigh_for_max.max(dim=1).squeeze(0)  # [H, W]

    # 如果整个 3x3 都是无效，local_min 会是 INF
    has_valid_neigh = torch.isfinite(local_min)         # [H, W]

    # ---- 2. 构造“可以填洞”的掩码 ----
    need_fill = (valid == 0)                            # [H, W]

    if max_depth_diff is not None:
        depth_span = local_max - local_min              # [H, W]
        smooth_enough = depth_span <= max_depth_diff    # [H, W]
        fill_mask = need_fill & has_valid_neigh & smooth_enough
    else:
        fill_mask = need_fill & has_valid_neigh

    # ---- 3. 只在 fill_mask 位置用 local_min 填洞 ----
    out = dm.clone()            # [H, W]
    out[fill_mask] = local_min[fill_mask]

    return out.to(dtype)



def create_sphere_mesh(center, radius, segments=16):
    verts = []
    if segments < 4:
        segments = 4

    verts.append(np.array(center) + np.array([0.0, 0.0, radius], dtype=np.float32))

    for i in range(1, segments):
        phi = np.pi * i / segments
        z = radius * np.cos(phi)
        r_ring = radius * np.sin(phi)
        for j in range(segments):
            theta = 2 * np.pi * j / segments
            x = r_ring * np.cos(theta)
            y = r_ring * np.sin(theta)
            verts.append(np.array(center) + np.array([x, y, z], dtype=np.float32))

    verts.append(np.array(center) + np.array([0.0, 0.0, -radius], dtype=np.float32))
    verts = np.array(verts, dtype=np.float32)

    faces = []
    v0 = 0
    for j in range(segments):
        v1 = j + 1
        v2 = ((j + 1) % segments) + 1
        faces.append([v0, v2, v1])

    for i in range(1, segments - 1):
        idx_prev_ring_start = 1 + (i - 1) * segments
        idx_curr_ring_start = 1 + i * segments
        for j in range(segments):
            v_curr_j = idx_curr_ring_start + j
            v_curr_next_j = idx_curr_ring_start + ((j + 1) % segments)
            v_prev_j = idx_prev_ring_start + j
            v_prev_next_j = idx_prev_ring_start + ((j + 1) % segments)

            faces.append([v_prev_j, v_prev_next_j, v_curr_j])
            faces.append([v_curr_j, v_prev_next_j, v_curr_next_j])

    bottom_pole_idx = len(verts) - 1
    last_ring_start = 1 + (segments - 2) * segments
    for j in range(segments):
        v1 = bottom_pole_idx
        v2 = last_ring_start + j
        v3 = last_ring_start + ((j + 1) % segments)
        faces.append([v1, v2, v3])

    return verts, np.array(faces, dtype=np.int32)


def compute_bounding_box_info(verts, margin=0.3):
    if isinstance(verts, torch.Tensor):
        mn, mx = verts.min(dim=0), verts.max(dim=0)
        return float(((mx - mn).max() * (1.0 + margin)).item())
    else:
        mn, mx = verts.min(axis=0), verts.max(axis=0)
    scale = (mx - mn).max() * (1.0 + margin)
    return scale


# ----------------------------------------------------------------------
# 可见性矩阵 + 贪心排序方向
# ----------------------------------------------------------------------
def load_visibility_matrix(txt_path):
    V = np.loadtxt(txt_path)
    if V.ndim == 1:
        V = V.reshape(1, -1)
    return (V != 0)


def greedy_sort_views(visibility_matrix):
    V = visibility_matrix
    m, n = V.shape
    covered = np.zeros(n, dtype=bool)
    unused = np.ones(m, dtype=bool)
    order = []

    gain_counts = V.sum(axis=1).astype(np.int32)
    for _ in range(m):
        gain_counts[~unused] = -1
        best_idx = int(np.argmax(gain_counts))
        best_gain = gain_counts[best_idx]
        if best_gain < 0:
            break
        order.append(best_idx)
        unused[best_idx] = False

        newly_covered = V[best_idx] & (~covered)
        if newly_covered.any():
            covered[newly_covered] = True
            gain_counts -= V[:, newly_covered].sum(axis=1)
        if best_gain == 0:
            break

    order.extend(np.where(unused)[0].tolist())
    return order
# def greedy_sort_views(visibility_matrix):
#     """
#     超快近似：按每个方向能覆盖的点数降序排序。
#     复杂度：O(m*n)（只做一次 row sum），不再有 m 轮迭代。
#     """
#     V = visibility_matrix
#     # V 是 bool：count_nonzero 比 sum 更稳
#     gain = np.count_nonzero(V, axis=1).astype(np.int32)  # (m,)
#     # 降序索引
#     order = np.argsort(-gain, kind="stable").astype(np.int32).tolist()
#     return order


# ----------------------------------------------------------------------
# Jittor 光栅化
# ----------------------------------------------------------------------
def rasterize_jittor(mesh, R, T, rasterizer, W=256, H=256, device=None, max_scale=None):
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if R.dim() == 2:
        R = R.unsqueeze(0)
    if T.dim() == 1:
        T = T.unsqueeze(0)

    scale_factor = 2.0 / max_scale
    focal_length = torch.tensor([[scale_factor, scale_factor]], dtype=torch.float32, device=device)
    cameras = OrthographicCameras(R=R, T=T, focal_length=focal_length, device=device)

    fragments = rasterizer(mesh.to(device), cameras=cameras)

    pix_to_face = fragments.pix_to_face[0, ..., 0]
    zbuf = fragments.zbuf[0, ..., 0]

    depth_map = zbuf.clone()
    valid = (depth_map >= 0) & torch.isfinite(depth_map)
    depth_map[valid == 0] = float('inf')

    tri_id_map = pix_to_face.clone()

    return {
        "depth_map": depth_map,   # GPU tensor
        "tri_id_map": tri_id_map  # GPU tensor
    }


# ----------------------------------------------------------------------
# 圆形窗口最小池化（offsets缓存）
# ----------------------------------------------------------------------
_CIRC_CACHE = {}  # key: (R, device_str)


def _get_circular_offsets(R, device):
    key = (int(R), str(device))
    if key in _CIRC_CACHE:
        return _CIRC_CACHE[key]

    R = int(max(R, 0))
    rel = torch.arange(-R, R + 1, device=device, dtype=torch.long)
    dv, du = torch.meshgrid(rel, rel, indexing='ij')

    dist2 = du * du + dv * dv
    # 原来：mask = dist2 <= R * R
    # 现在：不让刚好在圆周上的点参与池化
    mask = dist2 <= R * R

    offsets = torch.stack([dv[mask], du[mask]], dim=1)  # (K,2)
    _CIRC_CACHE[key] = offsets
    return offsets


def compute_vertex_normals(verts, faces, eps=1e-12):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)

    normals = torch.zeros_like(verts)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)

    normals = FF.normalize(normals, dim=1, eps=eps)
    return normals


def compute_min_pooled_depth_for_vertices(
        vtx_depth_map, u_final, v_final, W, H, R, device,
        batch_size=2048#2048
):
    num_verts = u_final.shape[0]
    if num_verts == 0:
        return torch.empty(0, dtype=vtx_depth_map.dtype, device=device)

    offsets = _get_circular_offsets(R, device)  # cached
    K = offsets.shape[0]
    out = torch.empty(num_verts, device=device, dtype=vtx_depth_map.dtype)

    with torch.no_grad():
        for start in range(0, num_verts, batch_size):
            end = min(start + batch_size, num_verts)

            u_c = u_final[start:end]
            v_c = v_final[start:end]

            centers = torch.stack([v_c, u_c], dim=1)[:, None, :]  # (b,1,2)
            abs_coords = centers + offsets[None, :, :]            # (b,K,2)

            v_abs = abs_coords[..., 0]
            u_abs = abs_coords[..., 1]

            inb = (v_abs >= 0) & (v_abs < H) & (u_abs >= 0) & (u_abs < W)

            v_abs = v_abs.clamp(0, H - 1)
            u_abs = u_abs.clamp(0, W - 1)

            depths = vtx_depth_map[v_abs, u_abs]                  # (b,K)
            depths = depths.masked_fill(inb == 0, float('inf'))

            out[start:end] = depths.amin(dim=1)

    return out


# ----------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 默认使用仓库自带的小型示例；从任意工作目录启动均可。
    project_root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run DeepMill++ accessibility analysis.")
    parser.add_argument(
        "--input-root", default=os.path.join(project_root, "examples", "done"),
        help="Directory containing one subdirectory per OBJ model."
    )
    parser.add_argument(
        "--output-root", default=os.path.join(project_root, "outputs", "rendering"),
        help="Directory for generated rendering results."
    )
    parser.add_argument(
        "--model", action="append", dest="models",
        help="Process only this model-folder name; may be specified more than once."
    )
    parser.add_argument(
        "--max-views", type=int, default=None,
        help="Optional limit on processed views, intended for a quick smoke test."
    )
    args = parser.parse_args()
    if args.max_views is not None and args.max_views < 1:
        parser.error("--max-views must be at least 1")

    input_root = args.input_root
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)

    # W = 512
    # H = 512
    W = 512
    H = 512
    WINDOW_R_WORLD = 1.5   #1.5
    WINDOW_R_WORLD_L = 20  #20
    TOOL_OFFSET_A = 1.5

    R_SMALL_3D = WINDOW_R_WORLD
    R_MEDIUM_3D = WINDOW_R_WORLD_L

    T_GLOBAL = 35   #35
    DIFF_THRESHOLD_S = 0
    DIFF_THRESHOLD_M = 20  #20
    

    if torch.cuda.is_available():
        # Jittor controls placement globally; CUDA_VISIBLE_DEVICES selects the
        # physical GPU before launch, so use the first visible device here.
        jt.flags.use_cuda = 1
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    vecs_text = """
        0.0998749 -0.995 0
        -0.127236 -0.985 0.116559
        0.159563 -0.965 0.208122
        -0.092068 -0.935 0.342489
        0.378973 -0.915 0.1384
        -0.393227 -0.905 0.162319
        0.139344 -0.885 0.44425
        -0.298278 -0.855 0.42427
        0.420748 -0.835 0.354607
        -0.56465 -0.825 0.0233504
        -0.0274038 -0.805 0.592642
        0.613963 -0.785 0.0826074
        -0.51875 -0.775 0.360934
        0.326014 -0.755 0.568937
        -0.265898 -0.725 0.635353
        0.627783 -0.705 0.329944
        -0.695261 -0.695 0.183269
        0.12497 -0.675 0.727157
        -0.51894 -0.645 0.56096
        0.524491 -0.625 0.578173
        -0.143852 -0.595 0.790747
        0.789058 -0.575 0.216247
        -0.734071 -0.565 0.376715
        0.316128 -0.545 0.776555
        -0.433413 -0.515 0.739546
        0.708363 -0.495 0.503186
        -0.8678 -0.485 0.108154
        0.0322655 -0.465 0.884723
        0.894362 -0.445 0.0457273
        -0.694141 -0.435 0.573536
        0.51598 -0.415 0.749359
        -0.28479 -0.385 0.877878
        0.857067 -0.365 0.363609
        -0.881226 -0.355 0.312115
        0.235717 -0.335 0.912257
        -0.586792 -0.305 0.7501
        0.70044 -0.285 0.654339
        -0.0939264 -0.255 0.962368
        0.95544 -0.235 0.178631
        -0.827182 -0.225 0.514922
        0.442625 -0.205 0.872959
        -0.425983 -0.175 0.887645
        0.849883 -0.155 0.503661
        -0.96838 -0.145 0.203015
        0.117901 -0.125 0.985127
        -0.712438 -0.095 0.695275
        0.631407 -0.075 0.771816
        -0.228754 -0.045 0.972444
        0.949387 -0.025 0.313114
        -0.911736 -0.015 0.410503
        0.329255 0.005 0.944228
        -0.548679 0.035 0.8353
        0.783427 0.055 0.619046
        -0.995234 0.065 0.0726883
        -0.014577 0.085 0.996274
        0.989286 0.105 0.101426
        -0.797094 0.115 0.592804
        0.519527 0.135 0.843722
        -0.351854 0.165 0.921398
        0.883911 0.185 0.429507
        -0.940041 0.195 0.279816
        0.195722 0.215 0.956801
        -0.635585 0.245 0.732125
        0.67009 0.265 0.693365
        -0.141252 0.295 0.944999
        0.922803 0.315 0.221831
        -0.827336 0.325 0.458137
        0.381164 0.345 0.857723
        -0.443239 0.375 0.814196
        0.765376 0.395 0.508108
        -0.903275 0.405 0.141665
        0.0615329 0.425 0.903099
        0.89536 0.445 0.0174683
        -0.668229 0.455 0.588596
        0.521694 0.475 0.708668
        -0.240285 0.505 0.828998
        0.793611 0.525 0.307499
        -0.787074 0.535 0.307065
        0.23343 0.555 0.798427
        -0.479302 0.585 0.654251
        0.598708 0.605 0.524903
        -0.788339 0.615 0.0172067
        -0.0507286 0.635 0.770845
        0.746773 0.655 0.115349
        -0.621256 0.665 0.414507
        0.349819 0.685 0.639063
        -0.282433 0.715 0.639536
        0.593951 0.735 0.327105
        -0.648225 0.745 0.157415
        0.0966825 0.765 0.636732
        -0.420543 0.795 0.437171
        0.380886 0.815 0.436693
        -0.105958 0.845 0.524164
        0.481249 0.865 0.142038
        -0.434947 0.875 0.212594
        0.160096 0.895 0.416346
        -0.198477 0.925 0.324009
        0.262898 0.945 0.194575
        -0.294988 0.955 0.0309341
        0.00376534 0.975 0.222173
        0.0996262 0.995 0.00704444
        0 0 1
        """
    vecs = np.array([list(map(float, line.replace(',', ' ').split()))
                     for line in vecs_text.strip().splitlines()
                     if line.strip() != ""], dtype=np.float32)

    for shape_name in sorted(os.listdir(input_root)):
        if args.models is not None and shape_name not in args.models:
            continue
        shape_t0 = time.perf_counter()
        shape_times = {}

        # -------- scan obj --------
        t_path0 = time.perf_counter()
        shape_dir = os.path.join(input_root, shape_name)
        if not os.path.isdir(shape_dir):
            continue


        obj_files = [f for f in os.listdir(shape_dir) if f.lower().endswith(".obj")]
        shape_times["scan_obj_files"] = time.perf_counter() - t_path0
        if not obj_files:
            print(f"[Skip] {shape_name} 未找到 .obj 文件")
            continue

        obj_path = os.path.join(shape_dir, obj_files[0])
        print(f"\n=== 处理模型: {shape_name} ===")

        # -------- load obj --------
        t_load0 = time.perf_counter()
        mesh = trimesh.load(obj_path, process=False)
        verts_np = mesh.vertices
        faces_np = mesh.faces
        shape_times["load_obj"] = time.perf_counter() - t_load0

        # -------- center & tensor --------
        t_prep0 = time.perf_counter()
        center = (verts_np.min(axis=0) + verts_np.max(axis=0)) / 2.0
        verts_centered_np = verts_np - center
        num_vertices = verts_np.shape[0]

        verts_t = torch.from_numpy(verts_centered_np).float().to(device)
        faces_t = torch.from_numpy(faces_np).long().to(device)

        base_normals_t = compute_vertex_normals(verts_t, faces_t)

        # [OPT-1] 全局安全状态改为 GPU tensor
        is_safe_t = torch.zeros(num_vertices, dtype=torch.bool, device=device)

        shape_times["prepare_base_mesh_normals"] = time.perf_counter() - t_prep0

        shape_out_root = os.path.join(output_root, shape_name)
        os.makedirs(shape_out_root, exist_ok=True)

        R_cam = torch.tensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=torch.float32, device=device)

        # -------- visibility sort --------
        t_sort0 = time.perf_counter()
        vis_txt_candidates = [
            os.path.join(shape_dir, "visibility_matrix.txt"),
            os.path.join(shape_dir, "vis_matrix.txt"),
            os.path.join(shape_dir, "visible_matrix.txt"),
        ]
        vis_txt_path = None
        for p in vis_txt_candidates:
            if os.path.exists(p):
                vis_txt_path = p
                break

        Vmat_use = None
        Vmat_use_t = None  # [OPT-1] GPU version
        if vis_txt_path is not None:
            try:
                Vmat = load_visibility_matrix(vis_txt_path)
                if Vmat.shape[0] == vecs.shape[0]:
                    t_sort11 = time.perf_counter()
                    greedy_order = greedy_sort_views(Vmat)
                    shape_times["visibility_sort_real"] = time.perf_counter() - t_sort11
                    vecs_use = vecs[np.array(greedy_order, dtype=np.int32)]
                    Vmat_use = Vmat[np.array(greedy_order, dtype=np.int32)]
                    Vmat_use_t = torch.from_numpy(Vmat_use).to(device)
                    print(f"[Info] 已从 {vis_txt_path} 读入可见性矩阵并完成贪心排序。")
                else:
                    vecs_use = vecs
                    print("[Warning] 可见性矩阵行数不一致，使用原始方向顺序。")
            except Exception as e_greedy:
                vecs_use = vecs
                print(f"[Warning] 读入/贪心排序可见性矩阵失败: {e_greedy}")
        else:
            vecs_use = vecs
            print("[Info] 未找到可见性矩阵 .txt，使用原始方向顺序。")
        shape_times["visibility_sort"] = time.perf_counter() - t_sort0

        # -------- rasterizer init --------
        t_rast0 = time.perf_counter()
        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False
        )
        dummy_focal = torch.tensor([[1.0, 1.0]], device=device)
        dummy_cam = OrthographicCameras(
            R=R_cam.unsqueeze(0),
            T=torch.tensor([[0.0, 0.0, 1.0]], device=device),
            focal_length=dummy_focal,
            device=device
        )
        rasterizer = MeshRasterizer(cameras=dummy_cam, raster_settings=raster_settings).to(device)
        shape_times["init_rasterizer"] = time.perf_counter() - t_rast0

        print(f"--- {shape_name}: 第一轮渲染并检测 ---")

        rotate_total = 0.0
        render_total = 0.0
        detect_total = 0.0

        # 统计池化次数（以参与池化的“点数”为单位）
        pool_small_cnt_stage1 = 0   # 第一轮，小窗口 S
        pool_large_cnt_stage1 = 0   # 第一轮，大窗口 L
        pool_small_cnt_stage2 = 0   # 第二轮，小窗口 S
        pool_large_cnt_stage2 = 0   # 第二轮，大窗口 L

        # 每方向缓存（给第二轮） —— [OPT-1] 全部存 GPU tensor
        view_cache_list = []

        for i, v_np in enumerate(vecs_use):
            if args.max_views is not None and i >= args.max_views:
                print(f"[Info] Reached --max-views={args.max_views}; stopping view loop.")
                break
            view_t0 = time.perf_counter()
            view_times = {}
            det_times = {}
            total_checked_points_firstpass = 0
            newly_safe_count = 0

            # early stop
            t_early0 = time.perf_counter()
            if not bool((is_safe_t == 0).any().item()):
                view_times["early_stop_check"] = time.perf_counter() - t_early0
                print(f"[{shape_name} - {i+1:03d}] 所有点已安全，提前结束。")
                break
            view_times["early_stop_check"] = time.perf_counter() - t_early0

            with torch.no_grad():
                # -------- calc rot --------
                t_rot0 = time.perf_counter()
                v_norm = v_np / (np.linalg.norm(v_np) + 1e-8)
                target = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                axis = np.cross(v_norm, target)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-8:
                    if np.dot(v_norm, target) > 0:
                        rot_matrix = np.eye(3, dtype=np.float32)
                    else:
                        rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
                else:
                    axis /= axis_norm
                    angle = np.arccos(np.clip(np.dot(v_norm, target), -1.0, 1.0))
                    rot = R.from_rotvec(axis * angle)
                    rot_matrix = rot.as_matrix().astype(np.float32)

                rot_matrix_t = torch.from_numpy(rot_matrix).float().to(device)
                view_times["calc_rot_matrix"] = time.perf_counter() - t_rot0

                # -------- apply rotation --------
                t_apply0 = time.perf_counter()
                verts_rotated_t = verts_t @ rot_matrix_t.T
                mesh_rotated = Meshes(verts=[verts_rotated_t], faces=[faces_t])
                normals_rotated_t = base_normals_t @ rot_matrix_t.T
                view_times["apply_rotation_build_mesh"] = time.perf_counter() - t_apply0

                # -------- bbox + cam --------
                t_bbox0 = time.perf_counter()
                max_scale = compute_bounding_box_info(verts_rotated_t, margin=0.3)
                D = max_scale * 1.5
                T_cam = torch.tensor([0.0, 0.0, D], device=device)
                scale_factor = 2.0 / max_scale
                pixels_per_unit = (W / 2) * scale_factor
                WINDOW_R = int(np.ceil(R_SMALL_3D * pixels_per_unit))
                WINDOW_R_L = int(np.ceil(R_MEDIUM_3D * pixels_per_unit))
                view_times["bbox_and_cam_params"] = time.perf_counter() - t_bbox0

                # -------- rasterize --------
                t_rend0 = time.perf_counter()
                out = rasterize_jittor(
                    mesh_rotated, R_cam, T_cam,
                    rasterizer=rasterizer, W=W, H=H,
                    device=device, max_scale=max_scale
                )
                view_times["rasterize"] = time.perf_counter() - t_rend0

            # ---------------- detect ----------------
            det_t0 = time.perf_counter()

            tri_id_map = out["tri_id_map"]
            depth_map_t = out["depth_map"]

            visible_face_ids_t = torch.unique(tri_id_map[tri_id_map >= 0]).long()

            if visible_face_ids_t.numel() == 0:
                view_cache_list.append({
                    "vert_ids_t": torch.empty(0, dtype=torch.long, device=device),
                    "u_p_t": torch.empty(0, dtype=torch.long, device=device),
                    "v_p_t": torch.empty(0, dtype=torch.long, device=device),
                    "z_offset_t": torch.empty(0, dtype=torch.float32, device=device),
                    "u_surf_t": torch.empty(0, dtype=torch.long, device=device),
                    "v_surf_t": torch.empty(0, dtype=torch.long, device=device),
                    "z_surf_t": torch.empty(0, dtype=torch.float32, device=device),
                    "global_min_depth_t": depth_map_t.min(),
                    "WINDOW_R": WINDOW_R,
                    "WINDOW_R_L": WINDOW_R_L
                })
                det_times["detect_total"] = time.perf_counter() - det_t0
                view_times["detect_total"] = det_times["detect_total"]
                view_times["view_total"] = time.perf_counter() - view_t0

                rotate_total += view_times["calc_rot_matrix"] + view_times["apply_rotation_build_mesh"]
                render_total += view_times["rasterize"]
                detect_total += view_times["detect_total"]

                print(f"[{shape_name} - {i+1:03d}] 无可见面，跳过检测。")
                continue

            t_visv0 = time.perf_counter()
            visible_faces_t = faces_t[visible_face_ids_t]
            visible_vert_ids_t = torch.unique(visible_faces_t.reshape(-1)).long()
            det_times["visible_vertex_extract"] = time.perf_counter() - t_visv0

            focal_length = torch.tensor([[scale_factor, scale_factor]], dtype=torch.float32, device=device)
            cameras_for_proj = OrthographicCameras(
                R=R_cam.unsqueeze(0),
                T=T_cam.unsqueeze(0),
                focal_length=focal_length,
                device=device
            )

            with torch.no_grad():
                t_proj0 = time.perf_counter()
                visible_verts_world = verts_rotated_t[visible_vert_ids_t]
                xy_z_screen = cameras_for_proj.transform_points_screen(
                    visible_verts_world, image_size=(H, W)
                )
                det_times["project_visible_vertices"] = time.perf_counter() - t_proj0

                t_dedupe0 = time.perf_counter()
                xy_coords = xy_z_screen[..., :2].long()
                z_depths = xy_z_screen[..., 2]
                u = xy_coords[..., 0]
                v = xy_coords[..., 1]
                mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)

                if not mask.any():
                    det_times["screen_filter_sort_dedupe"] = time.perf_counter() - t_dedupe0
                    continue

                u_valid = u[mask]
                v_valid = v[mask]
                z_valid = z_depths[mask]
                ids_valid = visible_vert_ids_t[mask]

                z_sorted, z_idx = torch.sort(z_valid)
                u_sorted = u_valid[z_idx]
                v_sorted = v_valid[z_idx]
                ids_sorted = ids_valid[z_idx]

                vu_pairs_sorted = torch.stack([v_sorted, u_sorted], dim=1)
                is_duplicate = torch.cat([
                    torch.zeros(1, dtype=torch.bool, device=device),
                    (vu_pairs_sorted[1:] == vu_pairs_sorted[:-1]).all(dim=1)
                ])
                is_first = (is_duplicate == 0)

                u_final = u_sorted[is_first]
                v_final = v_sorted[is_first]
                z_final = z_sorted[is_first]
                ids_final = ids_sorted[is_first]
                det_times["screen_filter_sort_dedupe"] = time.perf_counter() - t_dedupe0

                # 工具偏移点投影
                t_tool0 = time.perf_counter()
                verts_final_world = verts_rotated_t[ids_final]
                normals_final_world = normals_rotated_t[ids_final]
                tool_centers_world_t = verts_final_world + (normals_final_world * TOOL_OFFSET_A)
                tool_xy_z_screen = cameras_for_proj.transform_points_screen(
                    tool_centers_world_t, image_size=(H, W)
                )
                det_times["project_tool_offset_points"] = time.perf_counter() - t_tool0

                # filter tool points
                t_tool_filter0 = time.perf_counter()
                z_offset_all = tool_xy_z_screen[..., 2]
                xy_p = tool_xy_z_screen[..., :2].long()
                u_p_all = xy_p[..., 0]
                v_p_all = xy_p[..., 1]
                p_mask_valid = (u_p_all >= 0) & (u_p_all < W) & (v_p_all >= 0) & (v_p_all < H)

                u_p_final = u_p_all[p_mask_valid]
                v_p_final = v_p_all[p_mask_valid]
                z_offset = z_offset_all[p_mask_valid]

                u_final = u_final[p_mask_valid]
                v_final = v_final[p_mask_valid]
                z_final = z_final[p_mask_valid]
                ids_final = ids_final[p_mask_valid]
                det_times["tool_point_filter"] = time.perf_counter() - t_tool_filter0

                # build full depth map for FIRST PASS pooling (float32)
                t_depthmap0 = time.perf_counter()
                vtx_depth_map = torch.full((H, W), float('inf'), device=device, dtype=torch.float32)
                vtx_depth_map[v_final, u_final] = z_final.to(torch.float32)

                # ===【新增：边界膨胀 1 像素】===
                vtx_depth_map = dilate_depth_map_1px(vtx_depth_map, max_depth_diff=1)
                det_times["build_vtx_depth_map"] = time.perf_counter() - t_depthmap0

            # [OPT-1] first-pass 的 needs/allow/筛选全部在 GPU 做
            t_need0 = time.perf_counter()
            needs_check_t = (is_safe_t[ids_final] == 0)
            if Vmat_use_t is not None and i < Vmat_use_t.shape[0]:
                allow_t = Vmat_use_t[i, ids_final]
            else:
                allow_t = torch.ones_like(needs_check_t)
            firstpass_t = needs_check_t & allow_t
            total_checked_points_firstpass = int(firstpass_t.sum().item())
            det_times["filter_needs_check"] = time.perf_counter() - t_need0

            # === 缓存（第二轮用稀疏 surf depth） ===
            view_cache_list.append({
                "vert_ids_t": ids_final,      # GPU
                "u_p_t": u_p_final,
                "v_p_t": v_p_final,
                "z_offset_t": z_offset,
                "u_surf_t": u_final,
                "v_surf_t": v_final,
                "z_surf_t": z_final.to(torch.float32),
                "global_min_depth_t": depth_map_t.min(),
                "WINDOW_R": WINDOW_R,
                "WINDOW_R_L": WINDOW_R_L
            })

            # 第一轮真正判定：改为 先 Global，再小窗口 S，再大窗口 L
            if firstpass_t.any():
                with torch.no_grad():
                    t_gather0 = time.perf_counter()
                    u_p_t = u_p_final[firstpass_t]
                    v_p_t = v_p_final[firstpass_t]
                    z_offset_t = z_offset[firstpass_t].float()
                    ids_t = ids_final[firstpass_t]
                    det_times["gather_tocheck_tensors"] = time.perf_counter() - t_gather0

                    pooled_total_time = 0.0
                    t_poolM = 0.0
                    t_poolS = 0.0

                    # 0) 先做全局阈值检查（全局池化标量 + 逐点比较）
                    t_global0 = time.perf_counter()
                    a_t = view_cache_list[-1]["global_min_depth_t"]  # GPU scalar
                    is_unsafe = (z_offset_t - a_t) > T_GLOBAL
                    det_times["global_threshold_check"] = time.perf_counter() - t_global0

                    # 1) 小窗口 S (R=WINDOW_R)，只对 Global 后仍未 unsafe 的点检查
                    remain_S = (is_unsafe == 0)
                    if remain_S.any():
                        t_pool0 = time.perf_counter()

                        # 参与小窗口池化的点数（第一轮）
                        u_p_S = u_p_t[remain_S]
                        v_p_S = v_p_t[remain_S]
                        z_offset_S = z_offset_t[remain_S]
                        pool_small_cnt_stage1 += int(u_p_S.numel())

                        min_S = compute_min_pooled_depth_for_vertices(
                            vtx_depth_map=vtx_depth_map,
                            u_final=u_p_S, v_final=v_p_S,
                            W=W, H=H, R=WINDOW_R, device=device
                        )
                        valid_S = min_S < float('inf')
                        if valid_S.any():
                            diff_S = z_offset_S - min_S
                            bad_S = valid_S & (diff_S > DIFF_THRESHOLD_S)
                            if bad_S.any():
                                bad_idx_global = torch.nonzero(remain_S).reshape(-1)[bad_S]
                                is_unsafe[bad_idx_global] = True
                        t_poolS = time.perf_counter() - t_pool0
                        pooled_total_time += t_poolS

                    # 2) 大窗口 L (R=WINDOW_R_L)，只对 Global+S 后仍未 unsafe 的点检查
                    remain_L = (is_unsafe == 0)
                    if remain_L.any():
                        t_pool0 = time.perf_counter()

                        u_p_L = u_p_t[remain_L]
                        v_p_L = v_p_t[remain_L]
                        z_offset_L = z_offset_t[remain_L]
                        # 参与大窗口池化的点数（第一轮）
                        pool_large_cnt_stage1 += int(u_p_L.numel())

                        min_L = compute_min_pooled_depth_for_vertices(
                            vtx_depth_map=vtx_depth_map,
                            u_final=u_p_L, v_final=v_p_L,
                            W=W, H=H, R=WINDOW_R_L, device=device
                        )
                        valid_L = min_L < float('inf')
                        if valid_L.any():
                            diff_L = z_offset_L - min_L
                            bad_L = valid_L & (diff_L > DIFF_THRESHOLD_M)
                            if bad_L.any():
                                bad_idx_L = torch.nonzero(remain_L).reshape(-1)[bad_L]
                                is_unsafe[bad_idx_L] = True
                        t_poolM = time.perf_counter() - t_pool0
                        pooled_total_time += t_poolM

                    det_times["pool_S"] = t_poolS
                    det_times["pool_M"] = t_poolM
                    det_times["pooled_checks_total"] = pooled_total_time

                    # 写回 safe 标记
                    t_write0 = time.perf_counter()
                    safe_mask = (is_unsafe == 0)
                    if safe_mask.any():
                        safe_ids_t = ids_t[safe_mask]
                        newly_safe_count = int(safe_ids_t.numel())
                        is_safe_t[safe_ids_t] = True
                    det_times["writeback_safe_status"] = time.perf_counter() - t_write0

            det_times["detect_total"] = time.perf_counter() - det_t0
            view_times["detect_total"] = det_times["detect_total"]
            view_times["view_total"] = time.perf_counter() - view_t0

            rotate_total += view_times["calc_rot_matrix"] + view_times["apply_rotation_build_mesh"]
            render_total += view_times["rasterize"]
            detect_total += view_times["detect_total"]

            print(
                f"[{shape_name} - {i+1:03d}/{len(vecs_use)}] "
                f"Rend={view_times['rasterize']:.3f}s, "
                f"Det={view_times['detect_total']:.3f}s, "
                f"poolM={det_times.get('pool_M', 0):.4f}s, poolS={det_times.get('pool_S', 0):.4f}s, "
                f"新安全={newly_safe_count}, "
                f"第一轮检测={total_checked_points_firstpass}"
            )

        # ------------------------------------------------------------------
        # 第二轮：无渲染 + ROI 重建（GPU cache）（带计时）
        # ------------------------------------------------------------------
        print(f"--- {shape_name}: 第二轮无渲染检查（ROI稀疏depth） ---")
        t_second0 = time.perf_counter()

        unsafe_mask_global_t = (is_safe_t == 0)
        second_pass_new_safe = 0
        second_pass_pool_total = 0.0
        second_pass_roi_build_total = 0.0

        if unsafe_mask_global_t.any() and len(view_cache_list) > 0:
            for vi, cache in enumerate(view_cache_list):
                t_sv0 = time.perf_counter()

                vert_ids_t = cache["vert_ids_t"]
                if vert_ids_t.numel() == 0:
                    continue

                in_unsafe_t = unsafe_mask_global_t[vert_ids_t]
                if not in_unsafe_t.any():
                    continue

                ids_t = vert_ids_t[in_unsafe_t]
                u_p_t = cache["u_p_t"][in_unsafe_t]
                v_p_t = cache["v_p_t"][in_unsafe_t]
                z_offset_t = cache["z_offset_t"][in_unsafe_t].float()

                R_max = max(cache["WINDOW_R_L"], cache["WINDOW_R"])

                # === 1) 计算 ROI bbox ===
                umin = int(max((u_p_t.min() - R_max).item(), 0))
                umax = int(min((u_p_t.max() + R_max).item(), W - 1))
                vmin = int(max((v_p_t.min() - R_max).item(), 0))
                vmax = int(min((v_p_t.max() + R_max).item(), H - 1))
                roi_w = umax - umin + 1
                roi_h = vmax - vmin + 1

                # === 2) ROI 重建（GPU）===
                t_roi0 = time.perf_counter()

                u_surf_t = cache["u_surf_t"]
                v_surf_t = cache["v_surf_t"]
                z_surf_t = cache["z_surf_t"]

                if jt.flags.use_cuda:
                    # Jittor specializes kernels by tensor shape. Creating a
                    # different ROI shape for nearly every view repeatedly
                    # recompiles gather/dilation kernels on CUDA. Use the
                    # fixed full screen instead: every pooling window remains
                    # inside the ROI margin, so sampled depths are identical.
                    roi_w, roi_h = W, H
                    umin, vmin = 0, 0
                    roi_depth_map_t = torch.full(
                        (H, W), float('inf'), device=device, dtype=torch.float32
                    )
                    if u_surf_t.numel() > 0:
                        roi_depth_map_t[v_surf_t, u_surf_t] = z_surf_t
                else:
                    in_roi_t = (
                        (u_surf_t >= umin) & (u_surf_t <= umax) &
                        (v_surf_t >= vmin) & (v_surf_t <= vmax)
                    )
                    u_roi_t = u_surf_t[in_roi_t] - umin
                    v_roi_t = v_surf_t[in_roi_t] - vmin
                    z_roi_t = z_surf_t[in_roi_t]
                    roi_depth_map_t = torch.full(
                        (roi_h, roi_w), float('inf'),
                        device=device, dtype=torch.float32
                    )
                    if u_roi_t.numel() > 0:
                        roi_depth_map_t[v_roi_t, u_roi_t] = z_roi_t

                # ===【新增：ROI 边界膨胀 1 像素】===
                roi_depth_map_t = dilate_depth_map_1px(roi_depth_map_t, max_depth_diff=1)

                roi_build_time = time.perf_counter() - t_roi0
                second_pass_roi_build_total += roi_build_time

                # === 3) 平移 unsafe 点坐标到 ROI ===
                u_p_shift_t = (u_p_t - umin).long()
                v_p_shift_t = (v_p_t - vmin).long()

                # === 4) 第二轮判定（改顺序：Global -> S -> L） ===
                t_pool0 = time.perf_counter()

                # 初始化为 Global 检查结果
                a_t = cache["global_min_depth_t"]
                is_unsafe_t = (z_offset_t - a_t) > T_GLOBAL

                # 4.1 小窗口 S：只对 Global 后仍未 unsafe 的点
                remain_S = (is_unsafe_t == 0)
                if remain_S.any():
                    # 参与第二轮小窗口池化的点数
                    u_p_S = u_p_shift_t[remain_S]
                    v_p_S = v_p_shift_t[remain_S]
                    z_offset_S = z_offset_t[remain_S]
                    pool_small_cnt_stage2 += int(u_p_S.numel())

                    min_S = compute_min_pooled_depth_for_vertices(
                        vtx_depth_map=roi_depth_map_t,
                        u_final=u_p_S, v_final=v_p_S,
                        W=roi_w, H=roi_h, R=cache["WINDOW_R"], device=device
                    )
                    valid_S = min_S < float('inf')
                    if valid_S.any():
                        diff_S = z_offset_S - min_S
                        bad_S = valid_S & (diff_S > DIFF_THRESHOLD_S)
                        if bad_S.any():
                            bad_idx_S = torch.nonzero(remain_S).reshape(-1)[bad_S]
                            is_unsafe_t[bad_idx_S] = True

                # 4.2 大窗口 L（只对 Global+S 后仍未 unsafe 的点）
                remain_L = (is_unsafe_t == 0)
                if remain_L.any():
                    u_p_L = u_p_shift_t[remain_L]
                    v_p_L = v_p_shift_t[remain_L]
                    z_offset_L = z_offset_t[remain_L]
                    # 参与第二轮大窗口池化的点数
                    pool_large_cnt_stage2 += int(u_p_L.numel())

                    min_L = compute_min_pooled_depth_for_vertices(
                        vtx_depth_map=roi_depth_map_t,
                        u_final=u_p_L, v_final=v_p_L,
                        W=roi_w, H=roi_h, R=cache["WINDOW_R_L"], device=device
                    )
                    valid_L = min_L < float('inf')
                    if valid_L.any():
                        diff_L = z_offset_L - min_L
                        bad = valid_L & (diff_L > DIFF_THRESHOLD_M)
                        if bad.any():
                            bad_idx_L = torch.nonzero(remain_L).reshape(-1)[bad]
                            is_unsafe_t[bad_idx_L] = True

                pool_time = time.perf_counter() - t_pool0
                second_pass_pool_total += pool_time

                safe_mask_t = (is_unsafe_t == 0)
                newly_cnt = 0
                if safe_mask_t.any():
                    safe_ids_t = ids_t[safe_mask_t]
                    # 这些一定是之前 unsafe 的
                    is_safe_t[safe_ids_t] = True
                    unsafe_mask_global_t[safe_ids_t] = False
                    newly_cnt = int(safe_ids_t.numel())
                    second_pass_new_safe += newly_cnt

                sv_time = time.perf_counter() - t_sv0
                print(
                    f"  [SecondPass-View {vi+1:03d}] "
                    f"roi={roi_h}x{roi_w}, roi_build={roi_build_time:.4f}s, "
                    f"pool={pool_time:.4f}s, total={sv_time:.4f}s, "
                    f"new_safe={newly_cnt}"
                )

        second_total = time.perf_counter() - t_second0
        print(
            f"[Second Pass Summary] 新增安全点={second_pass_new_safe}, "
            f"roi_build_total={second_pass_roi_build_total:.3f}s, "
            f"pool_total={second_pass_pool_total:.3f}s, "
            f"second_total={second_total:.3f}s"
        )

        # ------------------------------------------------------------------
        # 输出结果 + shape summary timing
        # ------------------------------------------------------------------
        unsafe_vertex_indices = torch.nonzero(is_safe_t == 0).reshape(-1).numpy()

        safe_status_path = os.path.join(shape_out_root, f"safety_status.txt")
        with open(safe_status_path, "w") as f:
            safe_cpu = is_safe_t.detach().cpu().numpy()
            for safe in safe_cpu:
                f.write(f"{int(safe)}\n")
        print(f"[输出] safety_status: {safe_status_path}")

        shape_times["views_rotate_total"] = rotate_total
        shape_times["views_render_total"] = render_total
        shape_times["views_detect_total"] = detect_total
        shape_times["views_second_total"] = second_total
        shape_times["shape_total"] = time.perf_counter() - shape_t0 - shape_times["visibility_sort"]

        cutter_txt_path = os.path.join(shape_out_root, f"cutter.txt")
        write_numbers(
            cutter_txt_path,
            [WINDOW_R_WORLD, WINDOW_R_WORLD_L, T_GLOBAL, DIFF_THRESHOLD_M]
        )

        timing_txt_path = os.path.join(shape_out_root, f"timing.txt")
        timing_order = [
            "scan_obj_files",
            "load_obj",
            "prepare_base_mesh_normals",
            "visibility_sort",
            "visibility_sort_real",
            "init_rasterizer",
            "views_rotate_total",
            "views_render_total",
            "views_detect_total",
            "views_second_total",
            "shape_total",
        ]
        timing_values = [shape_times[k] for k in timing_order if k in shape_times]
        write_timing_numbers(timing_txt_path, timing_values)

        print(f"\n[Timing - Shape Summary] {shape_name}")
        for k in [
            "scan_obj_files",
            "load_obj",
            "prepare_base_mesh_normals",
            "visibility_sort",
            "visibility_sort_real",
            "init_rasterizer",
            "views_rotate_total",
            "views_render_total",
            "views_detect_total",
            "views_second_total",
            "shape_total"
        ]:
            if k in shape_times:
                print(f"  - {k:28s}: {shape_times[k]:.6f}s")

        # 新增：池化次数统计输出
        print(f"\n[Pooling - Shape Summary] {shape_name}")
        print(f"  第一轮 小窗口(S) 池化点数: {pool_small_cnt_stage1}")
        print(f"  第一轮 大窗口(L) 池化点数: {pool_large_cnt_stage1}")
        print(f"  第二轮 小窗口(S) 池化点数: {pool_small_cnt_stage2}")
        print(f"  第二轮 大窗口(L) 池化点数: {pool_large_cnt_stage2}")
        total_pool_points = (
            pool_small_cnt_stage1 + pool_large_cnt_stage1 +
            pool_small_cnt_stage2 + pool_large_cnt_stage2
        )
        print(f"  两轮池化总点数(所有窗口合计): {total_pool_points}")

        if unsafe_vertex_indices.size > 0:
            print(f"  发现 {unsafe_vertex_indices.size} / {num_vertices} 个全局不安全点。生成球体 obj ...")

            all_sphere_verts = []
            all_sphere_faces = []
            v_offset = 0

            verts_min = verts_centered_np.min(axis=0)
            verts_max = verts_centered_np.max(axis=0)
            model_scale = np.linalg.norm(verts_max - verts_min)
            sphere_radius = model_scale / 200.0
            print(f"  使用球体半径: {sphere_radius:.4f}")

            SPHERE_SEGMENTS = 16
            for v_idx in unsafe_vertex_indices:
                center_coord_uncentered = verts_centered_np[v_idx] + center
                sphere_verts_centered, sphere_faces_centered = create_sphere_mesh(
                    center=[0.0, 0.0, 0.0],
                    radius=sphere_radius,
                    segments=SPHERE_SEGMENTS
                )
                sphere_verts_world = sphere_verts_centered + center_coord_uncentered
                all_sphere_verts.append(sphere_verts_world)
                shifted_faces = sphere_faces_centered + v_offset
                all_sphere_faces.append(shifted_faces)
                v_offset += sphere_verts_centered.shape[0]

            final_verts = np.concatenate(all_sphere_verts, axis=0)
            final_faces = np.concatenate(all_sphere_faces, axis=0)
            unsafe_obj_path = os.path.join(shape_out_root, f"unsafe_points_sphere.obj")
            save_obj(unsafe_obj_path, final_verts, final_faces)
            print(f"  unsafe obj: {unsafe_obj_path}")
        else:
            print("  所有点安全。")

        print(f"=== {shape_name} 处理完成 ===\n")
