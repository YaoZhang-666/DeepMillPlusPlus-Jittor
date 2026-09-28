import os
from pathlib import Path
import numpy as np

try:
    import trimesh
except ImportError as e:
    raise ImportError(
        "需要安装 trimesh 才能读取 .obj：\n"
        "pip install trimesh"
    ) from e


PROJECT_ROOT = Path(__file__).resolve().parent
DONE_ROOT = PROJECT_ROOT / "examples" / "done"
GEOM_ROOT = PROJECT_ROOT / "outputs" / "geometric"

REACHABLE_NAME = "reachable_vertices.txt"
OUT_STATUS_NAME = "new_safety_status.txt"
OUT_SKIP_NAME = "skip.txt"


def find_obj_in_folder(folder: Path) -> Path:
    """返回该文件夹中唯一/第一个 .obj 文件路径。找不到则抛错。"""
    objs = sorted(folder.glob("*.obj"))
    if not objs:
        raise FileNotFoundError(f"未在 {folder} 找到 .obj 文件")
    return objs[0]  # 若有多个，默认取第一个；可按需改成更严格检查


def read_reachable_indices(path: Path) -> set[int]:
    """
    读取 reachable_vertices.txt，支持：
    - 每行一个整数
    - 一行多个整数（空格/逗号分隔）
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    tokens = text.replace(",", " ").split()
    idx = set()
    for t in tokens:
        # 允许出现像 "39" 这样的纯数字
        if t.strip().lstrip("+-").isdigit():
            idx.add(int(t))
    return idx


def load_vertex_z(obj_path: Path) -> np.ndarray:
    """
    读取 obj 网格顶点，返回 z 坐标数组 (n,).
    注意：如果 obj 是 scene（多个 mesh），会合并为一个 mesh。
    """
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if mesh is None or not hasattr(mesh, "vertices"):
        raise ValueError(f"无法解析网格：{obj_path}")

    v = np.asarray(mesh.vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] < 3:
        raise ValueError(f"顶点数据异常：{obj_path}")
    return v[:, 2]


def write_status_file(out_path: Path, n_verts: int, reachable: set[int]):
    """生成 new_safety_status.txt：总行数 = 顶点数；reachable 中的索引置 1，其余 0。"""
    status = np.zeros(n_verts, dtype=np.int8)

    # 过滤越界索引，避免写入时报错
    # print(list(reachable)[4000:4500])
    valid = [i for i in reachable if 0 <= i < n_verts]
    if valid:
        status[np.array(valid, dtype=np.int64)] = 1
    #print(valid[700:1500])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(map(str, status.tolist())) + "\n", encoding="utf-8")


def write_skip_file(out_path: Path, z: np.ndarray, thresh: float = 1.0):
    """
    skip.txt：记录所有满足 (z - min_z) <= thresh 的顶点索引（0-based）。
    这里按“每行一个索引”写入，最稳妥。
    """
    min_z = float(np.min(z))
    skip_idx = np.where((z - min_z) <= thresh)[0].astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(map(str, skip_idx.tolist())) + "\n", encoding="utf-8")


def main():
    if not DONE_ROOT.exists():
        raise FileNotFoundError(f"done 目录不存在：{DONE_ROOT}")
    if not GEOM_ROOT.exists():
        raise FileNotFoundError(f"geometric 目录不存在：{GEOM_ROOT}")

    done_folders = sorted([p for p in DONE_ROOT.iterdir() if p.is_dir()], key=lambda x: x.name)
    print(f"发现 done 子文件夹数量：{len(done_folders)}")

    ok = 0
    for folder in done_folders:
        name = folder.name
        geom_folder = GEOM_ROOT / name

        try:
            obj_path = find_obj_in_folder(folder)
            if not geom_folder.exists():
                raise FileNotFoundError(f"对应 geometric 子文件夹不存在：{geom_folder}")

            reachable_path = geom_folder / REACHABLE_NAME
            reachable = read_reachable_indices(reachable_path)
            

            z = load_vertex_z(obj_path)
            n_verts = int(z.shape[0])

            # 1) new_safety_status.txt
            out_status = geom_folder / OUT_STATUS_NAME
            write_status_file(out_status, n_verts, reachable)

            # 2) skip.txt
            out_skip = geom_folder / OUT_SKIP_NAME
            write_skip_file(out_skip, z, thresh=1.0)

            ok += 1
            print(f"✅ {name}: verts={n_verts}, reachable={len(reachable)} -> 写入 {out_status.name}, {out_skip.name}")

        except Exception as e:
            print(f"⚠️ {name}: 处理失败 -> {e}")

    print(f"\n完成：成功 {ok}/{len(done_folders)}")

if __name__ == "__main__":
    main()
