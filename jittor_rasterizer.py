"""Hard orthographic mesh rasterizer for the Jittor port.

The rasterizer exposes only the two buffers consumed by DeepMill++: nearest
face id and z-buffer. On CUDA it uses a Jittor ``jt.code`` custom operation;
the NumPy scan converter is retained as a CPU fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np
import jittor as jt


class Meshes:
    def __init__(self, verts, faces):
        if len(verts) != 1 or len(faces) != 1:
            raise ValueError("DeepMill++ currently rasterizes one mesh at a time")
        self.verts = verts[0]
        self.faces = faces[0]

    def to(self, device=None):
        return self


@dataclass
class RasterizationSettings:
    image_size: tuple[int, int]
    blur_radius: float = 0.0
    faces_per_pixel: int = 1
    cull_backfaces: bool = False


class OrthographicCameras:
    def __init__(self, R, T, focal_length, device=None):
        self.R = R
        self.T = T
        self.focal_length = focal_length

    def transform_points_screen(self, points, image_size):
        height, width = image_size
        rot = self.R[0] if len(self.R.shape) == 3 else self.R
        trans = self.T[0] if len(self.T.shape) == 2 else self.T
        focal = self.focal_length[0] if len(self.focal_length.shape) == 2 else self.focal_length
        view = points @ rot + trans
        x_ndc = view[:, 0] * focal[0]
        y_ndc = view[:, 1] * focal[1]
        # PyTorch3D screen convention: origin at top-left.
        u = (1.0 - x_ndc) * (width - 1) * 0.5
        v = (1.0 - y_ndc) * (height - 1) * 0.5
        return jt.stack((u, v, view[:, 2]), dim=1)


@dataclass
class _Fragments:
    pix_to_face: jt.Var
    zbuf: jt.Var


def _scan_convert(screen, faces, height, width):
    depth = np.full((height, width), np.inf, dtype=np.float32)
    face_map = np.full((height, width), -1, dtype=np.int32)
    eps = 1.0e-7

    for face_id, (ia, ib, ic) in enumerate(faces):
        a, b, c = screen[ia], screen[ib], screen[ic]
        min_x = max(int(np.floor(min(a[0], b[0], c[0]))), 0)
        max_x = min(int(np.ceil(max(a[0], b[0], c[0]))), width - 1)
        min_y = max(int(np.floor(min(a[1], b[1], c[1]))), 0)
        max_y = min(int(np.ceil(max(a[1], b[1], c[1]))), height - 1)
        if min_x > max_x or min_y > max_y:
            continue

        denom = ((b[1] - c[1]) * (a[0] - c[0]) +
                 (c[0] - b[0]) * (a[1] - c[1]))
        if abs(denom) <= eps:
            continue
        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        wa = ((b[1] - c[1]) * (xx - c[0]) +
              (c[0] - b[0]) * (yy - c[1])) / denom
        wb = ((c[1] - a[1]) * (xx - c[0]) +
              (a[0] - c[0]) * (yy - c[1])) / denom
        wc = 1.0 - wa - wb
        inside = (wa >= -eps) & (wb >= -eps) & (wc >= -eps)
        if not inside.any():
            continue
        z = wa * a[2] + wb * b[2] + wc * c[2]
        old = depth[min_y:max_y + 1, min_x:max_x + 1]
        update = inside & (z >= 0.0) & (z < old)
        if update.any():
            old[update] = z[update]
            face_roi = face_map[min_y:max_y + 1, min_x:max_x + 1]
            face_roi[update] = face_id
    return depth, face_map


def _cuda_hard_rasterize(screen, faces, height, width):
    """Rasterize projected triangles on CUDA with a packed atomic z-buffer.

    Non-negative IEEE-754 float bit patterns preserve depth ordering. Packing
    those bits above the face id lets ``atomicMin`` choose the nearest triangle
    and resolve exact ties by lower face id, as the sequential CPU loop does.
    """
    packed = jt.code(
        (height, width), "int64", [screen.float32(), faces.int64()],
        cuda_header=r'''
            #include <cuda_runtime.h>
            #include <stdint.h>
            #include <math.h>

            __global__ static void deepmill_init_zbuffer(unsigned long long* keys, int count) {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < count) keys[index] = 0x7f800000ffffffffULL;
            }

            __global__ static void deepmill_rasterize_triangles(
                const jittor::float32* screen, const jittor::int64* faces,
                unsigned long long* keys,
                int vertex_count, int face_count, int height, int width) {
                const int face_id = blockIdx.x;
                if (face_id >= face_count) return;

                const int64_t ia = faces[face_id * 3];
                const int64_t ib = faces[face_id * 3 + 1];
                const int64_t ic = faces[face_id * 3 + 2];
                if (ia < 0 || ib < 0 || ic < 0 ||
                    ia >= vertex_count || ib >= vertex_count || ic >= vertex_count) return;

                const float ax = screen[ia * 3], ay = screen[ia * 3 + 1], az = screen[ia * 3 + 2];
                const float bx = screen[ib * 3], by = screen[ib * 3 + 1], bz = screen[ib * 3 + 2];
                const float cx = screen[ic * 3], cy = screen[ic * 3 + 1], cz = screen[ic * 3 + 2];
                const int min_x = max((int)floorf(fminf(ax, fminf(bx, cx))), 0);
                const int max_x = min((int)ceilf(fmaxf(ax, fmaxf(bx, cx))), width - 1);
                const int min_y = max((int)floorf(fminf(ay, fminf(by, cy))), 0);
                const int max_y = min((int)ceilf(fmaxf(ay, fmaxf(by, cy))), height - 1);
                if (min_x > max_x || min_y > max_y) return;

                const float denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
                if (fabsf(denom) <= 1.0e-7f) return;

                const int roi_width = max_x - min_x + 1;
                const int pixel_count = roi_width * (max_y - min_y + 1);
                for (int linear = threadIdx.x; linear < pixel_count; linear += blockDim.x) {
                    const int px = min_x + linear % roi_width;
                    const int py = min_y + linear / roi_width;
                    const float x = (float)px + 0.5f;
                    const float y = (float)py + 0.5f;
                    const float wa = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / denom;
                    const float wb = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / denom;
                    const float wc = 1.0f - wa - wb;
                    if (wa < -1.0e-7f || wb < -1.0e-7f || wc < -1.0e-7f) continue;
                    const float z = wa * az + wb * bz + wc * cz;
                    if (!(z >= 0.0f) || !isfinite(z)) continue;
                    const unsigned long long key =
                        (static_cast<unsigned long long>(__float_as_uint(z)) << 32) |
                        static_cast<unsigned int>(face_id);
                    atomicMin(keys + py * width + px, key);
                }
            }
        ''',
        cuda_src=r'''
            const int count = out0_shape0 * out0_shape1;
            deepmill_init_zbuffer<<<(count + 255) / 256, 256>>>(
                reinterpret_cast<unsigned long long*>(out0_p), count);
            deepmill_rasterize_triangles<<<in1_shape0, 256>>>(
                in0_p, in1_p, reinterpret_cast<unsigned long long*>(out0_p),
                in0_shape0, in1_shape0, out0_shape0, out0_shape1);
        ''')
    depth, face_map = jt.code(
        [(height, width), (height, width)], ["float32", "int32"], [packed],
        cuda_header=r'''
            #include <cuda_runtime.h>
            #include <stdint.h>

            __global__ static void deepmill_unpack_zbuffer(
                const jittor::int64* packed, jittor::float32* depth,
                jittor::int32* face_map, int count) {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index >= count) return;
                const unsigned long long key = static_cast<unsigned long long>(packed[index]);
                const unsigned int face = static_cast<unsigned int>(key & 0xffffffffULL);
                if (face == 0xffffffffU) {
                    depth[index] = __int_as_float(0x7f800000);
                    face_map[index] = -1;
                } else {
                    depth[index] = __uint_as_float(static_cast<unsigned int>(key >> 32));
                    face_map[index] = static_cast<int>(face);
                }
            }
        ''',
        cuda_src=r'''
            const int count = out0_shape0 * out0_shape1;
            deepmill_unpack_zbuffer<<<(count + 255) / 256, 256>>>(
                in0_p, out0_p, out1_p, count);
        ''')
    return depth, face_map


class MeshRasterizer:
    def __init__(self, cameras, raster_settings):
        self.cameras = cameras
        self.settings = raster_settings

    def to(self, device=None):
        return self

    def __call__(self, mesh, cameras=None):
        camera = cameras or self.cameras
        height, width = self.settings.image_size
        projected = camera.transform_points_screen(mesh.verts, (height, width))
        use_cuda = bool(jt.flags.use_cuda) and not bool(
            int(os.environ.get("DEEPMILL_FORCE_CPU_RASTERIZER", "0")))
        if use_cuda:
            depth, face_map = _cuda_hard_rasterize(projected, mesh.faces, height, width)
        else:
            screen = projected.numpy()
            faces = mesh.faces.numpy().astype(np.int64, copy=False)
            depth, face_map = _scan_convert(screen, faces, height, width)
            depth, face_map = jt.array(depth), jt.array(face_map)
        return _Fragments(
            pix_to_face=face_map[None, :, :, None],
            zbuf=depth[None, :, :, None],
        )
