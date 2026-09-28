# DeepMill++ (Jittor)

Official Jittor implementation of the ACM SIGGRAPH 2026 paper **“DeepMill++:
Neural Guidance Meets Rasterization for Efficient Accessibility Analysis.”**

> Fanchao Zhong, Yao Zhang, Guanze Xin, Peng-Shuai Wang, Lin Lu, Changhe Tu,
> and Haisen Zhao. *DeepMill++: Neural Guidance Meets Rasterization for
> Efficient Accessibility Analysis.* ACM SIGGRAPH 2026 Conference Papers,
> Article 40, 2026. [Paper](https://doi.org/10.1145/3799902.3811091)

This version uses **Jittor** for DeepM inference and the tensor operations in
DeepMill++. PyTorch and PyTorch3D are not required. The pipeline takes an OBJ
mesh, predicts a directional accessibility matrix with DeepM, and refines the
prediction with rasterization to produce one safe/unsafe label per mesh vertex.

## Contents

- [1. Environment](#1-environment)
- [2. Quick start: run dcSim end to end](#2-quick-start-run-dcsim-end-to-end)
- [3. Understand the inputs and outputs](#3-understand-the-inputs-and-outputs)
- [4. Run your own OBJ model](#4-run-your-own-obj-model)
- [5. Run with an existing visibility matrix](#5-run-with-an-existing-visibility-matrix)
- [6. Evaluation](#6-evaluation)
- [7. Cutter parameters](#7-cutter-parameters)
- [8. Troubleshooting](#8-troubleshooting)
- [9. Jittor port notes](#9-jittor-port-notes)
- [10. Citation](#10-citation)

## 1. Environment

### 1.1 Requirements

The recommended platform is Ubuntu Linux with Python 3.10 or 3.11. Jittor
compiles operators on first use, so a working C++ compiler is required.

On Ubuntu, install the system build tools if they are not already available:

```bash
sudo apt update
sudo apt install -y build-essential
```

An NVIDIA GPU with a Jittor-compatible CUDA installation is recommended for
DeepM training and rasterization. `deepmill2.py` includes a CUDA custom hard
rasterizer/z-buffer; CPU remains supported through a NumPy fallback, but is
much slower for the 102-view rasterization stage.

### 1.2 Create the environment

```bash
conda create -n deepmill-jittor python=3.10 -y
conda activate deepmill-jittor

cd /path/to/DeepMill++Jittor
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Replace `/path/to/DeepMill++Jittor` with your clone location. All remaining
commands in this README are run from the project root.

Verify that Jittor imports correctly:

```bash
python -c "import jittor as jt; print('Jittor', jt.__version__, 'CUDA:', jt.has_cuda)"
```

The first Jittor command may spend several minutes compiling and caching
operators. Later runs reuse that cache.

### 1.3 Enable the CUDA custom rasterizer (recommended)

`deepmill2.py` enables Jittor CUDA automatically when Jittor detects a usable
GPU. Select a GPU before starting the process; for example, to use physical
GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python deepmill2.py --model dcSim
```

Jittor compiles the custom z-buffer with `nvcc`. If automatic CUDA discovery
selects the wrong compiler, set the compiler and architecture explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 \
nvcc_path=/usr/local/cuda/bin/nvcc \
cuda_archs=86 \
python deepmill2.py --model dcSim
```

Use the architecture supported by the selected `nvcc`; `86` is a portable
choice for Ampere and, with CUDA 11.7, RTX 4090 systems. The first GPU run
compiles and caches the Jittor custom operations, so it is slower than later
runs. For CPU debugging only, force the fallback with
`DEEPMILL_FORCE_CPU_RASTERIZER=1`.

### 1.4 Check the pretrained model

DeepM inference needs the checkpoint below:

```text
DeepM/pretrained/deepm_00100.model.pth
```

The checkpoint is intentionally not committed to this source repository.
Download the published DeepMill++ checkpoint before running inference:

```bash
mkdir -p DeepM/pretrained
curl -L --fail \
  -o DeepM/pretrained/deepm_00100.model.pth \
  https://raw.githubusercontent.com/YaoZhang-666/DeepMillPlusPlus/main/DeepM/pretrained/deepm_00100.model.pth
```

The download must be stored at this exact path. Verify it with:

```bash
test -f DeepM/pretrained/deepm_00100.model.pth && echo "checkpoint OK"
```

If you clone the original DeepMill++ repository instead, install Git LFS and
run `git lfs pull` to retrieve its large model files. See
`DeepM/pretrained/README.md` for the expected file name.

## 2. Quick start: run dcSim end to end

The following three commands reproduce the complete workflow, starting with
the neural-network prediction rather than a hand-written visibility matrix.

### Step 1: predict directional inaccessibility with DeepM

```bash
python predict_visibility.py \
  --mesh examples/done/dcSim/dcSim.obj \
  --checkpoint DeepM/pretrained/deepm_00100.model.pth \
  --output examples/done/dcSim/deepm_inaccessible_matrix.txt \
  --probabilities examples/done/dcSim/deepm_inaccessible_probabilities.txt
```

The default tool-conditioning vector is `1.5 20 15 30`, matching the DeepM
data-preparation default. Override it when necessary:

```bash
python predict_visibility.py \
  --mesh examples/done/dcSim/dcSim.obj \
  --output examples/done/dcSim/deepm_inaccessible_matrix.txt \
  --tool-params 1.5 20 15 30
```

### Step 2: convert the matrix for DeepMill++

```bash
python prepare_visibility_matrix.py \
  --input examples/done/dcSim/deepm_inaccessible_matrix.txt \
  --mesh examples/done/dcSim/dcSim.obj \
  --output examples/done/dcSim/visibility_matrix.txt
```

DeepM uses `1 = inaccessible`, while DeepMill++ uses `1 = allowed for this
view`. The converter validates and inverts the matrix automatically.

### Step 3: run all 102 rasterization directions

```bash
python deepmill2.py \
  --input-root examples/done \
  --model dcSim \
  --output-root outputs/rendering_network_prediction
```

The final result is written to:

```text
outputs/rendering_network_prediction/dcSim/
├── safety_status.txt          # one line per OBJ vertex: 1=safe, 0=unsafe
├── unsafe_points_sphere.obj   # spheres marking unsafe vertices
├── cutter.txt                 # rasterization cutter settings
└── timing.txt                 # stage timings in seconds
```

Open `unsafe_points_sphere.obj` together with `dcSim.obj` in MeshLab, Blender,
or another OBJ viewer to inspect the unsafe locations.

With the supplied checkpoint and default parameters, the included `dcSim.obj`
has 7,302 vertices. A verified CPU run produced a `102 × 7302` DeepM matrix and
reported 1,077 final unsafe vertices. Exact timings depend on the machine (the
verified run took about 3 seconds for network inference and 59 seconds for the
full rasterization stage). These counts are useful for checking a fresh setup.

On the RTX 4090 used to validate this checkout, the CUDA custom-rasterizer run
completed the measured shape stage in about 1.2 seconds after JIT warm-up.
CUDA and CPU use different floating-point execution paths: pixels choose the
same frontmost face, but vertices close to safety thresholds can receive a
different final label. Treat CPU and GPU counts as tolerance-based checks, not
bit-for-bit golden outputs.

For an installation-only smoke test, process one direction in a separate
output directory:

```bash
python deepmill2.py \
  --model dcSim \
  --max-views 1 \
  --output-root outputs/smoke_test
```

`--max-views 1` intentionally produces an incomplete result and must not be
used for evaluation or reported experiments.

## 3. Understand the inputs and outputs

### 3.1 Per-model input layout

Each model must have its own folder under `examples/done` or another directory
passed through `--input-root`:

```text
examples/done/
└── model_name/
    ├── model_name.obj
    └── visibility_matrix.txt
```

Only one OBJ should be placed in a model folder. Matrix columns must follow the
OBJ vertex order exactly; loading or exporting the mesh with vertex merging can
break that correspondence.

### 3.2 Matrix conventions

| File | Shape | Meaning of `1` |
| --- | --- | --- |
| `deepm_inaccessible_matrix.txt` | `102 × N` | inaccessible in this direction |
| `deepm_inaccessible_probabilities.txt` | `102 × N` | probability of inaccessibility |
| `visibility_matrix.txt` | `102 × N` | allowed for this rasterization view |
| `safety_status.txt` | `N` lines | vertex is safe/reachable in the final result |

Here `N` is the OBJ vertex count. The 102 rows use the fixed direction order in
`deepmill2.py`. The binary DeepM matrix is produced with a default probability
threshold of 0.5; use `predict_visibility.py --threshold VALUE` to change it.

### 3.3 Main scripts

| Script | Purpose |
| --- | --- |
| `predict_visibility.py` | Runs the Jittor DeepM UNet directly on an OBJ. |
| `prepare_visibility_matrix.py` | Validates and converts DeepM output to DeepMill++ convention. |
| `deepmill2.py` | Sorts predicted views and performs rasterization-based refinement. |
| `Calculate_skip_file.py` | Prepares labels and skip indices from geometric results. |
| `Calculate_metrics.py` | Exports comparison metrics to `outputs/metrics.xlsx`. |

## 4. Run your own OBJ model

Assume the new model is named `part01.obj`.

1. Create its input folder and copy the OBJ without changing its vertex order:

   ```bash
   mkdir -p examples/done/part01
   cp /path/to/part01.obj examples/done/part01/part01.obj
   ```

2. Generate the DeepM prediction:

   ```bash
   python predict_visibility.py \
     --mesh examples/done/part01/part01.obj \
     --output examples/done/part01/deepm_inaccessible_matrix.txt \
     --probabilities examples/done/part01/deepm_inaccessible_probabilities.txt \
     --tool-params 1.5 20 15 30
   ```

3. Convert it:

   ```bash
   python prepare_visibility_matrix.py \
     --input examples/done/part01/deepm_inaccessible_matrix.txt \
     --mesh examples/done/part01/part01.obj \
     --output examples/done/part01/visibility_matrix.txt
   ```

4. Run DeepMill++:

   ```bash
   python deepmill2.py --model part01 --output-root outputs/rendering
   ```

To keep input data outside this repository, arrange it in the same folder
layout and pass its parent directory with `--input-root`.

## 5. Run with an existing visibility matrix

If `visibility_matrix.txt` already exists, skip neural prediction and matrix
conversion:

```bash
python deepmill2.py --model model_name --output-root outputs/rendering
```

If an external matrix already uses `1 = reachable`, convert or validate it
without inversion:

```bash
python prepare_visibility_matrix.py \
  --input /path/to/reachable_matrix.txt \
  --mesh examples/done/model_name/model_name.obj \
  --output examples/done/model_name/visibility_matrix.txt \
  --input-is-reachable
```

When no supported matrix name (`visibility_matrix.txt`, `vis_matrix.txt`, or
`visible_matrix.txt`) is present, DeepMill++ runs all directions in their
original order without network guidance.

## 6. Evaluation

The supplied evaluation scripts compare rendering results with independently
generated geometric reference results. Put each reference under:

```text
outputs/geometric/model_name/reachable_vertices.txt
```

Then run:

```bash
python Calculate_skip_file.py
python Calculate_metrics.py
```

The scripts generate `new_safety_status.txt`, `skip.txt`, and
`outputs/metrics.xlsx`. The historical geometric collision-detection program
is not included in this release. Supply a reference status file for each model
when running this optional evaluation; it is not an input consumed
automatically by `deepmill2.py`.

## 7. Cutter parameters

DeepM and the rasterization stage have separate parameter entry points:

- Neural conditioning is set with `predict_visibility.py --tool-params`; its
  default is `1.5 20 15 30`.
- Rasterization constants are near the `deepmill2.py` entry point:
  `WINDOW_R_WORLD`, `WINDOW_R_WORLD_L`, `TOOL_OFFSET_A`, `T_GLOBAL`, and the
  depth-difference thresholds.

Keep these settings consistent with the cutter/holder geometry represented by
your data. Changing the physical tool requires both a matching DeepM
conditioning vector and suitable rasterization constants.

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'jittor'`

Activate the environment and install from the project root:

```bash
conda activate deepmill-jittor
pip install -r requirements.txt
```

### Jittor finds CUDA but the machine should run on CPU

Disable CUDA compiler discovery for that command:

```bash
nvcc_path="" python deepmill2.py --model dcSim
```

The same prefix can be used with `predict_visibility.py`.

### `GLIBCXX_* not found` or a `libstdc++.so.6` error

This usually means the Conda C++ runtime is older than the system compiler
runtime. First update it inside the environment:

```bash
conda install -c conda-forge libstdcxx-ng -y
```

On Ubuntu, a temporary diagnostic workaround is:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python deepmill2.py --model dcSim
```

### Checkpoint not found

Ensure `DeepM/pretrained/deepm_00100.model.pth` exists and is a real file or a
valid symbolic link. See [Section 1.3](#13-check-the-pretrained-model).

### Matrix row or column mismatch

The matrix must contain exactly 102 rows and one column for every OBJ vertex.
Always pass `--mesh` to `prepare_visibility_matrix.py` so this is checked before
the long rasterization run.

### Out of memory

The pretrained UNet has about 159 million parameters. Close other GPU-heavy
programs, use a smaller training batch size, or run OBJ inference on CPU. The
optional probability file is large; omit `--probabilities` if only the binary
matrix is needed.

## 9. Jittor port notes

- DeepM tensors, modules, optimizers, autograd, and PyTorch-format `.pth`
  checkpoint loading are implemented through Jittor.
- Local O-CNN code is included under `DeepM/projects/ocnn`; do not install a
  separate `ocnn` package.
- The PyTorch3D renderer is replaced by `jittor_rasterizer.py`. Projection and
  hard z-buffer rasterization use Jittor; on CUDA the z-buffer is a `jt.code`
  custom operation with a packed atomic depth/face key. NumPy scan conversion
  remains as the CPU fallback.
- Single-process execution is supported. Multi-process distributed training is
  intentionally rejected.
- The CPU fallback prioritizes portability; a full 102-view CPU run can be
  much slower than the CUDA custom-rasterizer path.

See [`DeepM/README.md`](DeepM/README.md) for DeepM dataset preparation,
training, and test-set inference.

## 10. Citation

```bibtex
@inproceedings{zhong2026deepmill,
  author    = {Fanchao Zhong and Yao Zhang and Guanze Xin and Peng-Shuai Wang and Lin Lu and Changhe Tu and Haisen Zhao},
  title     = {DeepMill++: Neural Guidance Meets Rasterization for Efficient Accessibility Analysis},
  booktitle = {ACM SIGGRAPH 2026 Conference Papers},
  articleno = {40},
  numpages  = {11},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3799902.3811091},
  url       = {https://doi.org/10.1145/3799902.3811091}
}
```
