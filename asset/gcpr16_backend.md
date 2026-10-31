# GCPR16 pretrained reconstruction backend

[中文版](#中文版)

This independent backend reads the existing [Eigen matrices](../data/matrices/) directly. It retains the current scene/reconstruction method names and leaves the original `vehicle_reconstruction` backend unchanged. No ApolloScape meshes, PCA training, Ceres build, OpenPCDet, custom CUDA extensions, or `voxeltorch` are needed to run it.

## Environment and commands

Use Conda environment `shapepriors-gcpr16` (tested Python 3.11.17). Runtime dependencies are listed in [requirements-gcpr16.txt](../requirements-gcpr16.txt). NumPy performs fitting in CPU float64, and scikit-image extracts meshes. PyTorch is optional for tensor input/output; PyTorch3D is optional for native `Meshes` output.

From the repository root:

```bat
scripts\run_gcpr16.bat --test
scripts\run_gcpr16.bat
```

```bash
bash scripts/run_gcpr16.sh --test
bash scripts/run_gcpr16.sh
```

Both launchers select the named Conda environment. Alternatively, set `GCPR16_PYTHON` to its Python executable. They do not fall back to the default/base Python. To install missing runtime dependencies:

```bash
conda run -n shapepriors-gcpr16 python -m pip install -r requirements-gcpr16.txt
```

The default demo reads the bundled points and boxes. It writes `vehicles.obj`, `reconstruction.npz`, and `diagnostics.json` under `outputs/gcpr16/`. The archive includes original box order, coefficients `z`, optimized scene poses, fitted flags, mesh-to-box indices, packed vertices/faces, and the packed rooftop mask. Each fit reports solver status and costs; skipped boxes retain their slot.

Original eligibility filters are defaults: at least 200 points per vehicle, native forward translation at most 30 m. For sparse or surrounding LiDAR scenes, deliberately relax these:

```bat
scripts\run_gcpr16.bat --min-points 11 --max-distance 0 --outer-iterations 10
```

`--max-distance 0` disables the forward-distance filter. The original filter tests forward translation, not Euclidean range, so negative-forward objects are not excluded by it. The Python API uses the existing strict `count > point_threshold` convention; `point_threshold=199` means at least 200.

## Compatible Python usage

Replace the reconstruction and segmentation imports with the independent backend:

```python
import numpy as np
from vehicle_reconstruction_gcpr16 import vehicle_reconstructor
from vehicle_reconstruction_gcpr16.scene_segmentation import point_cloud_scene

# Default path is <repository>/data/matrices, independent of the working directory.
model = vehicle_reconstructor()
# Or: vehicle_reconstructor(matrices_dir="data/matrices", k=5)
scene = point_cloud_scene(
    pts=np.load("data/00043_pts.npy"),
    gt_boxes=np.load("data/00043_gt.npy"),
)
vehicles = scene.get_vehicles()
results = scene.pose_estimate(model, iter=10)
mesh, packed_rooftop_idx = vehicles.reconstruct_at_scene(
    model, threshold=199, smoothing=False, show_rooftop=True,
)
vertices = mesh.verts_packed()
triangles = mesh.faces_packed()
z = vehicles.vehicle_latent
optimized_xyz_yaw = vehicles.translation
```

The scene constructor also accepts `vehi_reconstructor=model`, followed by `scene.pose_estimate(iter=10)`. `iter` is the maximum number of pose/shape alternations, and `max_solver_iterations=100` is the subproblem iteration limit. Each outer loop uses the original cost-improvement threshold `0.1`; an explicit outer limit fixes the ineffective iteration cap in the C++ demo.

Supported decoder names include `encode`, `encode_aux`, `decode`, `decode_aux`, `reconstruct`, and the existing misspelling `reconsturct`. `decode(z)` returns native `[B,60,40,60]` fields. `decode(z, to_meshes=True)` returns meshes in native camera/local coordinates. Native-grid TSDF arrays are accepted by `encode`; arbitrary mesh-to-TSDF preprocessing is not part of this pretrained backend. Omit the old `fit_meshes()`/ApolloScape preparation calls. The grid is fixed by the supplied matrices; adaptive grid arguments that conflict with it raise an error.

For a tensor workflow, instantiate with `device=points.device, dtype=points.dtype` if desired. `encode`/`decode` preserve tensor device, floating dtype and autograd. Scene optimization explicitly detaches inputs for the CPU solver; it is not differentiable through the solver. Fitted coefficients and scene outputs return as floating tensors on the input device. Integer tensor coordinates are promoted rather than truncating fitted values. If PyTorch3D is installed, mesh output is its native `Meshes`; otherwise an adapter provides the list, packed, padded, indexing, offset and update methods used by this project. It is not a full replacement for every PyTorch3D operation.

`vehicle_latent` now stores the original, unwhitened `z`; old latent values are not interchangeable. The optional `save_parameters`/`load_parameters` snapshot format is a numeric NumPy archive even if a caller supplies a different filename extension. Old PyTorch/PCA checkpoints are incompatible. Normal startup loads the three original `.bin` files, not a snapshot.

## Coordinate conventions

All geometry is in meters. Default `coordinate_frame="lidar"` accepts `[x,y,z,l,w,h,yaw]` with xyz at the box center, +x forward, +y left, +z up, and yaw counterclockwise around +z. Points may be `[N,3]` or `[N,4]`; intensity is ignored for fitting. The adapter maps scene points to original camera axes `(right,down,forward)`, shifts the pose origin from box center to bottom center, and uses `rotation_y = pi - yaw` because the published native shape faces -z. Optimized output converts back to the input frame. Mesh size comes from the prior, not a per-box dimension rescale.

`coordinate_frame="camera"` accepts `[bottom_x,bottom_y,bottom_z,l,w,h,rotation_y]` already in the original native convention. Raw KITTI label angles require `rotation_y -= pi/2`, matching the original reader. Unlike LiDAR boxes, camera xyz is the bottom center. `get_local_point_clouds()` returns native local coordinates; `get_training_point_clouds()` adds `(3,3,3)` to obtain native grid coordinates. Transformations use the latest optimized pose.

The supplied scene adapter selects points inside oriented 3D boxes (first matching box owns an overlap) and preserves actual points without ROI padding/duplication. This replaces the C++ stereo/2D-rectangle/ground-plane data preparation. To fit exactly the same observations as the original experiment, call `fit_vehicle(prior, camera_points, native_pose)` directly. Camera calibration, disparity conversion and ground-plane estimation are intentionally outside the existing point-cloud API.

## Original model and objective

The fixed grid has 60 x 40 x 60 samples, origin `(-3,-3,-3)`, spacing `0.1`, z-fastest flattening, and default `r=5`. The last sample is `(2.9,0.9,2.9)`. The supplied basis contains 38 components; the first five are used without modifying or regenerating the files. The Eigen loader handles both 32- and 64-bit dimension headers, float64 column-major payloads, and validates sizes, dimensions, finite entries and selected positive singular values.

The model is `shape = mean + V z`. Define standard Huber `h_delta(s)=s²/2` for `|s|<=delta`, otherwise `delta*(|s|-delta/2)`:

- Shape cost: `sum(h_0.1(SDF(point))) / (0.03*N) + 0.5*sum((z/S)²)`.
- Pose cost: `sum(h_0.1(SDF(point))) / (0.03*N) + (ty-reference_ty)²/6`.
- The pose reference is frozen at the start of each pose subproblem, as in the C++ code.
- Shape interpolation accepts lower cell indices `0..dimension-2`; outside, residual is `3` and shape gradient is zero.
- Pose interpolation accepts indices `1..dimension-2`; outside, residual is `norm(local_point)+3` with its spatial gradient.

Fitting alternates three translations plus native yaw with shape coefficients, using analytic derivatives. Dense damped LM/IRLS steps solve an augmented least-squares system and are accepted only if they decrease the exact robust objective. This preserves the published code's objective but replaces Ceres, so floating-point iterates and final results are not promised to be identical. Original translation-Jacobian block overflow and the erroneous y entry in the rotation derivative are corrected. NaN/Inf inputs are rejected.

History `total_cost` is the sum of the costs when each subproblem finished, reproducing the original outer stopping calculation. `final_costs` reevaluates both subproblems at the same returned state. `success` denotes a finite usable solution, not proof of a unique or correct reconstruction; inspect solver statuses for stationary/iteration-limit cases.

The default mesh is the unsmoothed zero isosurface. Optional `smoothing=True` applies Taubin smoothing as an extension. Rooftop flags mark vertices within 0.2 m of each mesh's highest point; this is an existing-project extension, not an original GCPR16 output. `vehicles.reconstructed_indices` maps output mesh order back to input box order.

## Source and validation

Method: Engelmann, Stückler, and Leibe, “Joint Object Pose Estimation and Shape Reconstruction in Urban Street Scenes Using 3D Shape Priors,” GCPR 2016. This backend was written after inspecting the original C++ objectives, matrix format and coordinate transforms. See the [original source](https://github.com/RWTHVision/ShapePriors_GCPR16) and its GPL-3.0-or-later notices for that implementation and its supplied assets.

Regression tests cover Eigen layout/errors, real matrices, PCA round trips, independent trilinear values, both boundary modes, finite-difference shape/space/pose Jacobians, exact objective scaling, degenerate solves, actual-prior surface fitting, ROI ownership, coordinate round trips, scene pose updates, packed meshes and empty batches. Optional tensor tests are skipped explicitly when PyTorch is absent. Use the commands above to see the actual pass/skip count in your environment.

# 中文版

[English](#gcpr16-pretrained-reconstruction-backend)

这个独立版本直接读取已有的 [Eigen 权重](../data/matrices/)，保留当前场景与重建方法名称。原有 `vehicle_reconstruction` 实现不变。运行不需要 ApolloScape 网格、重新训练 PCA、编译 Ceres、OpenPCDet、自定义 CUDA 扩展或 `voxeltorch`。

## 环境与运行

使用 Conda 环境 `shapepriors-gcpr16`，已验证 Python 3.11.17。依赖见 [requirements-gcpr16.txt](../requirements-gcpr16.txt)：NumPy 在 CPU 上进行双精度拟合，scikit-image 提取网格。PyTorch 用于可选的张量输入输出，PyTorch3D 用于可选的原生 `Meshes` 输出。

在仓库根目录运行：

```bat
scripts\run_gcpr16.bat --test
scripts\run_gcpr16.bat
```

```bash
bash scripts/run_gcpr16.sh --test
bash scripts/run_gcpr16.sh
```

两个入口均选择上述 Conda 环境；也可以用 `GCPR16_PYTHON` 指定该环境的解释器，不会回退到系统或 base Python。缺少运行依赖时执行：

```bash
conda run -n shapepriors-gcpr16 python -m pip install -r requirements-gcpr16.txt
```

默认读取仓库自带点云和框，在 `outputs/gcpr16/` 下生成 `vehicles.obj`、`reconstruction.npz` 和 `diagnostics.json`。归档保留输入框顺序、形状系数、优化后位姿、拟合标记、输出网格对应的输入框索引、顶点、三角面和车顶掩码；跳过的框保留原位置，诊断包含跳过原因或求解状态与代价。

默认采用原版筛选：每车至少 200 点，原版相机前向平移不超过 30 米。对于稀疏或环视 LiDAR，可显式放宽：

```bat
scripts\run_gcpr16.bat --min-points 11 --max-distance 0 --outer-iterations 10
```

`--max-distance 0` 关闭前向距离筛选。该规则与原版一致，判断前向平移而非欧氏距离，因此不会排除前向坐标为负的目标。Python API 保留 `count > point_threshold` 的严格大于约定，故 `point_threshold=199` 对应至少 200 点。

## 接口迁移

上面的英文段落有完整 Python 示例。迁移的主要改动是：

```python
from vehicle_reconstruction_gcpr16 import vehicle_reconstructor
from vehicle_reconstruction_gcpr16.scene_segmentation import point_cloud_scene

model = vehicle_reconstructor()  # 默认直接读取仓库 data/matrices，与工作目录无关
scene = point_cloud_scene(pts=pts, gt_boxes=gt_boxes)
vehicles = scene.get_vehicles()
results = scene.pose_estimate(model, iter=10)
mesh, roof = vehicles.reconstruct_at_scene(
    model, threshold=199, smoothing=False, show_rooftop=True)
```

也支持在场景构造时传 `vehi_reconstructor=model`，之后调用 `scene.pose_estimate(iter=10)`。`iter` 表示位姿/形状交替次数上限；子问题默认最多 100 次迭代。每轮按原版代价改善阈值 0.1 停机，并补上原 C++ 示例未生效的外循环次数上限。

保留 `encode`、`encode_aux`、`decode`、`decode_aux`、`reconstruct` 及旧拼写 `reconsturct`。`decode(z)` 输出 `[B,60,40,60]`，`to_meshes=True` 输出原版相机局部坐标下的网格。编码接受该原生网格的距离场；不包含任意网格转 TSDF 的训练预处理。请删除旧的 ApolloScape 准备与 `fit_meshes()` 调用。网格受权重固定，冲突的自适应网格参数会明确报错。

张量编解码保留设备、浮点类型和自动求导；可在构造时传 `device`、`dtype`。场景拟合显式转为 CPU 双精度，不支持对求解过程反向传播。结果返回输入设备上的浮点张量，整数框会提升为浮点，避免截断。安装 PyTorch3D 时返回其原生 `Meshes`；否则提供本项目常用的列表、拼接、填充、索引、偏移与更新方法适配器，不承诺实现整个 PyTorch3D API。

`vehicle_latent` 保存原版未白化的 `z`，不能直接使用旧版潜变量。可选 `save_parameters`/`load_parameters` 快照使用数值型 NumPy 归档，即使文件名使用其他扩展名；不兼容旧 PyTorch/PCA checkpoint。正常启动始终直接读取三个原版 `.bin` 权重。

## 坐标约定

全部几何量使用米。默认 `coordinate_frame="lidar"`：框为 `[x,y,z,l,w,h,yaw]`，xyz 是几何中心，+x 向前、+y 向左、+z 向上，yaw 绕 +z 逆时针。点云可以有 xyz 或 xyz 加强度，拟合忽略强度。适配器转换到原版的相机轴（右、下、前），将位姿原点移到车底中心，并采用 `rotation_y = pi - yaw`，因为原版标准车辆朝向 -z。优化后转换回输入坐标。网格尺寸来自先验，不按每个框额外缩放。

`coordinate_frame="camera"` 接受 `[bottom_x,bottom_y,bottom_z,l,w,h,rotation_y]`，xyz 为车底中心，朝向已经是原版内部约定。原始 KITTI 标签角度需要先减 `pi/2`，与 C++ 读取器一致。`get_local_point_clouds()` 返回原生局部坐标；`get_training_point_clouds()` 再加 `(3,3,3)` 得到网格坐标。所有变换使用最新优化后的位姿。

场景适配器通过有向三维框取点，重叠点归第一个框，保留真实点数，不把补零或重复采样当观测。这一层替换原版双目视差、二维框和地面筛选。若要使用与原版完全一致的观测点，可直接调用 `fit_vehicle(prior, camera_points, native_pose)`；标定、视差转换及地面估计位于现有点云接口之外。

## 复刻内容与明确差异

固定网格为 `60×40×60`、原点 `(-3,-3,-3)`、间距 `0.1`，展平时 z 最快，最后采样点为 `(2.9,0.9,2.9)`。权重含 38 个主成分，默认使用前 5 个，不修改或重建原文件。读取器支持 32/64 位 Eigen 维度头、双精度列主序数据，检查文件大小、矩阵维度、有限性和选用的正奇异值。

保持原版形状表示 `shape = mean + V z`、Huber 阈值 0.1、数据项 `1/(0.03*N)`、形状正则 `0.5*sum((z/S)²)`，位姿正则为竖直平移相对本次子问题初始值的平方除以 6。形状越界残差固定为 3 且形状梯度为零；位姿越界残差为局部点范数加 3，保留位置梯度。两子问题沿用原版不同的有效网格边界。

位姿的三个平移分量与朝向、形状系数交替优化。使用解析梯度、阻尼 LM/IRLS 和增广最小二乘，只接受精确 Huber 目标下降的步骤。目标与原版保持一致，但求解器替换了 Ceres，迭代轨迹与最终值不保证逐位相同。修复原版平移 Jacobian 块越界写入及旋转导数中错误的 y 项，并拒绝 NaN/Inf 输入。

历史 `total_cost` 按原版记录两个子问题各自结束时的代价之和；`final_costs` 在最终同一状态重新评估。`success` 表示有限、可用的解，不代表重建唯一或准确，驻点和迭代上限需要结合诊断判断。

默认输出未平滑的零等值面。`smoothing=True` 的 Taubin 平滑和距最高点 0.2 米内顶点的车顶标记属于当前项目扩展，非原版输出。`vehicles.reconstructed_indices` 将输出网格顺序映射回输入框。

方法来源为 Engelmann、Stückler、Leibe 的 GCPR 2016 论文及 [原版源码](https://github.com/RWTHVision/ShapePriors_GCPR16)。原版实现及其提供的资源附有 GPL-3.0-or-later 声明。测试覆盖矩阵格式、真实权重、PCA 往返、独立插值值、边界、有限差分梯度、目标系数、退化求解、真实先验表面拟合、ROI 归属、坐标往返、场景位姿更新、网格和空批次；缺少 PyTorch 时显式跳过可选张量测试。
