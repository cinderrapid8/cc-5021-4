# 服装走秀角色蒙皮检视器 (Runway Skinning Inspector)

读取可分发的骨架 / 网格权重 / 动画资产（`rskin` JSON 格式），计算逆绑定矩阵与每帧骨骼变换，
在 GPU 上执行线性混合蒙皮（LBS），并提供 CPU 参考实现的逐顶点数值核对。

## 运行

```bash
npm run gen     # 重新生成内置资产 assets/runway_character.rskin.json
npm test        # 运行测试（CPU 参考、校验政策、动画采样、打包布局；GPU 测试需可选依赖 gl）
npm run serve   # 打开 http://localhost:8080/src/app/index.html
```

GPU ↔ CPU 数值核对在检视器界面内实时进行（任意支持 WebGL1 的浏览器即可）。
在无浏览器的环境中，`test/test_gpu_vs_cpu.mjs` 会在安装了可选依赖
`npm i -D gl`（headless-gl，需要系统编译环境）后自动执行同一套核对；未安装时跳过。

## 空间约定：模型空间与世界空间

- **模型空间（model space）**：网格位置、法线与逆绑定矩阵（IBM）所在的创作空间。
  蒙皮全程在模型空间计算。
- **世界空间（world space）**：通过网格节点的世界矩阵到达：

```
skinMatrix_j = inverse(meshWorld) · jointWorld_j · IBM_j
worldPos     = meshWorld · ( Σ_k w_k · skinMatrix_{j_k} · modelPos )
```

`meshWorld` / `jointWorld` 都是完整节点图求值的结果，因此骨架与网格节点之上
可以存在任意非单位父变换（平移、旋转、**非均匀缩放**），结果依然正确：
共享的祖先变换在 `inverse(meshWorld) · jointWorld` 中自然抵消，
蒙皮结果对共享祖先的运动**不变**（有测试覆盖：移动 Stage 节点后模型空间蒙皮逐顶点不变）。
内置资产故意包含 `Stage`（平移+35° 偏航）、`UnitFix`（非均匀缩放
[0.011, 0.01, 0.0095]）与独立的 `BodyMesh` 缩放节点，且 IBM 经由此真实节点图计算，
绑定姿势下全部 17 个关节的 skinMatrix 均为单位阵（测试断言）。

## 权重校验政策（加载时，明确拒绝或修正）

| 情况 | 政策 |
|---|---|
| 任何 NaN / Infinity（位置、法线、权重、IBM、动画） | **拒绝整个资产**（AssetError，含元素定位） |
| 关节索引 < 0 或 ≥ 关节数（权重非零时） | **拒绝** |
| 权重 < -1e-4 | **拒绝** |
| 权重 ∈ [-1e-4, 0) | 钳制为 0，然后归一化（警告） |
| 权重和 ≠ 1（有限且 > 0） | **自动归一化**，记录警告（顶点列表与最大偏差） |
| 同一顶点重复关节 | 合并权重（警告） |
| 零影响顶点（四权重全 0） | 保留，**刚性跟随网格节点**（恒等蒙皮），警告列出 |
| IBM 奇异（不可逆） | **拒绝** |

由于所有数值在加载时即被校验或修复，NaN 不会进入 CPU 蒙皮器或 GPU 纹理，
也就不会扩散成整屏异常。GPU 侧另有一层保障：零影响顶点在着色器中走恒等分支，
`normalize(0)` 不会产生 NaN。

## 法线处理与非均匀缩放支持范围

- 位置：`p' = Σ w_k · (skinMatrix_k · p)`（模型空间）。
- 法线：每个关节使用其蒙皮矩阵的**逆转置 3×3**（normal matrix），
  `n' = normalize(Σ w_k · N_k · n)`。逆转置对任意可逆仿射矩阵（含非均匀缩放、
  剪切）都是精确的，有解析测试（变换后法线与切线保持垂直）。
- 世界空间法线再经 `inverse-transpose(meshWorld)` 变换用于着色。
- **支持范围**：节点链中的任意非均匀缩放（包括网格节点自身的非均匀缩放）对位置
  完全支持；法线在每个关节内部精确。唯一的近似是 LBS 标准做法——
  对每关节法线矩阵的加权混合，而不是对混合后矩阵再取逆转置；
  这与所有实时蒙皮引擎一致。每顶点最多 4 个影响（超出在生成端截断并归一化）。

## 测试动画（内置资产）

1. **elbow_flex**（1.0s）：左肘屈曲 0 → -115° → 0（绕局部 Y 轴，T  Pose 下即竖直轴），
   检验肘部权重过渡带（UpperArm/Forearm 平滑混合）。
2. **forearm_twist**（1.2s）：前臂绕长轴扭转 0 → 180° → 0，
   前臂权重沿轴向渐变，扭转量随之渐变（90° 处可观察到梯度扭转；
   接近 180° 时可见 LBS 固有的体积塌陷，属预期行为）。
3. **big_turn**（2.0s）：髋部偏航 0 → 200° 并伴随位移（根运动），
   检验模型空间/世界空间分离：世界空间中角色整体转身移动，
   模型空间蒙皮结果与共享祖先变换无关。

每个动画都有 CPU 参考的解析数值测试（旋转后的顶点位置与解析解对比，误差 < 1e-3 cm）。

## GPU ↔ CPU 数值核对

- GPU：WebGL1 顶点着色器蒙皮（骨骼矩阵经 RGBA32F 纹理传入，每关节 4 纹素；
  法线矩阵每关节 3 纹素）。读回通道将每个顶点的位置/法线分量编码为
  RGBA8 32 位定点（量程 ±256，精度 ~4e-5），不依赖浮点帧缓冲，任何实现可复现。
- CPU：`src/core/skinning.js` 参考实现（float64）。
- 界面“GPU ↔ CPU 数值核对”面板给出最大位置/法线误差与最差顶点；
  误差热力图模式按顶点误差着色；选中顶点显示双方逐分量数值。
- 容差：位置 1e-3（模型空间单位，cm），法线 1e-3。GPU 为 float32，
  CPU 为 float64，误差主要来自浮点精度与定点编码。

## 检视配置（保存 / 复现）

“检视配置”面板可导出当前状态为 JSON（资产名、动画、时间、选中顶点、
视图模式、热力关节、相机、是否连续核对），下载存档；把 JSON 粘回文本框并
“导入配置”即可复现同一检视状态，用于问题报告与回归确认。

## rskin 格式（v1）

单文件 JSON：`nodes`（拓扑序 TRS 节点）、`skeleton.joints`（节点索引）、
`skeleton.inverseBindMatrices`、`mesh`（positions/normals/joints/weights/indices，
`mesh.node` 指向网格节点）、`animations`（channels: node/path/times/values/
interpolation=LINEAR|STEP）。所有批量数值为 base64 小端类型化数组，
避免文本浮点往返。生成器见 `tools/gen_assets.mjs`。

## 目录

```
src/core/    数学、格式、校验、加载、动画采样、CPU 蒙皮、骨骼纹理打包
src/gpu/     WebGL1 着色器、GPU 蒙皮与读回、渲染器
src/app/     检视器界面（无构建步骤，原生 ES 模块）
tools/       资产生成器、静态服务器
test/        Node 测试（npm test）
assets/      生成的内置资产
```
