# 连续体机械臂编码器精度与工作空间误差分析

## 1. 分析目标

项目提供两套互补脚本：

```text
scripts/cal_accuracy.py
scripts/cal_accuracy_workspace.py
```

- `cal_accuracy.py`：在一个指定连续体姿态上枚举六通道误差盒的全部 `2^6 = 64` 个角点，适合编码器选型和局部最坏情况分析。
- `cal_accuracy_workspace.py`：随机采样项目允许的连续体工作空间，对多种编码器精度统计平均值、95% 分位数和最大值，并输出 TCP 误差热力图。

两套脚本分析的是角度测量误差经过当前理想运动学模型后的传播，不包含结构、负载、安装、基座和标定误差。

## 2. 项目定义

### 2.1 六维编码器状态

每段使用两个总弯曲角，六维输入顺序为：

$$
\boldsymbol\theta =
[\theta_{kx,1},\theta_{ky,1},
 \theta_{kx,2},\theta_{ky,2},
 \theta_{kx,3},\theta_{ky,3}]^{\mathrm T}
$$

这里的输入单位为度，是与局部曲率分量对应的整段总弯曲角，不是曲率值。二者关系为：

$$
\theta_{kx,i}=k_{x,i}L_{f,i},\qquad
\theta_{ky,i}=k_{y,i}L_{f,i}
$$

其中 $L_{f,i}$ 是第 $i$ 段的有效柔性长度。

### 2.2 与项目正运动学的映射

`segment_2dof_forward_kinematics()` 每段接收：

```text
[hinge_x, hinge_y]
```

编码器总弯曲角按照项目坐标约定转换为：

$$
\theta_{hinge_x,i}=-\theta_{ky,i},\qquad
\theta_{hinge_y,i}=\theta_{kx,i}
$$

当前执行臂每段使用 `Y/X/Y/X` 四个离散柔性铰链。同一方向有两个铰链，所以该方向的总弯曲角在两个同轴铰链之间平均分配。

### 2.3 当前几何和限制

脚本不再维护独立的硬编码机器人参数，而是直接读取项目配置：

| 定义 | 配置来源 | 当前值 |
| --- | --- | ---: |
| 三段几何和肌腱路径 | `configs/robots/spatial_arm_executor.yaml` | 每段 40 mm |
| 有效柔性长度 | 同上 | 每段 36.5 mm |
| 末端刚性长度 | 同上 | 每段 3.5 mm |
| 肌腱位移限制 | 同上 | 九根均为 ±20 mm |
| 运动学模式 | `configs/scenarios/mujoco_manual_control.yaml` | `discrete_hinge` |
| 单柔性铰链范围 | `configs/mujoco_dual.yaml` | ±15° |
| 每方向整段总角范围 | 由两个同轴铰链合成 | ±30° |
| 执行臂工具 TCP | `configs/tools/carbon_remover.yaml` | 裸臂端点前方 18 mm |

工具 TCP 使用项目已有刚体坐标组合：

$$
T_{tip}^{TCP}=T_{tip}^{sensor}T_{sensor}^{TCP}
$$

当前两段平移分别为 4 mm 和 14 mm，因此 TCP 距裸臂端点 18 mm。代码直接调用 `compute_tool_tcp_pose()`，不会再单独维护一个工具长度常量。

### 2.4 可行姿态筛选

工作空间脚本先在六维总弯曲角的配置范围内采样，再通过 `BendingSpaceModel` 把角度换算为曲率和九根物理肌腱位移：

$$
\boldsymbol\kappa=
[\theta_{kx,1}/L_{f,1},\theta_{ky,1}/L_{f,1},\ldots]^{\mathrm T}
$$

$$
\Delta\boldsymbol l=C_b\boldsymbol\kappa
$$

只有六个角度通道和九根物理肌腱均满足项目配置限制的姿态才进入统计。

## 3. 单姿态最坏情况分析

运行：

```powershell
python scripts/cal_accuracy.py
```

指定真实姿态和候选精度：

```powershell
python scripts/cal_accuracy.py `
  --theta-deg 10 -5 8 3 -12 6 `
  --accuracy-deg 0.5 0.25 0.1 0.05
```

设单通道绝对误差边界为 $a$，脚本枚举：

$$
\delta\boldsymbol\theta
\in\{-a,+a\}^{6}
$$

对于每个误差角点，分别计算参考位姿和测量位姿。TCP 位置误差为：

$$
e_p=1000\left\|
\boldsymbol p_{TCP}(\boldsymbol\theta+\delta\boldsymbol\theta)
-\boldsymbol p_{TCP}(\boldsymbol\theta)
\right\|_2
$$

单位为毫米。姿态误差使用两个旋转矩阵之间的测地角：

$$
e_R=\cos^{-1}\left(
\frac{\operatorname{tr}(R^T\hat R)-1}{2}
\right)
$$

### 3.1 直臂结果

当前直臂、18 mm TCP 的运行结果为：

| 单通道角度误差边界 | 裸臂位置误差 | TCP 位置误差 | 姿态误差 |
| ---: | ---: | ---: | ---: |
| ±2.000° | 9.837 mm | 12.495 mm | 8.485° |
| ±1.000° | 4.921 mm | 6.251 mm | 4.243° |
| ±0.500° | 2.461 mm | 3.126 mm | 2.121° |
| ±0.250° | 1.230 mm | 1.563 mm | 1.061° |
| ±0.100° | 0.492 mm | 0.625 mm | 0.424° |
| ±0.050° | 0.246 mm | 0.313 mm | 0.212° |

该表是直臂姿态的64角点结果，不能代替工作空间统计。

## 4. 10,000姿态工作空间分析

推荐运行命令：

```powershell
python scripts/cal_accuracy_workspace.py `
  --samples 10000 `
  --accuracy-deg 0.5 0.25 0.1 0.05
```

默认随机种子为 `20260812`，因此结果可复现。默认误差模式为：

```text
--error-model corners
```

它对每个真实姿态枚举64个误差角点，再记录该姿态的最大位置和姿态误差。因此：

- 表中的“平均误差”是10,000个“逐姿态角点最大误差”的平均值；
- 表中的“最大误差”是全部10,000姿态、全部64误差角点中的最大值；
- 该模式比每姿态随机抽取一次编码器误差更适合选型和边界评估。

如果需要概率型 Monte Carlo 结果，可使用：

```powershell
python scripts/cal_accuracy_workspace.py `
  --samples 10000 `
  --accuracy-deg 0.1 `
  --error-model uniform
```

`uniform` 模式在每个姿态为每个通道独立抽取一次 $[-a,a]$ 均匀误差，应在报告中明确注明误差分布和随机种子。

### 4.1 当前结果

使用固定随机种子、10,000个项目可行姿态、六维总弯曲角范围±30°、九肌腱行程筛选和64误差角点，当前18 mm TCP结果为：

| 编码器绝对精度边界 | 平均 TCP 位置误差 | 95% 分位 | 最大 TCP 位置误差 | 采样点全部小于5 mm |
| ---: | ---: | ---: | ---: | :---: |
| ±0.50° | 2.996 mm | 3.089 mm | 3.123 mm | 是 |
| ±0.25° | 1.497 mm | 1.544 mm | 1.561 mm | 是 |
| ±0.10° | 0.599 mm | 0.618 mm | 0.625 mm | 是 |
| ±0.05° | 0.299 mm | 0.309 mm | 0.312 mm | 是 |

在这套明确限定的分析条件内，`±0.1°` 六通道综合角度误差对应的 TCP 最大位置误差为 `0.625 mm`，10,000个采样姿态全部低于5 mm。

适合论文的准确表述为：

> 在当前三段离散铰链理想模型、项目配置的可行姿态范围、固定随机种子采样的10,000个姿态和六通道64角点误差模型下，±0.1°角度测量误差产生的18 mm工具TCP最大位置误差为0.625 mm，所有采样点均小于5 mm。

不建议把随机采样结果直接写成对连续无限工作空间和真实整机的数学严格保证。

### 4.2 输出文件

结果写入：

```text
output/accuracy_workspace/
  workspace_accuracy_summary.csv
  workspace_accuracy_samples.npz
  workspace_accuracy_heatmap.png
```

- `workspace_accuracy_summary.csv`：保存样本数、平均值、95%分位、最大值、5 mm通过率、误差模式和随机种子。
- `workspace_accuracy_samples.npz`：保存全部真实角度、TCP位置及逐姿态误差，可用于论文重新制图和复核最大误差姿态。
- `workspace_accuracy_heatmap.png`：将TCP工作空间投影到 `x-y` 平面，并显示各分箱中的平均角点误差上界。

## 5. 参数说明

### `cal_accuracy.py`

```text
--theta-deg       六个真实总弯曲角，单位deg
--accuracy-deg    一个或多个单通道误差边界，单位deg
--arm-config      执行臂配置
--tool-config     工具配置
--scenario-config 场景及运动学模式配置
```

### `cal_accuracy_workspace.py`

```text
--samples         随机有效姿态数量，默认10000
--accuracy-deg    一个或多个候选精度
--seed            随机种子，默认20260812
--error-model     corners或uniform，默认corners
--threshold-mm    通过阈值，默认5 mm
--output-dir      CSV、NPZ和热力图输出目录
--arm-config      执行臂配置
--tool-config     工具配置
--scenario-config 场景及运动学模式配置
```

修改机器人段长、柔性长度、肌腱路径、肌腱行程、铰链限制或工具TCP后，只需重新运行脚本；分析参数会随项目配置自动更新。

## 6. 结果边界和实验要求

当前脚本只包含理想运动学中的角度测量误差传播，没有包含：

- 编码器零位、非线性、温漂、安装轴偏斜和采样延迟；
- 段内弯曲分布不均、轴向伸长、扭转和剪切；
- 肌腱摩擦、回差、迟滞和负载变形；
- 六维力传感器、工具球及连接结构的弹性变形；
- 基座、安装坐标、世界坐标和发动机坐标标定误差；
- 工具TCP标定残差。

每段两个编码器能够恢复当前六维状态还依赖以下前提：

1. 编码器测得的是每段两个方向的总相对弯曲角；
2. 同方向两个离散铰链符合模型中的角度分配；
3. 轴向伸长和局部扭转可忽略或由其他传感器估计。

因此，真实系统应使用外部高精度位姿测量设备完成：

1. 编码器单体精度、重复性和温漂标定；
2. 单段总弯曲角与编码器读数的映射标定；
3. 三段工作空间静态验证；
4. 工具负载、重力方向和接触工况验证；
5. 整机误差预算和5 mm指标验证。

如果需要对连续工作空间给出数学保证，应在当前随机扫描之外增加全局优化、区间方法或带误差界的自适应空间细分。

## 7. 验证

精度脚本回归测试：

```powershell
python -m pytest tests/test_encoder_accuracy_scripts.py
```

测试内容包括：

- 从项目配置加载 `discrete_hinge`、±30°总角范围和18 mm TCP；
- 批量工作空间运动学与项目逐姿态正运动学一致；
- 随机姿态满足六角度范围和九肌腱行程；
- 不同编码器精度生成维度正确、单调合理的逐姿态角点误差。
