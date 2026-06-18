# 控制扩展说明

本次扩展保留原有运动学级控制链路，同时增加面向验收指标的可选控制与验证工具。

## 保留的基线

`controller.type: hybrid_force_position` 没有改变。它仍然是默认擦拭控制器，也是最快的回归检查入口。

## 新增实验模式

- `trajectory.type: dmp`：从示教任务空间轨迹学习形状，并迁移到新的起点和终点。
- `controller.type: navigation_cbf_qp`：在导航任务中使用 CBF 半空间投影维护中心线 clearance。
- `controller.type: dynamic_adaptive_impedance`：使用 9 维 PCC 降阶动力学模型进行擦拭力位混合控制实验。

## PCC 降阶动力学

降阶状态沿用现有 PCC 坐标向量：

```text
q = [kx_1, ky_1, eps_1, kx_2, ky_2, eps_2, kx_3, ky_3, eps_3]
```

当前实现采用尽量小的工程模型：

```text
M(q) qddot + D qdot + K q = tau + J_tip(q).T F_contact
```

参数来自 `configs/dynamics/pcc_reduced.yaml`。这些值目前是工程估计值，后续有实验数据后应替换为辨识参数。

## 验收脚本

`scripts/test_indicator_*.py` 会生成 Markdown 报告和曲线图，覆盖：

- 指标 3.1：末端稳态定位误差不超过 2 cm。
- 指标 3.3 扰动：最大位移偏差不超过 4 cm。
- 指标 3.3 力跟踪：接触力 RMSE 不超过 1 N。

推荐流程是先用 CLI 的 `--save-run` 生成 `output/runs/<run>/result.npz`，再用验收脚本读取该文件。若不提供 `--result-npz`，脚本会自行启动完整 MuJoCo 仿真；动态阻抗模式包含 PCC 质量矩阵计算，耗时会明显更长。

动力学级擦拭实验可以使用：

```powershell
python cli.py run-mujoco-wiping --config configs/main_config_dynamic_wiping.yaml --save-run
python scripts/test_indicator_3_3_force_tracking.py --result-npz output/runs/<run>/result.npz
```

扰动工况目前保持可配置，因为第三方尚未固定扰动力大小、方向和持续时间。
