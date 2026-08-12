# 发动机资产

本目录保存 `configs/scenes/engine_scene.yaml` 使用的发动机 CAD、可视网格和碰撞网格。

## 目录

```text
assets/engine/
  raw/
    engine.step
    engine_for_sim.sldprt
  meshes/
    engine_visual.stl
  collision/
    engine_collision.stl
```

- `raw/`：CAD 源文件，不由 MuJoCo 直接加载。
- `meshes/engine_visual.stl`：发动机渲染网格。
- `collision/engine_collision.stl`：发动机碰撞资产。

## 场景配置

`configs/scenes/engine_scene.yaml` 通过以下字段引用资产：

```yaml
engine:
  assets:
    visual_mesh: ../../assets/engine/meshes/engine_visual.stl
    collision_mesh: ../../assets/engine/collision/engine_collision.stl
  scale: 0.001
```

当前 STL 以毫米为单位，`scale: 0.001` 将其转换为米。`engine.pose` 设置网格在 MuJoCo 世界中的位置和姿态。

## 灰银色材质

发动机外观由同一场景配置中的 `preview_visualization` 定义：

```yaml
preview_visualization:
  visual_mesh_rgba: [0.66, 0.68, 0.71, 1.0]
  visual_material:
    name: engine_silver
    emission: 0.0
    specular: 0.72
    shininess: 0.48
```

场景构建器把 `engine_silver` 绑定到可视网格，并设置中性主光、补光和 headlight。MuJoCo viewer 与 observer 相机使用相同材质和灯光。

## 网格要求

- MuJoCo 加载 STL/OBJ/MSH 网格，不直接加载 STEP、SLDPRT 或 SLDASM。
- 可视网格负责外观，碰撞控制优先使用场景中的简化 primitive 几何。
- STL 单个 mesh 的面数应保持在 MuJoCo 可加载范围内；项目运行时会把过大网格无损拆成多个顺序分片并全部注入，既满足单 mesh 面数限制，也不会抽样丢弃三角面。
- 更新资产路径、单位或坐标原点时，应同步更新 `configs/scenes/engine_scene.yaml` 的路径、`scale`、`pose` 和局部标注。
