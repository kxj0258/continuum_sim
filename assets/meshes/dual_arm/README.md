# 双臂 SolidWorks STL 资产放置规范

本目录用于放置双连续体臂 MuJoCo 可视化 mesh。后续双臂 XML 生成脚本会按 `configs/robots/dual_arm_meshes.yaml` 中的路径加载这些文件。

## 放置原则

- STL 建议从 SolidWorks 以 `mm` 为单位导出；后续 MuJoCo 配置默认用 `mesh_scale: 0.001` 转成米。
- `assembly_preview.stl` 只用于整体预览、坐标检查和人工对齐，不用于逐 link 运动。
- 真正随关节运动的 mesh 必须按 body 拆分：统一基座、主臂臂基座、从臂臂基座、每段 4 个 link。
- 如果主臂和从臂的零件几何完全一致，可以先把两套 STL 复制成相同内容但不同文件名。这样后续代码可以稳定按 arm 名加载，避免在 XML 生成阶段做额外推断。
- 目前从臂初期设计为可控、可视化、不参与碰撞；因此这些 STL 都作为 visual mesh 使用，不作为 collision mesh。

## 目录结构

请按下面结构放置文件：

```text
assets/meshes/dual_arm/
  assembly_preview.stl

  shared_base/
    dual_base_visual.stl

  executor/
    base_visual.stl
    segment_1_link_1_visual.stl
    segment_1_link_2_visual.stl
    segment_1_link_3_visual.stl
    segment_1_link_4_visual.stl
    segment_2_link_1_visual.stl
    segment_2_link_2_visual.stl
    segment_2_link_3_visual.stl
    segment_2_link_4_visual.stl
    segment_3_link_1_visual.stl
    segment_3_link_2_visual.stl
    segment_3_link_3_visual.stl
    segment_3_link_4_visual.stl

  observer/
    base_visual.stl
    segment_1_link_1_visual.stl
    segment_1_link_2_visual.stl
    segment_1_link_3_visual.stl
    segment_1_link_4_visual.stl
    segment_2_link_1_visual.stl
    segment_2_link_2_visual.stl
    segment_2_link_3_visual.stl
    segment_2_link_4_visual.stl
    segment_3_link_1_visual.stl
    segment_3_link_2_visual.stl
    segment_3_link_3_visual.stl
    segment_3_link_4_visual.stl
```

## 命名含义

- `shared_base/dual_base_visual.stl`：两根臂固连的统一基座外观。
- `executor/base_visual.stl`：主臂自身臂基座外观。
- `observer/base_visual.stl`：从臂自身臂基座外观。
- `segment_<i>_link_<j>_visual.stl`：第 `i` 个连续体段、第 `j` 个离散 link 的外观；`i = 1..3`，`j = 1..4`。

## SolidWorks 导出建议

1. 先导出整体装配为 `assembly_preview.stl`。
2. 再按零件导出统一基座、两根臂的臂基座、以及 24 个 link STL。
3. 导出时尽量保持同一装配坐标系，不要让每个零件自动重置到自己的局部原点。
4. 如果只能导出局部坐标 STL，请记录每个零件相对装配坐标系的 pose，后续需要写入 mesh offset 配置。
5. 避免中文文件名和空格；使用上面的 ASCII 文件名。

## 后续加载约定

后续代码会优先读取：

```text
configs/robots/dual_arm_meshes.yaml
```

该 YAML 是机器可读资产清单。修改文件名或路径时，请同步修改该 YAML，而不是只改本 README。
