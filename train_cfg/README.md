# SPAD Pose Train Configs

这个目录用于管理 `ultralytics` 里的 SPAD pose 训练配置。

当前结构分成两条主线：

- `sequence/`: 旧的论文基线思路，保留多帧重建 + detector-side temporal plugin
- `frame/`: 新的创新思路，raw chunk 经过 preprocessor 后只输出单帧，再接 image-space adapter

## 目录结构

```text
train_cfg/
  README.md
  sequence/
    sequence_stea.yaml
    sequence_sum.yaml
    sequence_ppb.yaml
  frame/
    frame_stea.yaml
    frame_stea_ua.yaml
    frame_stea_residual.yaml
    frame_sum.yaml
    frame_ppb.yaml
```

## 启动方式

在 `ultralytics/` 根目录下运行。

### Sequence

```bash
python train_spad_pose_sequence.py --cfg train_cfg/sequence/sequence_stea.yaml
```

### Frame

```bash
python train_spad_pose_frame.py --cfg train_cfg/frame/frame_stea_ua.yaml
```

### 临时覆盖参数

不改 YAML，直接在命令行覆盖：

```bash
python train_spad_pose_frame.py --cfg train_cfg/frame/frame_stea_ua.yaml device=1 batch=2 epochs=50 spad_chunk_size=320
```

命令行覆盖格式统一为 `key=value`。

## 两条主线的区别

### Sequence

`sequence/*` 配置对应旧 baseline：

- dataset 按一个 raw window 生成多个 supervision frames
- 训练时仍保留 detector 侧的时序展开
- 常用参数：
  - `spad_output_frames`
  - `spad_subsampling`
  - `spad_stride_frames`
  - `spad_plugin`
  - `ssd_state_dim`
  - `ssd_head_divisor`

这条线主要用于和原论文中的 `SSD layer` 思路做对比。

### Frame

`frame/*` 配置对应新的创新路线：

- dataloader 仍提供 raw chunk
- preprocessor 只输出一个末帧语义的检测图像
- detector 输入是标准 `B,C,H,W`
- 第一版创新模块是接口层 UA Adapter

这条线最关键的参数是：

- `spad_chunk_size`
- `spad_stride_frames`
- `spad_preprocessor`
- `spad_frame_adapter`

## 每个配置文件的含义

### sequence/sequence_stea.yaml

旧 baseline，`STEA + temporal SSD plugin`。

适合：

- 复现旧思路
- 作为主对比基线

### sequence/sequence_sum.yaml

旧 baseline，`sum + temporal SSD plugin`。

适合：

- 看最简单预处理在旧 sequence 范式下的表现

### sequence/sequence_ppb.yaml

旧 baseline，`PPB + temporal SSD plugin`。

适合：

- 和 `sequence_stea` 比较不同 preprocessor 的影响

### frame/frame_stea_ua.yaml

新主线，`STEA + UA Adapter`。

当前最推荐先跑这一个。

特点：

- frame-mode
- uncertainty 使用 `w_mean`
- detector 冻结

### frame/frame_stea_residual.yaml

新消融线，`STEA + residual adapter`。

适合：

- 作为 `Frame` 路线最重要的第一组消融
- 验证增益到底来自接口层位置，还是来自 `w_mean` 条件化

特点：

- frame-mode
- 不使用 uncertainty 条件
- 只保留 image-space 残差 adapter

### frame/frame_sum.yaml

新对比线，`sum + UA Adapter`。

特点：

- frame-mode
- confidence 使用 Sobel 边缘图
- 用于验证最弱预处理时，adapter 是否仍然有帮助

### frame/frame_ppb.yaml

新对比线，`PPB + UA Adapter`。

特点：

- frame-mode
- confidence 使用 `1 - sample_weight`
- 用于验证 UA 是否只对 `STEA` 有效

## 需要优先修改的字段

当前配置默认已经指向：

- `model: /home/zvc/Project/SPADHand/ultralytics/weights/detector.pt`
- `spad_train_json: /home/zvc/Data/visionsim/outputs/train.json`
- `spad_test_json: /home/zvc/Data/visionsim/outputs/test.json`

第一次使用时，通常只需要再确认：

- `device`

常见还会改：

- `project`
- `name`
- `epochs`
- `batch`

## 推荐实验顺序

建议第一轮按下面顺序跑：

1. `sequence/sequence_stea.yaml`
2. `frame/frame_stea_residual.yaml`
3. `frame/frame_stea_ua.yaml`
4. `frame/frame_ppb.yaml`
5. `frame/frame_sum.yaml`

如果想看 sequence 侧不同预处理，再补：

6. `sequence/sequence_ppb.yaml`
7. `sequence/sequence_sum.yaml`

## 参数建议

### Frame 路线

第一轮建议固定：

- `spad_chunk_size: 320`
- `spad_bins_per_gt: 64`
- `spad_stride_frames: 1`
- `batch: 1`

这里 `320` 对应：

- `8000 FPS / 125 FPS = 64` raw bins per GT frame
- `320 / 64 = 5` GT-frame equivalents

所以它和 sequence 路线里的：

- `spad_output_frames: 5`
- `spad_subsampling: 64`

是严格对齐的。

后续可以再扫：

- `spad_chunk_size = 128`
- `spad_chunk_size = 320`
- `spad_chunk_size = 512`

### Sequence 路线

第一轮建议保留：

- `spad_output_frames: 5`
- `spad_subsampling: 64`
- `spad_stride_frames: 5`

避免一开始就同时改旧 baseline 的窗口定义。

## 命名约定

整个工程里统一遵守：

- 旧路径全部带 `Sequence`
- 新路径全部带 `Frame`

这样后续继续扩展：

- `Frame + Boundary`
- `Frame + FusionGate`

也会保持清晰。

## 后续扩展建议

如果后面继续加新实验，建议在 `frame/` 下继续新增：

- `frame_stea_boundary.yaml`
- `frame_stea_fusion.yaml`

以及在 `sequence/` 下继续保留对应旧 baseline 配置，方便实验表统一整理。
