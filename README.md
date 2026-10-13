# RoboFactory · World Agents environments

This repository provides the [RoboFactory](https://github.com/MARS-EAI/RoboFactory) environments and camera configurations used by WorldDreamer v5, with ManiSkill as the simulation platform. The supported tasks are PlaceFood, CameraAlignment and TakePhoto.

## Installation

Use Linux, an NVIDIA GPU with Vulkan support, Python 3.10, and uv. Keep the simulator environment separate from the model environment.

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e .
uv pip check --python .venv/bin/python
```

The simulator dependencies are ManiSkill 3.0.0b12 and SAPIEN 3.0.0b1. Install the system graphics dependencies and download the scene assets using the upstream RoboFactory instructions. From this repository:

```bash
(
  cd robofactory
  ../.venv/bin/python script/download_assets.py
  ../.venv/bin/python -m zipfile -e assets/assets.zip assets
)
```

## Camera configurations

| Task | Robots | Configuration |
| --- | ---: | --- |
| PlaceFood | 2 | `robofactory/configs/worlddreamer/place_food.yaml` |
| CameraAlignment | 3 | `robofactory/configs/worlddreamer/camera_alignment.yaml` |
| TakePhoto | 4 | `robofactory/configs/worlddreamer/take_photo.yaml` |

Each task provides a fixed global view and one camera mounted to each robot's TCP. The wrist camera poses match the regenerated demonstration dataset. TakePhoto additionally uses a side-facing view on the third robot. RGB, depth, camera intrinsics, and extrinsics remain available through the simulator observations.

Use the configuration from `configs/worlddreamer/` when generating data or evaluating WorldDreamer.

## Regenerating demonstrations

`tools/quality_rollout.py` records synchronized RGB-D observations, camera parameters, TCP states and executed joint commands using the same teacher logic as the regenerated dataset. The three supported tasks are selected explicitly. Generated episodes include an environment configuration snapshot and validation results.

```bash
.venv/bin/python tools/quality_rollout.py \
  --task PlaceFood --seed 0 --output-root outputs/demonstrations
```

CameraAlignment and TakePhoto use a source demonstration. Supply its HDF5 trajectory with `--source-h5` and the dataset record's NumPy seed with `--legacy-numpy-seed`. The default disk reserve is 180 GiB; use `--min-free-gb` to configure it for your recording volume. Automated validation and visual review are separate fields in the generated validation record.

## Acknowledgements and citation

We thank the RoboFactory authors and contributors for the original benchmark, tasks and data collection framework, and the [ManiSkill](https://www.maniskill.ai/) contributors for the simulation platform. Our environment depends on the official ManiSkill 3.0.0b12 package.

- [RoboFactory source code](https://github.com/MARS-EAI/RoboFactory)
- [RoboFactory project page](https://iranqin.github.io/robofactory/)
- [RoboFactory paper](https://arxiv.org/abs/2503.16408)

If you use these adapted environments or demonstrations in research, please cite the original RoboFactory paper:

```bibtex
@article{qin2025robofactory,
  title={RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints},
  author={Qin, Yiran and Kang, Li and Song, Xiufeng and Yin, Zhenfei and Liu, Xiaohong and Liu, Xihui and Zhang, Ruimao and Bai, Lei},
  journal={arXiv preprint arXiv:2503.16408},
  year={2025}
}
```
