# Stein Variational Belief Propagation for Multi-Robot Coordination

This repository is an application using [torch-bp](https://github.com/janapavlasek/torch-bp). The code contains Python scripts that run experiments for multi-robot collision tests in different environments using the belief propagation algorithms in `torch-bp`. This code accompanies the paper *Stein variational belief propagation for multi-robot coordination*. Please [cite us](#citation) if you use this code for your research.

Visit [*Project Webpage*](https://progress.eecs.umich.edu/projects/stein-bp/) for more details about the project.

## Installation

prerequisite:
* Python >=3.8 (tested with 3.8.18)
* PyTorch >=2.0 (tested with up to v2.2 with CUDA 12.1)
* [torch-bp](https://github.com/janapavlasek/torch-bp) (follow link to install)

After clone the repo, under `/multi-robot-svbp` run:
```bash
pip install -e .
```

Recommend installation:
```bash
# create folder for svbp project
mkdir svbp_workspace
cd svbp_workspace

# create python virtual environment
python3 -m venv svbp_venv
source svbp_venv/bin/activate
######### TODO: install suitable pytorch version #########
# for 12.1 eg. pip3 install torch torchvision torchaudio #

# clone the necessary repo
git clone https://github.com/janapavlasek/torch-bp.git
git clone https://github.com/janapavlasek/multi-robot-svbp.git

# install torch-bp
pip install -e torch-bp/
pip install -e multi-robot-svbp/

# may need additional package
pip install matplotlib
```

## Usage

The main experiment scripts are under `scripts` folder.

Basic use of the scripts:
```bash
# in the folder /multi-robo-svbp
# run stein mpc controller for point robot in simple scene
python scripts/point_robot_stein_mpc.py

# recommend to use cuda when running experiments with gabp or svbp
# use save flag to save images of the experiments during each step
python scripts/run_collision_avoid_gabp.py --cuda --save
python scripts/run_collision_avoid_svbp.py --cuda --save
```

All experiment outputs (data, visualization) will be sotred under `ouput` folder.

To use a different scene/map of the experiment, use the `--scene` flag and specify
a `.yml` file in `data/scenes` folder. For example,
```bash
# run the experiment using gabp in the map 'swap_rooms'
python scripts/run_collision_avoid_gabp.py --cuda --save --scene multi-robot-svbp/data/scenes/swap_rooms.yml
```

## Citation

This code accompanies the paper *Stein variational belief propagation for multi-robot coordination* (Robotics and Automation Letters, 2024). If you use it in your research, please cite:
```bibtex
@inproceedings{pavlasek2024stein,
  title={Stein Variational Belief Propagation for Multi-Robot Coordination},
  author={Pavlasek, Jana and Mah, Joshua Jing Zhi and Xu, Ruihan and Jenkins, Odest Chadwicke and Ramos, Fabio},
  booktitle={Robotics and Automation Letters (RA-L)},
  year={2024}
}
```
