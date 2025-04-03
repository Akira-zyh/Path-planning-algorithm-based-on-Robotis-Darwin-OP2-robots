# Path-planning-algorithm-based-on-Robotis-Darwin-OP2-robots


---

![Amusing robot](https://github.com/Akira-zyh/Path-planning-algorithm-based-on-Robotis-Darwin-OP2-robots/blob/main/amuse.gif)

---

## Environment setup
1. Install Python 3.10.x (recommend to install Python 3.10.x via [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main))
2. Install [Webot 2022b](https://github.com/cyberbotics/webots/releases/tag/R2022b) 
    - Why 2022b instead of latest version? Version 2022b has `managers.py` and `_managers.pyd`but these are not included in later version than 2022b. (By the way, if you use other programming language, you can use the latest version of Webot.)
3. Install PyTorch
    - If you have a NVIDIA GPU, you can install PyTorch with CUDA support.
    - If you don't have a NVIDIA GPU, you can install [DirectML for PyTorch](https://learn.microsoft.com/zh-cn/windows/ai/directml/pytorch-windows) with `Python 3.10` and `torch 2.4` (Really great for Windows 11 and WSL).

---
#TODO
