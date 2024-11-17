## EIPL

EIPL (Embodied Intelligence with Deep Predictive Learning) is a library for robot motion generation using deep predictive learning developed at [Ogata Laboratory](https://ogata-lab.jp/), [Waseda University](https://www.waseda.jp/top/en).
Highlighted features include:

- [**Full documentation**](https://ogata-lab.github.io/eipl-docs) for the systematic understanding of deep predictive learning
- **Easy model training:** Includes sample datasets, source code, and pre-trained weights
- **Applicable to real robots:** Generalized motion can be acquired with small data sets

Those are maintained under this [fork](https://github.com/yunkai1841/eipl).
- `eipl/tutorials/open_manipulator/sarnn`
- `eipl/tutorials/airec/sarnn`

## Install

```sh
https://github.com/yunkai1841/eipl.git
cd eipl

python3 -m venv .venv # virtualenv is recommended
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

ffmpeg is recommneded for test script
```sh
sudo apt install ffmpeg
```

## pre-commit

Linting and formatting are done by pre-commit.

```sh
pre-commit install
```

## Citation

```
@article{suzuki2023deep,
  author    = {Kanata Suzuki and Hiroshi Ito and Tatsuro Yamada and Kei Kase and Tetsuya Ogata},
  title     = {Deep Predictive Learning : Motion Learning Concept inspired by Cognitive Robotics}, 
  booktitle = {arXiv preprint arXiv:2306.14714},
  year      = {2023},
}
```
