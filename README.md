## 🤌 emg-signal-movement-clf
Hand movement classification from EMG signals across different subjects.

## ⚙️ Setup

To reproduce all results, this notebook should be run with the correct **Python version** inside the specified **virtual environment** to use all packages with the correct version.

We use [Poetry](https://python-poetry.org/) for package management and versioning. This project was developed for Python `3.10.13` and Poetry `1.7.1`. We recommend installing Python via [pyenv](https://github.com/pyenv/pyenv) and Poetry via [pipx](https://pypa.github.io/pipx/).

```bash
pyenv install 3.10.13
```

Then, install Poetry via `pipx` (if not installed yet):

```bash
pipx install poetry==1.7.1
```

The project has a `.python-version` in the root directory, which will automatically activate the correct Python version when you enter the project directory. You can check that the correct Python version is used by running `python --version` and similarly for poetry running `poetry --version`.

Next, you can install all dependencies via Poetry:

```bash
poetry install
```

Finally, run the following to add the virtual environment to Jupyter:

```bash
poetry run python -m ipykernel install --user --name=emg-signal-movement-clf
```

## 🔗 Reproduce

To reproduce all results, run the following command:

```bash
poetry run main.py
```

## 📑 Related Work

*State-of-the-art models*:
- [A Robust and Accurate Deep Learning based Pattern Recognition Framework for Upper Limb Prosthesis using sEMG](https://arxiv.org/pdf/2106.02463.pdf)

*NinaPro DB1 information:*
- [Characterization of a Benchmark Database for Myoelectric Movement Classification](https://ieeexplore.ieee.org/document/6825822?arnumber=6825822)
- [Electromyography data for non-invasive naturally-controlled robotic hand prostheses](https://www.nature.com/articles/sdata201453)
