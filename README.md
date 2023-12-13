
![](report/figures/report/preprocessing.pdf)

## 🤌 emg-signal-movement-clf
In recent years, machine learning-based approaches have demonstrated success across various fields,  including the medical domain. 
This project focuses on evaluating different classification models capable of predicting the type of movement executed by a patient based on surface electromyography (sEMG) signals recorded  from the patient's forearm. The dataset utilized for this project is the NinaPro dataset DB1 \cite{ninapro}, 
comprising sEMG signals recorded from 27 patients while performing various hand movements.

## ⚙️ Setup (Optional)

Below, we outlined instructtions on how to setup the project. This is optional, as the project can be run easily run within your own python environment. However, you might run into some errors due to different package versions etc.

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

You can check the [preprocessing pipeline](notebooks/preprocessing.ipynb) notebook to see how the data was preprocessed and the [classification pipeline](notebooks/classification.ipynb) notebook to see how the classification was performed. Each notebook is self-contained and explains in detail how the results were obtained.

## 📑 Related Work

*State-of-the-art models*:
- [A Robust and Accurate Deep Learning based Pattern Recognition Framework for Upper Limb Prosthesis using sEMG](https://arxiv.org/pdf/2106.02463.pdf)

*NinaPro DB1 information:*
- [Characterization of a Benchmark Database for Myoelectric Movement Classification](https://ieeexplore.ieee.org/document/6825822?arnumber=6825822)
- [Electromyography data for non-invasive naturally-controlled robotic hand prostheses](https://www.nature.com/articles/sdata201453)
