# glasz

[![Actions Status][actions-badge]][actions-link]
[![codecov](https://codecov.io/gh/James11222/glasz/graph/badge.svg?token=wPP0VytjPl)](https://codecov.io/gh/James11222/glasz)
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/James11222/glasz/workflows/CI/badge.svg
[actions-link]:             https://github.com/James11222/glasz/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/glasz
[conda-link]:               https://github.com/conda-forge/glasz-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/James11222/glasz/discussions
[pypi-link]:                https://pypi.org/project/glasz/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/glasz
[pypi-version]:             https://img.shields.io/pypi/v/glasz
[rtd-badge]:                https://readthedocs.org/projects/glasz/badge/?version=latest
[rtd-link]:                 https://glasz.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->


# Installation

This package is installable via `pip`. All one has to do is run

```
$ pip install glasz
```

to install the package into your current environment of choice.

## Development Version

This package uses `pixi` as the default task runner and environment manager. It
is significantly faster than any other competitor on the market (conda,
miniconda, mambda, etc...). To install [`pixi`](https://pixi.sh/latest/), click
on the link and install on your machine. It is very light weight and shouldn't
take long. From there, in this repository run the `pixi install` command and you
will gain a `pixi` environment with this package installed including all of its
dependencies. If you are not familiar with `pixi`, you can use the precomputed
`pixi.lock` file to resolve all dependencies in the creation or modification of
a `conda` environment. You can do this on your machine by running

```
$ pixi project export conda-explicit-spec conda_env_files --ignore-pypi-errors
```

which will create a directory called `conda_env_files` loaded with 3
`conda_spec.txt` files (similar to a `.yaml` but these have all the dependencies
precomputed and locked). Having the dependency conflicts precomputed is
important as `pyccl` is a monstrous library which takes a very long time to
install via `conda`. Take a look at the files, each one corresponds to a
different operating system: linux, mac-OS intel, mac-OS arm (M-chips). Choose
the file which resembles your machine and run the command

```
$ conda create --name ENV_NAME --file conda_env_files/default_{YOUR_SYSTEM}_conda_spec.txt
```

to create a new `conda` environment titled `ENV_NAME`. This environment is a
standard `conda` environment with a development version of `glasz` installed
alongside the dependencies. We provide a `conda_env_files` directory with
precomputed `conda_spec.txt` files from the current `pixi.lock` file
corresponding to the version on the main branch.
