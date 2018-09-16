<p align="center">
  <a href="https://libocca.org">
    <img alt="occa" src="https://libocca.org/assets/images/logo/blue.svg" width=250>
  </a>
</p>
&nbsp;
<p align="center">
  <a href="https://mybinder.org/v2/gh/libocca/occa.py/0.3.9?filepath=notebooks%2FTutorial.ipynb"><img alt="Binder" src="https://mybinder.org/badge.svg"></a>
  <a href="https://travis-ci.org/libocca/occa.py"><img alt="Build Status" src="https://travis-ci.org/libocca/occa.py.svg?branch=master"></a>
  <a href="https://codecov.io/github/libocca/occa.py"><img alt="codecov.io" src="https://codecov.io/github/libocca/occa.py/coverage.svg"></a>
  <a href="https://gitter.im/libocca/occa?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge"><img alt="Gitter" src="https://badges.gitter.im/libocca/occa.svg"></a>
</p>

&nbsp;

### What is OCCA?

In a nutshell, OCCA (like *oca*-rina) is an open-source library which aims to

- Make it easy to program different types of devices (e.g. _CPU_, _GPU_, _FPGA_)
- Provide a [unified API](https://libocca.org/#/guide/occa/introduction) for interacting with backend device APIs (e.g. _OpenMP_, _CUDA_, _OpenCL_)
- Use just-in-time compilation to build backend kernels
- Provide a [kernel language](https://libocca.org/#/guide/okl/introduction), a minor extension to C, to abstract programming for each backend

&nbsp;

### Links

- [Documentation](https://libocca.org)
- **Want to contribute?** Checkout the ['beginner' issues](https://github.com/libocca/occa/labels/beginner)
- 🌟 Who is using OCCA?
  - [Gallery](https://libocca.org/#/gallery)
  - [Publications](https://libocca.org/#/publications)

&nbsp;

### Installing

```bash
pip install occa
```

### Development

Try out local installation

```bash
git submodule update --init
pip install -e .
```

Between updates, run

```bash
# To avoid doing a `make clean` each time, use the `--no-clean` flag
python setup.py build_ext --no-clean --inplace
```

### Deployment

Test deployment before uploading package to pypi

```bash
python setup.py install sdist
pip install dist/occa-<version>.tar.gz
```

After testing, upload to pypi by running

```
twine upload dist/*
```
