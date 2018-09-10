<p align="center">
  <a href="https://libocca.org">
    <img alt="occa" src="https://libocca.org/assets/images/logo/blue.svg" width=250>
  </a>
</p>
&nbsp;
<p align="center">
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
python setup.py install
```

Test deployment before uploading package to pypi

```bash
python setup.py install sdist
pip install dist/occa-<version>.tar.gz
```

To avoid doing a `make clean` each time, use the `NO_CLEAN` environment variable

```bash
NO_CLEAN python setup.py install sdist
```

### Deployment

Upload to pypi

```
twine upload dist/*
```
