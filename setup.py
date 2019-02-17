#!/usr/bin/env python3
#
# The MIT License (MIT)
#
# Copyright (c) 2018 David Medina
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#

from setuptools import setup, find_packages, Command, Extension
from setuptools.command.build_ext import build_ext as setup_build_ext
import os
import sys
import versioneer


#---[ Commands ]------------------------
class build_ext(setup_build_ext):
    '''Compile occa.git'''

    user_options = setup_build_ext.user_options + [
        ('no-clean', 'n', "Don't rebuild OCCA"),
    ]

    def sys_call(self, command):
        self.spawn(command.split(' '))

    def replace_build_file(self, path):
        local_path = os.path.abspath(path)
        build_path = os.path.abspath(os.path.join(self.build_lib, path))

        os.makedirs(os.path.dirname(build_path), exist_ok=True)
        self.copy_file(local_path, build_path)

    def initialize_options(self):
        setup_build_ext.initialize_options(self)
        self.no_clean = False

    def finalize_options(self):
        # Use numpy after setup_requirements installs it
        setup_build_ext.finalize_options(self)

        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

    def pre_build(self):
        # Build occa
        if not self.no_clean:
            self.sys_call('make -C ./occa/c/occa.git clean')

        self.sys_call('make -C ./occa/c/occa.git -j4')

    def post_build(self):
        # Copy libocca.so manually, not sure why it isn't copied over
        #   even though bin/occa is...
        for path in ['./occa/c/occa.git/lib/libocca.so',
                     './occa/c/occa.git/bin/occa',
                     './occa/c/occa.git/include/occa/defines/compiledDefines.hpp']:
            self.replace_build_file(path)

        if sys.platform != 'darwin':
            return

        # Manually set relative rpath in OSX
        libocca_so = os.path.abspath('./occa/c/occa.git/lib/libocca.so')
        for output in self.get_outputs():
            self.sys_call('install_name_tool'
                          ' -change'
                          ' {libocca_so}'
                          ' @loader_path/occa.git/lib/libocca.so'
                          ' {output}'.format(libocca_so=libocca_so,
                                             output=output))

    def run(self):
        self.pre_build()
        setup_build_ext.run(self)
        self.post_build()


class coverage(Command):
    '''Run coverage'''

    description = 'Run coverage'
    user_options = [
        ('cov-report=', 'r', 'Coverage report type (default: term)'),
    ]

    def initialize_options(self):
        self.cov_report = 'term'

    def finalize_options(self):
        options = ['html', 'xml', 'annotate', 'term']
        if self.cov_report not in options:
            raise ValueError('--cov-report not in {}'.format(options))

    def run(self):
        import pytest
        pytest.main(['--cov-report', self.cov_report, '--cov=occa', 'tests/'])
#=======================================


if sys.version_info < (3, 6):
    sys.exit('Only Python 3.6 and above is supported')


def get_ext_module(module):
    return Extension(
        name='occa.c.{module}'.format(module=module),
        sources=['occa/c/{module}.cpp'.format(module=module)],
        include_dirs=[
            'occa/c',
            'occa/c/occa.git/include',
        ],
        depends=['./occa/c/occa.git/lib/libocca.so'],
        libraries=['occa'],
        library_dirs=['./occa/c/occa.git/lib'],
        extra_compile_args=['-Wno-unused-function'],
        extra_link_args=['-Wl,-rpath,$ORIGIN/occa.git/lib'],
    )


ext_modules = [
    get_ext_module(module)
    for module in ['base',
                   'device', 'kernel', 'memory',
                   'stream', 'streamtag',
                   'dtype']
]


long_description = ('''
In a nutshell, OCCA (like oca-rina) is an open-source library which aims to:

- Make it easy to program different types of devices (e.g. CPU, GPU, FPGA)

- Provide a unified API for interacting with backend device APIs (e.g. OpenMP, CUDA, OpenCL)

- Use just-in-time compilation to build backend kernels

- Provide a kernel language, a minor extension to C, to abstract programming for each backend
''')


keywords = ', '.join([
    'occa', 'hpc', 'gpu', 'jit',
    'openmp', 'opencl', 'cuda'
])


classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: MacOS',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: Implementation :: CPython',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development',
]


setup(
    name='occa',
    version=versioneer.get_version(),
    description='Portable Approach for Parallel Architectures',
    long_description=long_description,
    keywords=keywords,
    classifiers=classifiers,
    url='https://libocca.org',
    author='David Medina',
    license='MIT',
    py_modules=['occa'],
    cmdclass={
        'build_ext': build_ext,
        'coverage': coverage,
    },
    packages=find_packages(),
    include_package_data=True,
    ext_modules=ext_modules,
    setup_requires=[
        'numpy>=1.7',
        'setuptools>=28.0.0',
        'flake8',
        'pytest',
    ],
    install_requires=[
        'numpy>=1.7',
    ],
    zip_safe=False,
)
