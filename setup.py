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

from setuptools.command import build_ext
from setuptools import setup, find_packages, Extension
import os
import sys

class OccaInstaller(build_ext.build_ext):
    '''Compile occa.git'''

    def sys_call(self, command):
        self.spawn(command.split(' '))

    def pre_build(self):
        # Build occa
        if 'NO_CLEAN' not in os.environ:
            self.sys_call('make -C ./occa/c/occa.git clean')
        self.sys_call('make -C ./occa/c/occa.git -j4')

    def post_build(self):
        if sys.platform != 'darwin':
            return

        # Manually set relative rpath in OSX
        for output in self.get_outputs():
            self.sys_call('install_name_tool'
                          ' -change'
                          ' {libocca_so}'
                          ' @loader_path/occa.git/lib/libocca.so'
                          ' {output}'.format(libocca_so=self.libocca_so,
                                             output=output))

    def finalize_options(self):
        # Use numpy after setup_requirements installs it
        build_ext.build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

    def run(self):
        self.pre_build()
        build_ext.build_ext.run(self)
        self.post_build()


def get_ext_module(module):
    return Extension(
        name='occa.c.{module}'.format(module=module),
        sources=['occa/c/{module}.cpp'.format(module=module)],
        include_dirs=[
            'occa/c',
            'occa/c/occa.git/include',
        ],
        libraries=['occa'],
        library_dirs=['./occa/c/occa.git/lib'],
        extra_compile_args=['-Wno-unused-function'],
        extra_link_args=['-Wl,-rpath,$ORIGIN/occa.git/lib'],
    )


ext_modules = [
    get_ext_module(module)
    for module in ['base',
                   'device', 'kernel', 'memory',
                   'stream', 'streamtag']
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
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: Implementation :: CPython',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development',
]

package_data = {
    'occa.c': ['occa.git'],
}


setup(
    name='occa',
    version='0.3.4',
    description='Portable Approach for Parallel Architectures',
    long_description=long_description,
    keywords=keywords,
    classifiers=classifiers,
    url='https://libocca.org',
    author='David Medina',
    license='MIT',
    py_modules=['occa'],
    cmdclass={
        'build_ext': OccaInstaller,
    },
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    ext_modules=ext_modules,
    setup_requires=[
        'numpy>=1.7',
        'setuptools>=28.0.0',
    ],
    install_requires=[
        'numpy>=1.7',
    ],
    zip_safe=False,
)
