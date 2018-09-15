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
import argparse
import numpy as np
import occa


def main(args):
    # Device setup with string flags
    device = occa.Device(args.device)

    # Alternatively, try out:
    #   device.setup(mode='Serial');
    #
    #   device.setup(mode='OpenMP',
    #                schedule='compact',
    #                chunk=10)
    #
    #   device.setup(mode='OpenCL',
    #                platform_id=0,
    #                device_id=0)
    #
    #   device.setup(mode='CUDA'
    #                device_id=0)

    # Allocate memory in host
    entries = 10

    a  = np.arange(entries, dtype=np.float32)
    b  = 1 - a
    ab = np.zeros(entries, dtype=np.float32)

    # Allocate memory in device and copy over data
    o_a  = device.malloc(a)
    o_b  = device.malloc(b)
    o_ab = device.malloc(entries, dtype=np.float32)

    # Build kernel
    add_vectors_source = r'''
    @kernel void addVectors(const int entries,
                            const float *a,
                            const float *b,
                            float *ab) {
      for (int i = 0; i < entries; ++i; @tile(16, @outer, @inner)) {
        ab[i] = a[i] + b[i];
      }
    }
    '''

    add_vectors = device.build_kernel_from_string(add_vectors_source,
                                                  'addVectors')

    # Or you can build from a file
    # add_vectors = d.build_kernel('addVectors.okl',
    #                              'addVectors')

    # Launch kernel
    add_vectors(np.intc(entries),
                o_a, o_b, o_ab)

    # Copy device data to host
    o_ab.copy_to(ab)

    # Print results
    print(ab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Example adding two vectors'
    )
    parser.add_argument('-d', '--device',
                        type=str,
                        default='',
                        help='''Device properties (default: "mode: 'Serial'")''')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='''"Compile kernels in verbose mode"''')

    args = parser.parse_args()

    occa.set_setting('kernel/verbose', args.verbose)

    main(args)
