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
    # Create device
    d = occa.Device(args.device)

    # Allocate memory in host
    entries = 10

    a  = np.arange(entries, dtype=np.float32)
    b  = 1 - a
    ab = np.zeros(entries, dtype=np.float32)

    # Allocate memory in device and copy over data
    o_a  = d.malloc(src=a)
    o_b  = d.malloc(src=b)
    o_ab = d.malloc(src=ab)

    # Build kernel
    add_vectors = r'''
    @kernel void addVectors(const int entries,
                            const float *a,
                            const float *b,
                            float *ab) {
      for (int i = 0; i < entries; ++i; @tile(16, @outer, @inner)) {
        ab[i] = a[i] + b[i];
      }
    }
    '''

    k = d.build_kernel_from_string(add_vectors,
                                   'addVectors')

    # Launch kernel
    k(np.intc(entries),
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

    # TODO: Missing properties object
    # occa.settings.kernel.verbose = args.verbose

    main(args)
