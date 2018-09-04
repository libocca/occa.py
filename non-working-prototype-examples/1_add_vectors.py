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
raise Exception('WARNING: Still not functional, only showcasing how the API is intended to be used soon')

import argparse
import numpy as np
import occa


def main(args):
    # Setup host data
    a  = np.array([1,2,3,4,5], dtype=np.int32)
    b  = 1 - a

    # Start with 0
    # After adding a + b, ab should equal to 1
    ab = 0 * a

    # Device setup with string flags
    device = occa.Device(args.device)

    # Use Strings:
    #
    #   device = occa.Device("mode: 'Serial'")
    #
    # More Backends:
    #
    #   device = occa.Device(mode='OpenMP',
    #                        schedule='compact',
    #                        chunk=10)
    #
    #   device = occa.Device(mode='OpenCL',
    #                        platform_id=0,
    #                        device_id=0)
    #
    #   device = occa.Device(mode='CUDA',
    #                        device_id=0)

    # Allocate memory on the device
    o_a = device.malloc(a)
    o_b = device.malloc(b)

    # Allocate only from the bytes
    o_ab = device.malloc(occa.bytes(ab))

    # Compile the kernel at run-time
    add_vectors = device.build_kernel('addVectors.okl', 'addVectors')

    # Pass run-time defines
    #   add_vectors = device.build_kernel('addVectors.okl', 'addVectors', {
    #       'defines': {
    #           'TYPE': 'double',
    #       },
    #   })

    # Copy memory to the device
    o_a = a
    o_b = b

    # Launch device kernel
    add_vectors(entries, o_a, o_b, o_ab);

    # Copy result to the host
    ab = o_ab.to_numpy();

    # Assert values
    print(ab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Example adding two vectors'
    )
    parser.add_argument('-d', '--device',
                        type=str
                        help='''Device properties (default: "mode: 'Serial'")''')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='''"Compile kernels in verbose mode"''')

    args = parser.parse_args()

    occa.settings.kernel.verbose = args.verbose

    return main(args)
