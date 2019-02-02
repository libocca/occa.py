from ..base import get_device


class Kernel:
    def __init__(self, source, name):
        self.source = source
        self.name = name

    def _build(self, device, props):
        return device.build_kernel_from_string(self.source,
                                               self.name,
                                               props)

    def __getitem__(self, device):
        return self._build(device)

    def __call__(self, *args, props=None):
        return self._build(get_device(), props)(*args)
