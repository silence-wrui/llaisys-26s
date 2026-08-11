import llaisys
import torch
from test_utils import *
import argparse
import ctypes

def test_basic_runtime_api(device_name: str = "cpu"):

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    ndev = api.get_device_count()
    print(f"Found {ndev} {device_name} devices")
    if ndev == 0:
        print("     Skipped")
        return

    for i in range(ndev):
        print(f"Testing device {i}...")
        api.set_device(i)

        test_memcpy(api, 1024 * 1024)
        test_memcpy_async(api, 1024 * 1024)
        api.device_synchronize()

        print("     Passed")


def test_memcpy(api, size_bytes: int):
    a = torch.zeros((size_bytes,), dtype=torch.uint8, device=torch_device("cpu"))
    b = torch.ones_like(a)

    device_a = api.malloc_device(size_bytes)
    device_b = api.malloc_device(size_bytes)

    try:
        # a -> device_a
        api.memcpy_sync(
            device_a,
            a.data_ptr(),
            size_bytes,
            llaisys.MemcpyKind.H2D,
        )

        # device_a -> device_b
        api.memcpy_sync(
            device_b,
            device_a,
            size_bytes,
            llaisys.MemcpyKind.D2D,
        )

        # device_b -> b
        api.memcpy_sync(
            b.data_ptr(),
            device_b,
            size_bytes,
            llaisys.MemcpyKind.D2H,
        )

        torch.testing.assert_close(a, b)
    finally:
        api.free_device(device_b)
        api.free_device(device_a)

def test_memcpy_async(api, size_bytes: int):
    stream = api.create_stream()

    host_a = None
    host_b = None
    device_a = None
    device_b = None

    try:
        host_a = api.malloc_host(size_bytes)
        host_b = api.malloc_host(size_bytes)
        device_a = api.malloc_device(size_bytes)
        device_b = api.malloc_device(size_bytes)

        pattern = (
            bytes(range(256)) * ((size_bytes + 255) // 256)
        )[:size_bytes]

        ctypes.memmove(host_a, pattern, size_bytes)
        ctypes.memset(host_b, 0, size_bytes)

        # host_a -> device_a
        api.memcpy_async(
            device_a,
            host_a,
            size_bytes,
            llaisys.MemcpyKind.H2D,
            stream,
        )

        # device_a -> device_b
        api.memcpy_async(
            device_b,
            device_a,
            size_bytes,
            llaisys.MemcpyKind.D2D,
            stream,
        )

        # device_b -> host_b
        api.memcpy_async(
            host_b,
            device_b,
            size_bytes,
            llaisys.MemcpyKind.D2H,
            stream,
        )

        api.stream_synchronize(stream)

        result = ctypes.string_at(host_b, size_bytes)
        assert result == pattern

    finally:
        if device_b is not None:
            api.free_device(device_b)
        if device_a is not None:
            api.free_device(device_a)
        if host_b is not None:
            api.free_host(host_b)
        if host_a is not None:
            api.free_host(host_a)

        api.destroy_stream(stream)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()
    test_basic_runtime_api(args.device)
    
    print("\033[92mTest passed!\033[0m\n")
