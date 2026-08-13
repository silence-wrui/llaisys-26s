from ctypes import c_char_p


def load_error(lib):
    lib.llaisysGetLastError.argtypes = []
    lib.llaisysGetLastError.restype = c_char_p

    lib.llaisysClearLastError.argtypes = []
    lib.llaisysClearLastError.restype = None


def raise_last_error(lib):
    message = lib.llaisysGetLastError()

    if message is None:
        return

    lib.llaisysClearLastError()
    raise RuntimeError(message.decode("utf-8", errors="replace"))