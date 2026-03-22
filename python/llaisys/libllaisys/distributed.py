from ctypes import c_int


def load_distributed(lib):
    lib.llaisysDistributedInit.argtypes = []
    lib.llaisysDistributedInit.restype = c_int

    lib.llaisysDistributedIsEnabled.argtypes = []
    lib.llaisysDistributedIsEnabled.restype = c_int

    lib.llaisysDistributedRank.argtypes = []
    lib.llaisysDistributedRank.restype = c_int

    lib.llaisysDistributedWorldSize.argtypes = []
    lib.llaisysDistributedWorldSize.restype = c_int

    lib.llaisysDistributedBarrier.argtypes = []
    lib.llaisysDistributedBarrier.restype = None

    lib.llaisysDistributedFinalize.argtypes = []
    lib.llaisysDistributedFinalize.restype = None
