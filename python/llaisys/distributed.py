from .libllaisys import LIB_LLAISYS


class Distributed:
    @staticmethod
    def init() -> None:
        LIB_LLAISYS.llaisysDistributedInit()

    @staticmethod
    def is_enabled() -> bool:
        return bool(LIB_LLAISYS.llaisysDistributedIsEnabled())

    @staticmethod
    def rank() -> int:
        return int(LIB_LLAISYS.llaisysDistributedRank())

    @staticmethod
    def world_size() -> int:
        return int(LIB_LLAISYS.llaisysDistributedWorldSize())

    @staticmethod
    def barrier() -> None:
        LIB_LLAISYS.llaisysDistributedBarrier()

    @staticmethod
    def finalize() -> None:
        LIB_LLAISYS.llaisysDistributedFinalize()
