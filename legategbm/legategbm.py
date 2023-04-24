import struct
from enum import IntEnum
from typing import Any

import cunumeric as cn

import legate.core.types as types
from legate.core import Rect, Store, get_legate_runtime, Array
import pyarrow as pa

from .library import user_context, user_lib

class _Wrapper:
    def __init__(self, store: Store) -> None:
        self._store = store

    @property
    def __legate_data_interface__(self) -> dict[str, Any]:
        """
        Constructs a Legate data interface object from a store wrapped in this
        object
        """
        dtype = self._store.type.type
        array = Array(dtype, [None, self._store])

        # Create a field metadata to populate the data field
        field = pa.field("Array", dtype, nullable=False)

        return {
            "version": 1,
            "data": {field: array},
        }


class LegateGBMOpCode(IntEnum):
    QUANTILE = user_lib.cffi.QUANTILE
    QUANTILE_REDUCE = user_lib.cffi.QUANTILE_REDUCE
    QUANTILE_OUTPUT= user_lib.cffi.QUANTILE_OUTPUT
    QUANTISE_DATA = user_lib.cffi.QUANTISE_DATA




def _get_legate_store(input: Any) -> Store:
    """Extracts a Legate store from any object
       implementing the legete data interface

    Args:
        input (Any): The input object

    Returns:
        Store: The extracted Legate store
    """
    if isinstance(input, Store):
        return input
    data = input.__legate_data_interface__["data"]
    field = next(iter(data))
    array = data[field]
    _, store = array.stores()
    return store


def quantise(input: cn.ndarray, n : int) -> cn.ndarray:
    assert input.ndim == 2
    temp = user_context.create_store(
        types.float32, ndim=1)
    task = user_context.create_auto_task(
        LegateGBMOpCode.QUANTILE,
    )
    task.add_input(_get_legate_store(input))
    task.add_scalar_arg(input.shape[1], types.int64)
    task.add_output(temp)
    task.execute()

    temp = user_context.tree_reduce(
                 LegateGBMOpCode.QUANTILE_REDUCE, temp
            )
    quantile_output = user_context.create_store(
        types.float32, ndim=1)
    ptr_output = user_context.create_store(
        types.uint64, shape=input.shape[1] + 1)
    task = user_context.create_auto_task(
        LegateGBMOpCode.QUANTILE_OUTPUT,
    )
    task.add_scalar_arg(n, types.int64)
    task.add_input(temp)
    task.add_broadcast(temp)
    task.add_output(quantile_output)
    task.add_broadcast(quantile_output)
    task.add_output(ptr_output)
    task.add_broadcast(ptr_output)
    task.execute()

    
    quantised_output = user_context.create_store(
        types.uint16,
        shape=input.shape, optimize_scalar=True
    )
    task = user_context.create_auto_task(
        LegateGBMOpCode.QUANTISE_DATA,
    )
    task.add_input(quantile_output)
    task.add_input(ptr_output)
    #task.add_broadcast(ptr_output)
    task.add_input(_get_legate_store(input))
    task.add_output(quantised_output)
    task.execute()

    return cn.array(_Wrapper(quantile_output), copy=False),cn.array(_Wrapper(ptr_output), copy=False),  cn.array(_Wrapper(quantised_output), copy=False)