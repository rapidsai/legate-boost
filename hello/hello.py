import struct
from enum import IntEnum
from typing import Any

import cunumeric as np

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


class HelloOpCode(IntEnum):
    HELLO_WORLD = user_lib.cffi.HELLO_WORLD
    SUM = user_lib.cffi.SUM
    SQUARE = user_lib.cffi.SQUARE
    IOTA = user_lib.cffi.IOTA
    QUANTILE = user_lib.cffi.QUANTILE
    QUANTILE_REDUCE = user_lib.cffi.QUANTILE_REDUCE
    QUANTILE_OUTPUT= user_lib.cffi.QUANTILE_OUTPUT
    QUANTISE_DATA = user_lib.cffi.QUANTISE_DATA


def print_hello(message: str) -> None:
    """Create a Legate to task to print a message

    Args:
        message (str): The message to print
    """
    task = user_context.create_auto_task(HelloOpCode.HELLO_WORLD)
    task.add_scalar_arg(message, types.string)
    task.execute()


def print_hellos(message: str, n: int) -> None:
    """Create a Legate to task launch to print a message
       in n replicas of the task

    Args:
        message (str): The message to print
        n (int): The number of times to print
    """
    launch_domain = Rect(lo=[0], hi=[n], exclusive=True)
    task = user_context.create_manual_task(
        HelloOpCode.HELLO_WORLD, launch_domain=launch_domain
    )
    task.add_scalar_arg(message, types.string)
    task.execute()


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


def to_scalar(input: Store) -> float:
    """Extracts a Python scalar value from a Legate store
       encapsulating a single scalar

    Args:
        input (Store): The Legate store encapsulating a scalar

    Returns:
        float: A Python scalar
    """
    buf = input.storage.get_buffer(np.float32().itemsize)
    result = np.frombuffer(buf, dtype=np.float32, count=1)
    return float(result[0])

def to_array(input: Store) -> float:
    buf = input.storage.get_buffer(np.float32().itemsize)
    result = np.frombuffer(buf, dtype=np.float32, count=input.size)
    return result

def zero() -> Store:
    """Create a Legates store representing a single zero scalar

    Returns:
        Store: A Legate store representing a scalar zero
    """
    data = bytearray(4)
    buf = struct.pack(f"{len(data)}s", data)
    future = get_legate_runtime().create_future(buf, len(buf))
    return user_context.create_store(
        types.float32,
        shape=(1,),
        storage=future,
        optimize_scalar=True,
    )


def iota(size: int) -> Store:
    """Enqueues a task that will generate a 1-D array
       1,2,...size.

    Args:
        size (int): The number of elements to generate

    Returns:
        Store: The Legate store that will hold the iota values
    """
    output = user_context.create_store(
        types.float32,
        shape=(size,),
        optimize_scalar=True,
    )
    task = user_context.create_auto_task(
        HelloOpCode.IOTA,
    )
    task.add_output(output)
    task.execute()
    return output


def sum(input: Any) -> Store:
    """Sums a 1-D array into a single scalar

    Args:
        input (Any): A Legate store or any object implementing
                     the Legate data interface.

    Returns:
        Store: A Legate store encapsulating the array sum
    """
    input = _get_legate_store(input)

    task = user_context.create_auto_task(HelloOpCode.SUM)

    # zero-initialize the output for the summation
    output = zero()

    task.add_input(input)
    task.add_reduction(output, types.ReductionOp.ADD)
    task.execute()
    return output


def square(input: Any) -> Store:
    """Computes the elementwise square of a 1-D array

    Args:
        input (Any): A Legate store or any object implementing
                     the Legate data interface.

    Returns:
        Store: A Legate store encapsulating a 1-D array
               holding the elementwise square values
    """
    input = _get_legate_store(input)

    output = user_context.create_store(
        types.float32, shape=input.shape, optimize_scalar=True
    )
    task = user_context.create_auto_task(HelloOpCode.SQUARE)

    task.add_input(input)
    task.add_output(output)
    task.add_alignment(input, output)
    task.execute()

    return output

def quantise(input, n : int) -> np.ndarray:
    temp = user_context.create_store(
        types.float32, ndim=1)
    task = user_context.create_auto_task(
        HelloOpCode.QUANTILE,
    )
    task.add_input(_get_legate_store(input))
    task.add_output(temp)
    task.execute()

    temp = user_context.tree_reduce(
                 HelloOpCode.QUANTILE_REDUCE, temp
            )
    quantile_output = user_context.create_store(
        types.float32, ndim=1)
    task = user_context.create_auto_task(
        HelloOpCode.QUANTILE_OUTPUT,
    )
    task.add_scalar_arg(n, types.int64)
    task.add_input(temp)
    task.add_output(quantile_output)
    task.execute()
    quantised_output = user_context.create_store(
        types.uint16,
        shape=input.shape, optimize_scalar=True
    )
    task = user_context.create_auto_task(
        HelloOpCode.QUANTISE_DATA,
    )
    task.add_input(quantile_output)
    task.add_input(_get_legate_store(input))
    task.add_output(quantised_output)
    task.execute()

    return np.array(_Wrapper(quantile_output), copy=False), np.array(_Wrapper(quantised_output), copy=False)