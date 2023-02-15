from .library import user_context, user_lib
from enum import IntEnum
from legate.core import Rect
import legate.core.types as types

class HelloOpCode(IntEnum):
    HELLO_WORLD = user_lib.cffi.HELLO_WORLD_TASK

def print_hello(message: str):
   task = user_context.create_auto_task(
       HelloOpCode.HELLO_WORLD
   )
   task.add_scalar_arg(message, types.string)
   task.execute()

def print_hellos(message: str, n: int):
    launch_domain = Rect(lo=[0], hi=[n], exclusive=True)
    task = user_context.create_manual_task(
       HelloOpCode.HELLO_WORLD,
       launch_domain=launch_domain
    )
    task.add_scalar_arg(message, types.string)
    task.execute()
