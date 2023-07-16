import sys
import inspect
from typing import Any, List, Dict, Tuple, Union, Optional


def count_rocky() -> int:
    """
    Function to count the number of rocky objects in the namespace. This is
    useful for debugging purposes.
    """
    # get the list of objects in the namespace
    objects = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    # filter out the rocky objects
    rocky_objects = [obj for obj in objects if obj[1].__module__ == __name__]
    # return the number of rocky objects
    return len(rocky_objects)


def count_obj(obj: str = "rocky") -> int:
    """
    Function to count the number of `obj` objects in the namespace. This is
    a more general version of `count_rocky()`.
    """
    # get the list of objects in the namespace
    objects = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    # filter out the rocky objects
    obj_objects = [obj for obj in objects if obj[1].__module__ == __name__]
    # return the number of rocky objects
    return len(obj_objects)
