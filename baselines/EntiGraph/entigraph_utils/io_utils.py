import os
import io
import json
from typing import Dict

def set_openai_key():
    with open('data/dataset/openai.key', 'r') as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

def set_cohere_private_key():
    with open('data/dataset/cohere.key', 'r') as f:
        os.environ["COHERE_API_KEY"] = f.read().strip()

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f: str, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload_list(f, mode="r"):
    """Load multiple JSON objects from a file."""
    objects = []
    with open(f, mode) as file:
        for line in file:
            obj = json.loads(line)
            objects.append(obj)
    return objects

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def round_dict_values(input_dict: Dict, decimal_places: int = 4):
    def round_helper(value):
        if isinstance(value, dict):
            return {k: round_helper(v) for k, v in value.items()}
        elif isinstance(value, float):
            return round(value, decimal_places)
        else:
            return value
    return round_helper(input_dict)

def flatten_dict(nested_dict, parent_key='', sep='_'):
    flat_dict = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, float):
            flat_dict[new_key] = f"{value:.4f}"
        else:
            flat_dict[new_key] = str(value)
    return flat_dict
