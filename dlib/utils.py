from typing import Any, Hashable


def find_multiple(n: int, k: int) -> int:
    """From lit-gpt."""
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def wait_for_debugger(port: int = 5678):
    """
    Pauses the program until a remote debugger is attached. Should only be called on rank0.
    """

    import debugpy

    debugpy.listen(("0.0.0.0", port))
    print(
        f"Waiting for client to attach on port {port}... NOTE: if using docker, you need to forward the port with -p {port}:{port}."
    )
    debugpy.wait_for_client()


def format_elapsed_time(seconds: float):
    if seconds < 0.001:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        seconds %= 60
        return f"{minutes}:{int(seconds):02d}m"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        return f"{hours}:{minutes:02d}:{int(seconds):02d}h"
    else:
        days = int(seconds // 86400)
        seconds %= 86400
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        return f"{days}:{hours:02d}:{minutes:02d}:{int(seconds):02d}d"


def pretty_str_from_dict(data: dict, prefix: str = ""):
    """
    Utility function to print a dict of metric key-value pairs to the terminal.

    Returns a pretty string to print to the terminal. Uses some heuristics to prettify:
    - if `time` is in the key, we assume it's a elapsed time in seconds and format it accordingly
    - format all floats to 3 decimal places
    - if a key contains a `/`, we assume it's a path and only print the last part
    """
    print_str = prefix + " " if prefix else ""
    for k, v in data.items():
        if "time" in k and isinstance(v, float):
            v = format_elapsed_time(v)
        elif isinstance(v, float):
            v = f"{v:.3f}"

        if "/" in k:
            k = k.split("/")[-1]

        print_str += f"{k}: {v}, "
    return print_str[:-2]  # Remove trailing ", "


class ddict(dict):
    """Wrapper around the native dict class that allows access via dot syntax and JS-like behavior for KeyErrors."""

    def __getattr__(self, key: Hashable) -> Any:
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key: Hashable, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: Hashable) -> None:
        del self[key]

    def __dir__(self):
        return self.keys()