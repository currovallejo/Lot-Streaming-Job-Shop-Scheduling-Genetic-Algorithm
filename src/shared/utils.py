from pathlib import Path
import yaml
import time
import logging


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file, resolving paths relative to the project root.
    Args:
        config_path (str): Path to the configuration file relative to the project root.
    Returns:
        dict: Parsed configuration data.
    """
    project_root = Path(__file__).resolve().parents[2]
    config_file = project_root / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def timed(function):
    """Decorator to time the execution of a function and log the time taken."""

    def wrapper(*args, **kwargs):
        if args and hasattr(args[0], "__class__"):
            func_name = f"{args[0].__class__.__name__}.{function.__name__}"
        else:
            func_name = function.__name__
        start = time.time()
        value = function(*args, **kwargs)
        end = time.time()
        print(f"Function {func_name} took {end - start} seconds to run\n")
        return value

    return wrapper


def init_logger():
    """
    Configure the root logger at INFO level with a simple message format,
    and return a logger you can call .info() on.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    return logging.getLogger()
