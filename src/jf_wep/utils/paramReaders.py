"""Functions for reading parameters from files."""
import inspect
from pathlib import Path
from typing import Any, Union

import yaml


def getModuleDir() -> Path:
    """Get the root directory of the module.

    Returns
    -------
    Path
        Path to root directory of the module.
    """
    # Get the root name of the package
    package = __package__ or __name__
    package = package.split(".")[0]

    # Get the path of this file
    file = __file__

    # Get the path to the root of this package
    path = package.join(file.split(package)[:-2]) + package

    return Path(path)


def getPolicyDir() -> Path:
    """Get the path to the policy directory, which holds configuration files.

    Returns
    -------
    Path
        Path to policy directory.
    """
    return Path(getModuleDir()) / "policy"


def resolvePolicyPath(path: Union[Path, str]) -> Path:
    """Resolve a relative policy path into an absolute path.

    Note this returns the absolute path, regardless of whether that
    path points to an existing file.

    Parameters
    ----------
    path: Path or str
        Path relative to the policy directory. Can start with
        "policy/" or not.

    Returns
    -------
    Path
        Absolute path to the policy file
    """
    # Make a path object
    path = Path(path)

    # If the path starts with policy, remove it
    if path.parts[0] == "policy":
        path = Path(*path.parts[1:])

    return Path(getPolicyDir()) / path


def readParamYaml(path: Union[Path, str]) -> dict:
    """Read the parameter yaml file and return dictionary.

    Parameters
    ----------
    path : Path or str
        The path to the parameter file. Can be absolute or relative, but if
        path starts with "policy/", then it will be understood as relative
        to the jf_wep policy directory. If the path does not end with ".yaml"
        this will be appended before searching for the file.

    Returns
    -------
    dict
        The dictionary of parameters
    """
    # Make a path object
    path = Path(path)

    # Is this relative to the policy directory?
    if path.parts[0] == "policy":
        path = resolvePolicyPath(path)

    # Add .yaml to the end if it doesn't already exist
    path = path.with_suffix(".yaml")

    # Read the parameter file into a dictionary
    with open(path, "r") as file:
        params = yaml.safe_load(file)

    return params


def mergeParams(paramFile: Union[Path, str, None], **kwargs: Any) -> dict:
    """Load the default parameters and update using the kwargs.

    If the paramFile is none, then the keyword arguments are just returned
    verbatim.

    Parameters
    ----------
    paramFile : Path or str or None
        Path to the default parameter file.
    kwargs : Any
        Keyword arguments with which to update the default parameters.
        Note that None values are ignored.

    Returns
    -------
    dict
        Dictionary of parameters, with the defaults updated
        by the keyword arguments.
    """
    # Get the default params
    if paramFile is None:
        params = {}
    else:
        params = readParamYaml(paramFile)

    # Get the list of all keys
    keys = (params | kwargs).keys()

    # Merge the dictionaries
    mergedParams = {}
    for key in keys:
        if kwargs.get(key, None) is not None:
            mergedParams[key] = kwargs[key]
        else:
            mergedParams[key] = params.get(key, None)

    return mergedParams


def loadConfig(config: Union[Any, Path, str, dict], item: Any) -> Any:
    """Load the config into the item.

    This function is a very generic wrapper around the process of loading a
    config into an item. It is used in __init__ and config methods so that
    the parameters passed into those methods can take on a wide variety of
    types.

    The item that is passed to this function can be a class, in which case a
    new instance with the config is returned. Or the item can be an instance
    of a class that posses a config method, in which case the item is updated
    using the config. However, if the item is a class, and the config is an
    instance of that class, the config is just returned unaltered.

    Parameters
    ----------
    config : class instance or Path or str or dict
        The configuration to use for loading the item.
        If Path or str, it is assumed this points to a config file.
        If dictionary, it is assumed this holds keyword arguments.
        If item is a class, and config is an instance of that class,
        then config is just returned unaltered.
    item : Any
        A class representing the object to be configured.

    Returns
    -------
    class instance
        A class instance with the provided configuration.

    Raises
    ------
    TypeError
        If item is not a class
    """
    # Check that item is a class
    if not inspect.isclass(item):
        raise TypeError("item must be a class.")

    # If the config is an instance of this class, just return it
    if isinstance(config, item):
        return config

    # If config is a Path or string, pass config as configFile
    if isinstance(config, Path) or isinstance(config, str):
        return item(configFile=config)
    # If it's a dictionary, pass keyword arguments
    elif isinstance(config, dict):
        return item(**config)
    # If it's a None, try instantiating the class with its defaults
    elif config is None:
        return item()
    # If it's none of these types, raise an error
    else:
        raise TypeError(
            "config must be a Path, string, dictionary or "
            "an instance of the class specified by item. "
            "It can also be None, in which case the default "
            "config for item is used."
        )
