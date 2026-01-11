from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlops-project")
except PackageNotFoundError:
    __version__ = "0.0.0"