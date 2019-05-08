from __future__ import absolute_import, print_function

try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path

try:
    import tempfile
    tempfile.TemporaryDirectory
except (ImportError,AttributeError):
    from backports import tempfile

try:
    FileNotFoundError = FileNotFoundError
except NameError:
    FileNotFoundError = IOError
