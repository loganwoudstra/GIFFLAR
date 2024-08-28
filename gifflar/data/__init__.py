import pathlib
import sys

from rdkit import Chem, RDLogger

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
RDLogger.DisableLog('rdApp.info')

# If working on windows as there are problems with PosixPaths
if sys.platform.startswith("win"):
    tmp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath