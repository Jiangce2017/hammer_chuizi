import os
from setuptools import find_packages, setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_version():
    with open(os.path.join(_CURRENT_DIR, "src/hammer/__version__.py")) as file:
        for line in file:
            if line.startswith("__version__"):
                return line[line.find("=") + 1:].strip(' \'"\n')
        raise ValueError('`__version__` not defined in `hammer/__version__.py`')

__version__ = get_version()

if __name__=='__main__':
    setup(
        name="hammer",
        version=__version__,
        description="Generate hexahedron mesh and toolpath for thermal analysis in additive manufacturing",
        author="Jiangce Chen",
        author_email="jiangcechen@gmail.com",
        long_description=open(os.path.join(_CURRENT_DIR, "README.md")).read(),
        long_description_content_type='text/markdown',
        url="https://github.com/Jiangce2017/hammer_chuizi",
        license="GPL-3.0",
    )
