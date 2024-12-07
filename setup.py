#
#  _                 _     _            
# | |__  _   ___   _(_) __| | ___  ___  
# | '_ \| | | \ \ / / |/ _` |/ _ \/ _ \ 
# | | | | |_| |\ V /| | (_| |  __/ (_) |
# |_| |_|\__, | \_/ |_|\__,_|\___|\___/ 
#        |___/
#
# Maintained by the mage.space team.
# Thank you to the Tencent team.                     
from setuptools import setup, find_packages

# Read the contents of requirements.txt file
with open("requirements.txt") as f:
    required = f.read().splitlines()

# Read the contents of strict requirements.txt file
with open("requirements-strict.txt") as f:
    required_strict = f.read().splitlines()

setup(
    name="mageic",
    version="0.13.7",
    packages=find_packages(),
    license=open("LICENSE.txt").read(),
    long_description=open("README.md").read(),
    install_requires=required,
    extras_require={
        "strict": required_strict,
    },
    entry_points={
        "console_scripts": [
            "mageic=mageic.cli:main",
        ],
    },
)