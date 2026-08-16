"""Small legacy setuptools shim for environments with older pip."""

from setuptools import find_packages, setup


setup(
    name="drift-sense",
    version="0.2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["numpy>=1.24,<2.0", "Pillow>=9.5,<11.0"],
    python_requires=">=3.9",
)

