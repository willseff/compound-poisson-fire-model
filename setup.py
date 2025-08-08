from setuptools import setup, find_packages

setup(
    name="compound_poisson_fire_model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "geopandas"
    ],
)