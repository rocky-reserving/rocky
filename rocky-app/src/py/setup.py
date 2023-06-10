from setuptools import setup, find_packages

setup(
    name="rocky",
    version="0.1",
    packages=find_packages(),
    description="Triangle-based property-casualty loss reserving models.",
    author="Andy Weaver",
    author_email="andrew_weaver@cinfin.com",
    # url="https://github.com/username/my_package",  # Optional
    install_requires=[  # Optional
        "requests",
        "numpy",
        "pandas",
        "openpyxl",
        "plotly",
        "matplotlib",
        "scipy",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9.16",
    ],
)
