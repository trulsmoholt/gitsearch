from setuptools import setup, find_packages

setup(
    name="gitsearch",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "gitpython>=3.1.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0"
    ],
    entry_points={
        'console_scripts': [
            'gitsearch=gitsearch.__main__:main',
        ],
    },
    python_requires=">=3.8",
) 