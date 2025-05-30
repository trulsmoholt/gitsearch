from setuptools import setup

setup(
    name='gitsearch',
    version='0.1',
    py_modules=['gitsearch'],
    install_requires=[
        'openai',
        'gitpython',
    ],
    entry_points='''
        [console_scripts]
        gitsearch=gitsearch:main
    ''',
) 