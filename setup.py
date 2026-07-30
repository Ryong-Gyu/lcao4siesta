from setuptools import find_packages, setup


setup(
    name='lcao4siesta',
    version='0.1.0',
    description='Post-process SIESTA LCAO data in Python',
    packages=find_packages(include=('lcao', 'lcao.*')),
    py_modules=['FortranFile', 'lcao4siesta', 'siesta_io'],
    python_requires='>=3.8',
    install_requires=['numpy>=1.21', 'scipy>=1.7', 'numba>=0.56'],
    extras_require={'test': ['pytest>=7']},
)
