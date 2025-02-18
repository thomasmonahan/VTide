from setuptools import setup, find_packages

setup(
    name='vtide',  
    version='0.1.0', 
    author='Thomas Monahan',
    author_email='thomas.monahan@eng.ox.ac.uk',
    description='VTide: A python package for variational Bayesian tidal harmonic analysis ',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thomasmonahan/VTide',  
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'utide',
        'scipy',
        'matplotlib.pylot',
        'sys',
        'pdb'

    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)