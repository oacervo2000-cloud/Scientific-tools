from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='s_index_tool',
    version='0.1',
    packages=find_packages(),
    description='A tool to calculate the S-index from stellar spectra.',
    author='Rafael Ferreira',
    author_email='rafael.ferreira@email.com',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)
