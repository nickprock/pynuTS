from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Get the current directory
here = Path(__file__).parent.absolute()

# Read version from version.py
version_file = here / 'pynuTS' / 'version.py'
if version_file.exists():
    exec(open(version_file).read())
else:
    __version__ = '0.3.0'  # Fallback version

# Read the README file
readme_file = here / 'README.md'
if readme_file.exists():
    with open(readme_file, encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'A python library for Time Series based on IAML blog articles'

# Read requirements from requirements.txt if it exists
requirements_file = here / 'requirements.txt'
if requirements_file.exists():
    with open(requirements_file) as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    # Core dependencies for the improved modules
    install_requires = [
        'numpy>=1.19.0',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
        'scipy>=1.6.0',
        'tqdm>=4.50.0',
    ]

# Optional dependencies for enhanced performance and functionality
extras_require = {
    'fast': [
        'numba>=0.53.0',  # For JIT compilation in DTW and clustering
        'cython>=0.29.0',  # For potential Cython extensions
        'joblib>=1.0.0',   # Essential for parallel DTW clustering
    ],
    'parallel': [
        'joblib>=1.0.0',   # For parallel computing
        'dask>=2021.3.0',  # For distributed computing
        'psutil>=5.8.0',   # For memory monitoring
        'concurrent-futures>=3.1.1; python_version<"3.2"',  # Backport for older Python
    ],
    'streaming': [
        'dask>=2021.3.0',  # For streaming data processing
        'zarr>=2.8.0',     # For efficient array storage
        'h5py>=3.2.0',     # For HDF5 time series storage
        'pyarrow>=3.0.0',  # For efficient data serialization
    ],
    'plotting': [
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
        'bokeh>=2.3.0',    # For interactive DTW path visualization
    ],
    'advanced': [
        'dask>=2021.3.0',  # For parallel processing
        'joblib>=1.0.0',   # For parallel computing
        'numba>=0.53.0',   # For JIT compilation
        'fastdtw>=0.3.4',  # Alternative DTW implementation
        'tslearn>=0.5.0',  # Additional time series algorithms
        'sktime>=0.10.0',  # Time series machine learning
    ],
    'benchmark': [
        'memory-profiler>=0.60.0',  # For memory usage profiling
        'line-profiler>=3.3.0',    # For line-by-line profiling
        'py-spy>=0.3.0',           # For production profiling
        'psutil>=5.8.0',           # For system monitoring
        'matplotlib>=3.3.0',       # For benchmark visualization
    ],
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'pytest-xdist>=2.2.0',     # Parallel testing
        'pytest-benchmark>=3.4.0',  # Performance testing
        'pytest-timeout>=2.1.0',    # Timeout for long-running tests
        'black>=21.0.0',
        'flake8>=3.8.0',
        'mypy>=0.812',
        'pre-commit>=2.10.0',
        'sphinx>=3.5.0',
        'sphinx-rtd-theme>=0.5.0',
        'memory-profiler>=0.60.0',   # For profiling clustering algorithms
        'line-profiler>=3.3.0',     # For detailed performance analysis
    ],
    'notebook': [
        'jupyter>=1.0.0',
        'ipywidgets>=7.6.0',
        'jupyterlab>=3.0.0',
        'ipyparallel>=7.0.0',       # For parallel notebook computing
    ],
    'enterprise': [
        'dask>=2021.3.0',
        'joblib>=1.0.0',
        'numba>=0.53.0',
        'psutil>=5.8.0',
        'h5py>=3.2.0',
        'zarr>=2.8.0',
        'pyarrow>=3.0.0',
        'redis>=4.0.0',             # For distributed caching
        'sqlalchemy>=1.4.0',        # For database integration
    ]
}

# All optional dependencies
extras_require['all'] = list(set(sum(extras_require.values(), [])))

# Python version requirement
python_requires = '>=3.7'

# Classifiers for better package discovery
classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

# Entry points for command-line tools
entry_points = {
    'console_scripts': [
        'pynuts-benchmark=pynuTS.cli:benchmark_command',
        'pynuts-cluster=pynuTS.cli:cluster_command',
        'pynuts-profile=pynuTS.cli:profile_command',
    ],
}

# Project URLs for better package page
project_urls = {
    'Homepage': 'https://github.com/nickprock/pynuTS',
    'Documentation': 'https://pynuts.readthedocs.io/',  # When available
    'Source': 'https://github.com/nickprock/pynuTS',
    'Tracker': 'https://github.com/nickprock/pynuTS/issues',
    'Blog': 'https://iaml.it/blog/',
}

# Keywords for better discoverability
keywords = [
    'time-series', 'timeseries', 'time series analysis',
    'dtw', 'dynamic time warping', 'fast-dtw',
    'sax', 'symbolic aggregate approximation',
    'imputation', 'missing values',
    'clustering', 'k-means', 'ensemble clustering',
    'parallel computing', 'scalable algorithms',
    'streaming data', 'online learning',
    'machine learning', 'data science',
    'signal processing', 'pattern recognition',
    'performance optimization', 'numba', 'jit'
]

setup(
    name='pynuTS',
    version=__version__,
    description='A modern Python library for scalable Time Series analysis with advanced clustering and DTW algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # Author information
    author='Nicola Procopio',
    author_email='nico.pro412@gmail.com',
    maintainer='Nicola Procopio',
    maintainer_email='nico.pro412@gmail.com',
    
    # URLs
    url='https://github.com/nickprock/pynuTS',
    project_urls=project_urls,
    
    # Package discovery
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    
    # Dependencies
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Package data
    include_package_data=True,
    package_data={
        'pynuTS': [
            'data/*.csv',
            'data/*.json',
            'templates/*.html',
        ],
    },
    
    # Metadata
    license='MIT',
    classifiers=classifiers,
    keywords=keywords,
    
    # Entry points
    entry_points=entry_points,
    
    # Zip safety
    zip_safe=False,
    
    # Additional metadata for modern setuptools
    platforms=['any'],
)