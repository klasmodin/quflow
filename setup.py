import io
import re

from setuptools import setup

with io.open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    io.open('quflow/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

setup_args = {
    'name': 'quflow',
    'author': 'Klas Modin',
    'url': 'https://github.com/kmodin/quflow/',
    'license': 'MIT',
    'description': '',
    'long_description': readme,
    'long_description_content_type': 'text/markdown',
    'package_dir': {'quflow': 'quflow'},
    'packages': ['quflow', ],
    'version': __version__,
    'include_package_data': True,
    'test_suite': 'pytest',
    'tests_require': ['pytest', 'matplotlib'],
    'setup_requires': ['pytest-runner'],
    'install_requires': ['numpy', 'numba', 'scipy', 'pyssht>=1.5.2', 'appdirs'],
    'classifiers': ['Development Status :: 3 - Alpha',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: MIT License',
                    'Programming Language :: Python :: 3.8',
                    'Programming Language :: Python :: 3.9',
                    'Topic :: Scientific/Engineering :: Physics'],
    'keywords': 'quantization hydrodynamics vorticity'
}

if __name__ == '__main__':
    setup(**setup_args)
