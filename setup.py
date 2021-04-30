from setuptools import setup
import io

with io.open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()

setup_args = {
    'name': 'quflow',
    'author': 'Klas Modin',
    'url': 'https://bitbucket.org/kmodin/quflow',
    'license': 'MIT',
    'description': '',
    'long_description': readme,
    'long_description_content_type': 'text/markdown',
    'package_dir': {'quflow': 'quflow'},
    'packages': ['quflow', ],
    'version': '0.0.1',
    'include_package_data': True,
    'test_suite': 'pytest',
    'tests_require': ['pytest', 'matplotlib'],
    'setup_requires': ['pytest-runner'],
    'install_requires': ['numpy', 'numba', 'scipy', 'pyssht>=1.3.4'],
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
