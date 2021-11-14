import pathlib
from setuptools import setup, find_packages

"""
:copyright:
    Simon Schneider
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

INSTALL_REQUIRES = [
    'obspy>=1.2.2',
    'numpy>=1.17.2',
    'scipy'
    ]

MIN_PYTHON_VERSION = (3, 6)

setup(
    name="bowpy",
    version="0.1.1",
    description="Python toolbox for body wave seismologists",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/s-schneider/bowpy",
    author="Simon Schneider",
    author_email="s.a.schneider@mailbox.org",
    license='GNU Lesser General Public License, Version 3 (LGPLv3)',
    platforms='OS Independent',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Library or ' +
            'Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'],
    packages=find_packages(),

#     package_data={'bowpy': ['data/*/*/*', 'data/*/*']},

    python_requires=f'>={MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}',
    install_requires=INSTALL_REQUIRES
)
