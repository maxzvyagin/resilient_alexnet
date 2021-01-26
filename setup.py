"""Setup for the hyper_resilient_experiments package."""

import setuptools

with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Max Zvyagin",
    author_email="max.zvyagin7@gmail.com",
    name='resilient_alexnet',
    license="MIT",
    long_description_content_type="text/markdown",
    description='Experiments for alexnet resiliency and sensitivity',
    version='v0.0.2',
    long_description=README,
    url='https://github.com/maxzvyagin/resilient_alexnet',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=['scikit-learn', 'scikit-optimize', 'ray', "ray [tune]", "hyperspaces", "spaceray"],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)