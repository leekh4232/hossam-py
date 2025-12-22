from setuptools import setup, find_packages

setup(
    name="hossam",
    version="0.2.0",
    description="Hossam Data Helper",
    author="Lee Kwang-Ho",
    author_email="leekh4232@gmail.com",
    license="MIT",
    packages=find_packages(exclude=[]),
    keywords=["data", "analysis", "helper", "hossam", "tensorflow", "이광호"],
    python_requires=">=3.13.9",
    zip_safe=False,
    url="https://github.com/leekh4232/hossam-data",
    install_requires=[
        "tqdm",
        "tabulate",
        "pandas",
        "matplotlib",
        "seaborn",
        "requests",
        "openpyxl",
        "xlrd",
        "statsmodels",
        "scipy",
        "pingouin",
        "statannotations"
    ],
    include_package_data=True,
    long_description=open('README.md', encoding='utf-8').read() if __name__ == '__main__' else '',
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        # License classifier deprecated in favor of SPDX license expressions.
        # license is already set to 'MIT' above; remove the classifier to avoid warnings.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
    ],
)