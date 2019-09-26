from setuptools import setup, find_packages

setup(
    name='luke',
    version='0.0.1',
    description='Language understanding with knowledge-based embeddings',
    author='Ikuya Yamada',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    entry_points=dict(console_scripts=['luke=luke.cli:cli']),
)
