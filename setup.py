import setuptools

# Read the contents of your README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='point_cloud_moving_least_squares',
    version='0.1.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'jax',
        'jaxlib',
        'scikit-learn',
        'progressbar2'
    ],
    author='Dillon Cislo',
    author_email='dilloncislo@gmail.com',
    description='An implementation of Moving Least Squares for N-dimensional point clouds.',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
