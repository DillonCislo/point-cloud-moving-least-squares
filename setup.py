import setuptools

setuptools.setup(
    name='point_cloud_moving_least_squares',
    version='0.1.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy'
        'jax',
        'jaxlib',
        'scikit-learn'
        'progressbar2'
    ],
    author='Dillon Cislo',
    author_email='dilloncislo@gmail.com',
    description='An implementation of Moving Least Squares for N-dimensional point clouds.'
)
