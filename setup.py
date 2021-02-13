import setuptools

setuptools.setup(
    name='adacrowd',
    version='1.0.0',
    author='Mahesh Kumar Krishna Reddy',
    description='Pytorch implementation of the AdaCrowd method for unlabeled scene adaptation',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'yacs',
        'scipy',
        'tqdm',
        'scikit-image',
        'tensorboardX',
        'easydict',
        'opencv-python',
        'pandas',
        'pillow'
    ]
)
