from setuptools import setup
import setuptools

setup(
    name='dead_reckoning_forecast',
    version='0.1.0',    
    description='',
    url='https://github.com/R-N/dead_reckoning_forecast',
    author='Muhammad Rizqi Nur',
    author_email='rizqinur2010@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        "moviepy==1.0.3",
        "ffmpeg==1.4",
        "torch",
        "torchvision",
        "torchinfo",
        "opencv-python",
        "numpy",
        "scikit-learn",
        "pandas",
        "matplotlib",
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
)