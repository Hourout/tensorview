from io import open
from setuptools import setup, find_packages

def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

setup(name='tensorview',
      version='0.2.0',
      install_requires=['matplotlib', 'tensorflow>=2.0.0b0'],
      description='Dynamic visualization training service in Jupyter Notebook for Keras tf.keras and others.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/Hourout/tensorview',
      author='JinQing Lee, Gaojie Wei',
      author_email='hourout@163.com',
      keywords=['keras-visualization', 'tensorflow-visualization', 'keras', 'tensorflow', 'tf.keras', 'plot', 'chart'],
      license='Apache License Version 2.0',
      classifiers=[
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Visualization',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ],
      packages=find_packages(),
      zip_safe=False)
