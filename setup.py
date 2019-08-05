from setuptools import setup, find_packages

setup(name='Minari',
      version='1.0.0',
      description='Powerful library for solar power prediction',
      url='https://github.com/shhj1998/minari',
      author='shhj1998',
      author_email='seunghyuk.cho@triple3e.com',
      license='MIT',
      packages=find_packages(exclude=['contrib', 'docs', 'tests']),
      install_requires=['numpy', 'torch'])
