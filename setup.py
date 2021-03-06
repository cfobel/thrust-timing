import numpy as np
from path_helpers import path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import pkg_resources

import version


pyx_files = ['thrust_timing/DELAY_MODEL.pyx', 'thrust_timing/SORT_TIMING.pyx']
ext_modules = [Extension(f[:-4].replace('/', '.'), [f],
                         extra_compile_args=['-O3', '-msse3', '-std=c++0x',
                                             '-fopenmp'],
                         #extra_link_args=['-lgomp'],
                         include_dirs=['camip',
                                       path('~/local/include').expand(),
                                       '/usr/local/cuda-6.5/include',
                                       pkg_resources
                                       .resource_filename('cythrust', ''),
                                       np.get_include()],
                         define_macros=[('THRUST_DEVICE_SYSTEM',
                                         'THRUST_DEVICE_SYSTEM_CPP')])
                                         #'THRUST_DEVICE_SYSTEM_OMP')])
               for f in pyx_files]


setup(name='thrust-timing',
      version=version.getVersion(),
      description='Thrust FPGA timing calculations',
      keywords='fpga timing thrust',
      author='Christian Fobel',
      author_email='christian@fobel.net',
      #url='http://github.com/wheeler-microfluidics/microdrop_utility.git',
      license='GPL',
      install_requires=['pandas', 'numpy', 'scipy'],
      packages=['thrust_timing'],
      ext_modules=cythonize(ext_modules))
