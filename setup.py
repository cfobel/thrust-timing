import numpy as np
from path_helpers import path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import pkg_resources

import version


pyx_files = ['move_pair_thrust/DELAY_MODEL.pyx']
ext_modules = [Extension(f[:-4].replace('/', '.'), [f],
                         extra_compile_args=['-O3', '-msse3', '-std=c++0x'],
                         include_dirs=['camip',
                                       path('~/local/include').expand(),
                                       '/usr/local/cuda-6.5/include',
                                       pkg_resources
                                       .resource_filename('cythrust', ''),
                                       np.get_include()],
                         define_macros=[('THRUST_DEVICE_SYSTEM',
                                         'THRUST_DEVICE_SYSTEM_CPP')])
               for f in pyx_files]


setup(name='move_pair_thrust',
      version=version.getVersion(),
      description='Thrust FPGA timing calculations',
      keywords='fpga timing thrust',
      author='Christian Fobel',
      author_email='christian@fobel.net',
      #url='http://github.com/wheeler-microfluidics/microdrop_utility.git',
      license='GPL',
      install_requires=['pandas', 'numpy', 'scipy'],
      ext_modules=cythonize(ext_modules))
