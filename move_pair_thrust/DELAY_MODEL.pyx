#distutils: language=c++
#cython: embedsignature=True, boundscheck=False
from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t, int32_t
from libc.math cimport fmin
import numpy as np
cimport numpy as np
from cythrust.thrust.copy cimport copy_n, copy_if_w_stencil
from cythrust.thrust.fill cimport fill_n
from cythrust.thrust.functional cimport (unpack_binary_args, square, equal_to,
                                         not_equal_to, unpack_quinary_args,
                                         plus, minus, reduce_plus4, identity,
                                         logical_not)
from cythrust.thrust.iterator.counting_iterator cimport counting_iterator
from cythrust.thrust.iterator.permutation_iterator cimport make_permutation_iterator
from cythrust.thrust.iterator.repeated_range_iterator cimport repeated_range
from cythrust.thrust.iterator.transform_iterator cimport make_transform_iterator
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.thrust.reduce cimport accumulate, accumulate_by_key, reduce_by_key
from cythrust.thrust.scan cimport exclusive_scan, inclusive_scan
from cythrust.thrust.sequence cimport sequence
from cythrust.thrust.sort cimport sort_by_key, sort
from cythrust.thrust.transform cimport transform, transform2
from cythrust.thrust.tuple cimport make_tuple5, make_tuple4, make_tuple2
from cythrust.device_vector cimport DeviceVectorInt32, DeviceVectorFloat32
cimport cython


cpdef fill_arrival_times(DeviceVectorInt32 clocked_driver_block_keys,
                         DeviceVectorInt32 global_block_keys,
                         DeviceVectorInt32 single_connection_blocks,
                         DeviceVectorFloat32 arrival_times):
    cdef size_t count

    count = arrival_times._vector.size()

    fill_n(arrival_times._vector.begin(), count, -1)

    count = clocked_driver_block_keys._vector.size()

    fill_n(make_permutation_iterator(arrival_times._vector.begin(),
                                     clocked_driver_block_keys._vector.begin()),
           count, 0)

    count = global_block_keys._vector.size()

    fill_n(make_permutation_iterator(arrival_times._vector.begin(),
                                     global_block_keys._vector.begin()), count,
           0)

    count = single_connection_blocks._vector.size()

    fill_n(make_permutation_iterator(arrival_times._vector.begin(),
                                     single_connection_blocks._vector.begin()),
           count, 0)
