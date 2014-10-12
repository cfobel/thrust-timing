#distutils: language=c++
#cython: embedsignature=True, boundscheck=False
from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t, int32_t
from libc.math cimport fmin
import numpy as np
cimport numpy as np
from cythrust.thrust.copy cimport copy_n, copy_if_w_stencil
from cythrust.thrust.fill cimport fill_n
from cythrust.thrust.partition cimport partition_w_stencil
from cythrust.thrust.functional cimport (unpack_binary_args, square, equal_to,
                                         not_equal_to, unpack_quinary_args,
                                         plus, minus, reduce_plus4, identity,
                                         logical_not, negative)
from cythrust.thrust.iterator.counting_iterator cimport make_counting_iterator
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


cpdef fill_longest_paths(DeviceVectorInt32 external_block_keys,
                         DeviceVectorInt32 sync_logic_block_keys,
                         DeviceVectorInt32 single_connection_blocks,
                         DeviceVectorFloat32 longest_paths):
    cdef size_t count

    count = longest_paths._vector.size()

    fill_n(longest_paths._vector.begin(), count, -1)

    count = sync_logic_block_keys._vector.size()

    fill_n(make_permutation_iterator(longest_paths._vector.begin(),
                                     sync_logic_block_keys._vector.begin()),
           count, 0)

    count = external_block_keys._vector.size()

    fill_n(make_permutation_iterator(longest_paths._vector.begin(),
                                     external_block_keys._vector.begin()),
           count, 0)

    count = single_connection_blocks._vector.size()

    fill_n(make_permutation_iterator(longest_paths._vector.begin(),
                                     single_connection_blocks._vector.begin()),
           count, 0)


cpdef resolve_block_longest_paths(size_t unresolved_count,
                                  DeviceVectorFloat32 max_longest_path,
                                  DeviceVectorInt32 max_longest_path_index,
                                  DeviceVectorFloat32 block_longest_paths,
                                  DeviceVectorInt32 block_keys_to_resolve):
    r'''
    Equivalent to:

        block_longest_paths[block_keys_to_resolve] = \
            max_longest_path[max_longest_path_index]
    '''
    copy_n(
        make_permutation_iterator(max_longest_path._vector.begin(),
                                  max_longest_path_index._vector.begin()),
        unresolved_count,
        make_permutation_iterator(block_longest_paths._vector.begin(),
                                  block_keys_to_resolve._vector.begin()))


cpdef move_unresolved_data_to_front(DeviceVectorInt32 block_keys,
                                    DeviceVectorInt32 net_driver_block_key,
                                    DeviceVectorFloat32 driver_longest_path):
    '''
    Equivalent to the following, but without temporary intermediate data:

        ready_to_calculate = (driver_longest_path >= 0)
        temp = block_keys[~ready_to_calculate]
        temp2 = block_keys[ready_to_calculate]
        block_keys[:len(temp)] = temp
        block_keys[len(temp):] = temp2
    '''
    cdef negative[float] negative

    # result_type operator() (T1 j_is_sync, T2 delay_ij, T3 t_a_j) {
    partition_w_stencil(
        make_zip_iterator(
            make_tuple2(block_keys._vector.begin(),
                        net_driver_block_key._vector.begin())),
        make_zip_iterator(
            make_tuple2(block_keys._vector.end(),
                        net_driver_block_key._vector.end())),
        driver_longest_path._vector.begin(), negative)
