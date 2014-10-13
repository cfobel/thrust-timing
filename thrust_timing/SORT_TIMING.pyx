#distutils: language=c++
#cython: embedsignature=True, boundscheck=False
from libc.stdint cimport int32_t, bool, uint8_t
from cythrust.thrust.device_vector cimport device_vector
from cythrust.thrust.partition cimport (counted_stable_partition,
                                        counted_stable_partition_w_stencil)
from cythrust.thrust.sort cimport sort_by_key
from cythrust.thrust.fill cimport fill_n, fill
from cythrust.thrust.transform cimport transform2
from cythrust.thrust.reduce cimport reduce_by_key
from cythrust.thrust.functional cimport (positive, non_negative, minimum,
                                         equal_to, plus, maximum)
from cythrust.thrust.tuple cimport (make_tuple2, make_tuple3, make_tuple4,
                                    make_tuple5, make_tuple6, make_tuple7)
from cythrust.thrust.iterator.permutation_iterator cimport make_permutation_iterator
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.device_vector cimport (DeviceVectorViewInt32,
                                     DeviceVectorViewFloat32,
                                     DeviceVectorViewUint8)
from cythrust.thrust.copy cimport copy, copy_n
from cythrust.thrust.replace cimport replace_if_w_stencil


def step1(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewFloat32 longest_paths,
          DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          DeviceVectorViewFloat32 max_target_longest_path,
          DeviceVectorViewInt32 reduced_keys):
    # Equivalent to:
    #
    #     block_min_source_longest_path[:] = -1e6
    fill(block_min_source_longest_path._begin,
         block_min_source_longest_path._end, <float>(-1e6))

    # Equivalent to:
    #
    #     connections['source_longest_path'] = longest_paths[connections
    #                                                        .loc[:, 'source_key']
    #                                                        .values]
    copy(make_permutation_iterator(longest_paths._begin, source_key._begin),
         make_permutation_iterator(longest_paths._end, source_key._end),
         source_longest_path._begin)

def step2(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewFloat32 longest_paths,
          DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          DeviceVectorViewFloat32 max_target_longest_path,
          DeviceVectorViewInt32 reduced_keys):
    cdef positive[uint8_t] positive_f
    # Equivalent to:
    #
    #     connections.loc[connections.sync_source > 0, 'source_longest_path'] = 0
    replace_if_w_stencil(source_longest_path._begin, source_longest_path._end, 
        sync_source._begin, positive_f, 0)


def step3(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewFloat32 longest_paths,
          DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          DeviceVectorViewFloat32 max_target_longest_path,
          DeviceVectorViewInt32 reduced_keys):
    cdef equal_to[int32_t] equal_to
    cdef minimum[float] minimum_f

    sort_by_key(target_key._begin, target_key._end,
                make_zip_iterator(
                    make_tuple7(source_key._begin, sync_source._begin,
                                sync_target._begin, delay._begin,
                                target_longest_path._begin,
                                min_source_longest_path._begin,
                                source_longest_path._begin)))

    #min_source_longest_path = connections.groupby('target_key').agg({'source_longest_path': np.min})
    cdef size_t unique_block_count = (
        <device_vector[int32_t].iterator>reduce_by_key(
            target_key._begin, target_key._end, source_longest_path._begin,
            reduced_keys._begin, min_source_longest_path._begin, equal_to,
            minimum_f).first -
        <device_vector[int32_t].iterator>reduced_keys._begin)
    return unique_block_count


def step4(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewFloat32 longest_paths,
          DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          DeviceVectorViewFloat32 max_target_longest_path,
          DeviceVectorViewInt32 reduced_keys, size_t unique_block_count):
    cdef non_negative[int32_t] non_negative_f

    # block_min_source_longest_path[min_source_longest_path.index] = min_source_longest_path.values
    copy_n(min_source_longest_path._begin, unique_block_count,
           make_permutation_iterator(block_min_source_longest_path._begin,
                                     reduced_keys._begin))


def step5(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewFloat32 longest_paths,
          DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          DeviceVectorViewFloat32 max_target_longest_path,
          DeviceVectorViewInt32 reduced_keys):
    # connections.min_source_longest_path = block_min_source_longest_path[connections.target_key.values].values
    copy(make_permutation_iterator(block_min_source_longest_path._begin,
                                   target_key._begin),
         make_permutation_iterator(block_min_source_longest_path._begin,
                                   target_key._end),
         min_source_longest_path._begin)


def step6(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewFloat32 longest_paths,
          DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          DeviceVectorViewFloat32 max_target_longest_path,
          DeviceVectorViewInt32 reduced_keys):
    cdef non_negative[int32_t] non_negative_f

    # Equivalent to:
    #
    #     connections.sort(['min_source_longest_path', 'target_key'], inplace=True,
    #                      ascending=False)
    #     ready_connection_count = (connections.min_source_longest_path >= 0).sum()
    cdef ready_connection_count = counted_stable_partition_w_stencil(
        make_zip_iterator(
            make_tuple6(source_key._begin, target_key._begin,
                        sync_source._begin, sync_target._begin,
                        target_longest_path._begin,
                        source_longest_path._begin)),
        make_zip_iterator(
            make_tuple6(source_key._end, target_key._end, sync_source._end,
                        sync_target._end, target_longest_path._end,
                        source_longest_path._end)),
        min_source_longest_path._begin, non_negative_f)
    counted_stable_partition(min_source_longest_path._begin,
                             min_source_longest_path._end, non_negative_f)
    return ready_connection_count


def step7(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewFloat32 longest_paths,
          DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          DeviceVectorViewFloat32 max_target_longest_path,
          DeviceVectorViewInt32 reduced_keys, size_t ready_connection_count):
    # Equivalent to:
    #
    #     connections[:ready_connection_count].delay = 1.
    fill_n(delay._begin, ready_connection_count, 1)


def step8(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewFloat32 longest_paths,
          DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          DeviceVectorViewFloat32 max_target_longest_path,
          DeviceVectorViewInt32 reduced_keys, size_t ready_connection_count):
    cdef plus[float] plus_f
    # Equivalent to:
    #
    #     connections[:ready_connection_count].target_longest_path = \
    #         (connections[:ready_connection_count].delay +
    #          connections[:ready_connection_count].source_longest_path)
    transform2(delay._begin, delay._begin + ready_connection_count,
               source_longest_path._begin, target_longest_path._begin, plus_f)


def step9(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewFloat32 longest_paths,
          DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          DeviceVectorViewFloat32 max_target_longest_path,
          DeviceVectorViewInt32 reduced_keys, size_t ready_connection_count):
    cdef maximum[float] maximum_f
    cdef equal_to[int32_t] equal_to
    # Equivalent to:
    #
    #     max_target_longest_path = \
    #         (connections[:ready_connection_count].groupby('target_key')
    #          .agg({'target_longest_path': np.max}))
    cdef size_t resolved_block_count = (
        <device_vector[int32_t].iterator> reduce_by_key(
            target_key._begin, target_key._begin + ready_connection_count,
            target_longest_path._begin, reduced_keys._begin,
            max_target_longest_path._begin, equal_to, maximum_f).first -
        <device_vector[int32_t].iterator>reduced_keys._begin)
    return resolved_block_count


def step10(DeviceVectorViewFloat32 block_min_source_longest_path,
           DeviceVectorViewFloat32 longest_paths,
           DeviceVectorViewInt32 source_key,
           DeviceVectorViewInt32 target_key,
           DeviceVectorViewUint8 sync_source,
           DeviceVectorViewUint8 sync_target,
           DeviceVectorViewFloat32 delay,
           DeviceVectorViewFloat32 target_longest_path,
           DeviceVectorViewFloat32 min_source_longest_path,
           DeviceVectorViewFloat32 source_longest_path,
           DeviceVectorViewFloat32 max_target_longest_path,
           DeviceVectorViewInt32 reduced_keys, size_t resolved_block_count):
    # Equivalent to:
    #
    #     longest_paths[max_target_longest_path.index] = max_target_longest_path.values
    copy_n(max_target_longest_path._begin, resolved_block_count,
           make_permutation_iterator(longest_paths._begin, reduced_keys._begin))


cpdef fill_longest_paths(DeviceVectorViewInt32 external_block_keys,
                         DeviceVectorViewInt32 sync_logic_block_keys,
                         DeviceVectorViewInt32 single_connection_blocks,
                         DeviceVectorViewFloat32 longest_paths):
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


