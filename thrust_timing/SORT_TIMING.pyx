#distutils: language=c++
#cython: embedsignature=True, boundscheck=False
from libc.stdint cimport int32_t, bool, uint8_t
from cython.operator cimport dereference as deref
from cythrust.thrust.device_vector cimport device_vector
from cythrust.thrust.partition cimport (counted_stable_partition,
                                        counted_stable_partition_w_stencil)
from cythrust.thrust.extrema cimport max_element
from cythrust.thrust.sort cimport sort_by_key
from cythrust.thrust.scatter cimport scatter
from cythrust.thrust.fill cimport fill_n, fill
from cythrust.thrust.transform cimport transform2
from cythrust.thrust.iterator.constant_iterator cimport make_constant_iterator
from cythrust.thrust.iterator.transform_iterator cimport make_transform_iterator
from cythrust.thrust.reduce cimport reduce_by_key, reduce, accumulate_by_key
from cythrust.thrust.functional cimport (positive, non_negative, minimum,
                                         equal_to, plus, maximum, absolute,
                                         unpack_binary_args, minus,
                                         unpack_ternary_args, divides)
from cythrust.thrust.tuple cimport (make_tuple2, make_tuple3, make_tuple4,
                                    make_tuple5, make_tuple6, make_tuple7)
from cythrust.thrust.iterator.permutation_iterator cimport make_permutation_iterator
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.device_vector cimport (DeviceVectorViewInt32,
                                     DeviceVectorFloat32,
                                     DeviceVectorViewFloat32,
                                     DeviceVectorViewUint8)
from cythrust.thrust.copy cimport copy, copy_n
from cythrust.thrust.replace cimport replace_if_w_stencil


cdef extern from "delay.h" nogil:
    cdef cppclass timing_delay 'delay' [T]:
        timing_delay(T, int32_t, int32_t)

    cdef cppclass c_connection_criticality 'connection_criticality' [T]:
        c_connection_criticality(T)

    cdef cppclass c_connection_cost 'connection_cost' [T]:
        c_connection_cost(T, T)


def sort_by_target_key(DeviceVectorViewInt32 source_key,
                       DeviceVectorViewInt32 target_key,
                       DeviceVectorViewUint8 sync_source,
                       DeviceVectorViewUint8 sync_target,
                       DeviceVectorViewFloat32 delay):
    sort_by_key(target_key._begin, target_key._end,
                make_zip_iterator(make_tuple4(source_key._begin,
                                              sync_source._begin,
                                              sync_target._begin,
                                              delay._begin)))


def reset_block_min_source_longest_path(
    DeviceVectorViewFloat32 block_min_source_longest_path):
    # Equivalent to:
    #
    #     block_min_source_longest_path[:] = -1e6
    fill(block_min_source_longest_path._begin,
         block_min_source_longest_path._end, <float>(-1e6))


def step1(DeviceVectorViewFloat32 longest_paths,
          DeviceVectorViewInt32 source_key,
          DeviceVectorViewFloat32 source_longest_path):
    # Equivalent to:
    #
    #     connections['source_longest_path'] = longest_paths[connections
    #                                                        .loc[:, 'source_key']
    #                                                        .values]
    copy(make_permutation_iterator(longest_paths._vector.begin(), source_key._begin),
         make_permutation_iterator(longest_paths._vector.begin(), source_key._end),
         source_longest_path._begin)


def step2(DeviceVectorViewUint8 sync_source,
          DeviceVectorViewFloat32 source_longest_path):
    cdef positive[uint8_t] positive_f
    # Equivalent to:
    #
    #     connections.loc[connections.sync_source > 0, 'source_longest_path'] = 0
    replace_if_w_stencil(source_longest_path._begin, source_longest_path._end,
                         sync_source._begin, positive_f, 0)


def step3(DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          DeviceVectorViewInt32 reduced_keys):
    '''
    __NB__ Data **must** be sorted by **`target_key`** before call this
    function to work as expected.
    '''
    cdef equal_to[int32_t] equal_to
    cdef minimum[float] minimum_f

    #min_source_longest_path = connections.groupby('target_key').agg({'source_longest_path': np.min})
    cdef size_t unique_block_count = (
        <device_vector[int32_t].iterator>reduce_by_key(
            target_key._begin, target_key._end, source_longest_path._begin,
            reduced_keys._begin, min_source_longest_path._begin, equal_to,
            minimum_f).first -
        <device_vector[int32_t].iterator>reduced_keys._begin)
    return unique_block_count


def step4(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewInt32 reduced_keys, size_t unique_block_count):
    cdef non_negative[int32_t] non_negative_f

    # block_min_source_longest_path[min_source_longest_path.index] = min_source_longest_path.values
    copy_n(min_source_longest_path._begin, unique_block_count,
           make_permutation_iterator(block_min_source_longest_path._begin,
                                     reduced_keys._begin))


def step5(DeviceVectorViewFloat32 block_min_source_longest_path,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewFloat32 min_source_longest_path):
    # connections.min_source_longest_path = block_min_source_longest_path[connections.target_key.values].values
    copy(make_permutation_iterator(block_min_source_longest_path._begin,
                                   target_key._begin),
         make_permutation_iterator(block_min_source_longest_path._begin,
                                   target_key._end),
         min_source_longest_path._begin)


def step6(DeviceVectorViewInt32 source_key,
          DeviceVectorViewInt32 target_key,
          DeviceVectorViewUint8 sync_source,
          DeviceVectorViewUint8 sync_target,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 min_source_longest_path,
          DeviceVectorViewFloat32 source_longest_path):
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


def step7(DeviceVectorViewFloat32 delay, size_t ready_connection_count):
    # Equivalent to:
    #
    #     connections[:ready_connection_count].delay = 1.
    fill_n(delay._begin, ready_connection_count, 1)


def step8(DeviceVectorViewFloat32 delay,
          DeviceVectorViewFloat32 target_longest_path,
          DeviceVectorViewFloat32 source_longest_path,
          size_t ready_connection_count):
    cdef plus[float] plus_f
    # Equivalent to:
    #
    #     connections[:ready_connection_count].target_longest_path = \
    #         (connections[:ready_connection_count].delay +
    #          connections[:ready_connection_count].source_longest_path)
    transform2(delay._begin, delay._begin + ready_connection_count,
               source_longest_path._begin, target_longest_path._begin, plus_f)


def step9(DeviceVectorViewInt32 target_key,
          DeviceVectorViewFloat32 target_longest_path,
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


def step10(DeviceVectorViewFloat32 longest_paths,
           DeviceVectorViewFloat32 max_target_longest_path,
           DeviceVectorViewInt32 reduced_keys, size_t resolved_block_count):
    # Equivalent to:
    #
    #     longest_paths[max_target_longest_path.index] = max_target_longest_path.values
    #copy(max_target_longest_path._begin, max_target_longest_path._begin +
         #resolved_block_count, make_permutation_iterator(longest_paths._begin,
                                                         #reduced_keys._begin))
    scatter(max_target_longest_path._begin, max_target_longest_path._begin +
            resolved_block_count, reduced_keys._begin, longest_paths._vector.begin())


def scatter_longest_paths(DeviceVectorFloat32 longest_paths,
                          DeviceVectorViewFloat32 max_target_longest_path,
                          DeviceVectorViewInt32 reduced_keys,
                          size_t resolved_block_count):
    # Equivalent to:
    #
    #     longest_paths[max_target_longest_path.index] = max_target_longest_path.values
    #copy(max_target_longest_path._begin, max_target_longest_path._begin +
         #resolved_block_count, make_permutation_iterator(longest_paths._begin,
                                                         #reduced_keys._begin))
    scatter(max_target_longest_path._begin, max_target_longest_path._begin +
            resolved_block_count, reduced_keys._begin, longest_paths._vector.begin())


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


cpdef look_up_delay(DeviceVectorViewInt32 source_key,
                    DeviceVectorViewInt32 target_key,
                    DeviceVectorViewUint8 delay_type,
                    DeviceVectorViewInt32 p_x, DeviceVectorViewInt32 p_y,
                    DeviceVectorViewFloat32 arch_delays,
                    int32_t nrows, int32_t ncols,
                    DeviceVectorViewFloat32 delay):
    cdef size_t count = source_key.size
    cdef absolute[int32_t] abs_func
    cdef minus[int32_t] minus_func
    cdef unpack_binary_args[minus[int32_t]] *unpacked_minus = \
        new unpack_binary_args[minus[int32_t]](minus_func)
    cdef timing_delay[device_vector[float].iterator] *delay_f = \
        new timing_delay[device_vector[float].iterator](arch_delays._begin,
                                                        nrows, ncols)
    cdef unpack_ternary_args[timing_delay[device_vector[float].iterator]] \
        *unpacked_delay = new \
        unpack_ternary_args[timing_delay[device_vector[float].iterator]]\
        (deref(delay_f))

    # Compute `delay`
    # Equivalent to:
    #
    #     delay = timing_delay(np.abs(p_x[source_key] - p_x[target_key]),
    #                          np.abs(p_y[source_key] - p_y[target_key]))
    copy_n(
        make_transform_iterator(
            make_zip_iterator(
                make_tuple3(
                    delay_type._begin,
                    make_transform_iterator(
                        make_transform_iterator(
                            make_zip_iterator(
                                make_tuple2(
                                    make_permutation_iterator(
                                        p_x._begin,
                                        source_key._begin),
                                    make_permutation_iterator(
                                        p_x._begin,
                                        target_key._begin))),
                            deref(unpacked_minus)), abs_func),
                    make_transform_iterator(
                        make_transform_iterator(
                            make_zip_iterator(
                                make_tuple2(
                                    make_permutation_iterator(
                                        p_y._begin,
                                        source_key._begin),
                                    make_permutation_iterator(
                                        p_y._begin,
                                        target_key._begin))),
                            deref(unpacked_minus)), abs_func))),
            deref(unpacked_delay)), count, delay._begin)
    del unpacked_minus
    del delay_f
    del unpacked_delay


cpdef look_up_delay_prime(DeviceVectorViewInt32 source_key,
                    DeviceVectorViewInt32 target_key,
                    DeviceVectorViewUint8 delay_type,
                    DeviceVectorViewInt32 p_x, DeviceVectorViewInt32 p_y,
                    DeviceVectorViewInt32 p_x_prime, DeviceVectorViewInt32 p_y_prime,
                    DeviceVectorViewFloat32 arch_delays,
                    int32_t nrows, int32_t ncols,
                    DeviceVectorViewFloat32 delay):
    cdef size_t count = source_key.size
    cdef absolute[int32_t] abs_func
    cdef minus[int32_t] minus_func
    cdef unpack_binary_args[minus[int32_t]] *unpacked_minus = \
        new unpack_binary_args[minus[int32_t]](minus_func)
    cdef timing_delay[device_vector[float].iterator] *delay_f = \
        new timing_delay[device_vector[float].iterator](arch_delays._begin,
                                                        nrows, ncols)
    cdef unpack_ternary_args[timing_delay[device_vector[float].iterator]] \
        *unpacked_delay = new \
        unpack_ternary_args[timing_delay[device_vector[float].iterator]]\
        (deref(delay_f))

    # Compute `delay`
    # Equivalent to:
    #
    #     delay = timing_delay(np.abs(p_x[source_key] - p_x_prime[target_key]),
    #                          np.abs(p_y[source_key] - p_y_prime[target_key]))
    copy_n(
        make_transform_iterator(
            make_zip_iterator(
                make_tuple3(
                    delay_type._begin,
                    make_transform_iterator(
                        make_transform_iterator(
                            make_zip_iterator(
                                make_tuple2(
                                    make_permutation_iterator(
                                        p_x._begin,
                                        source_key._begin),
                                    make_permutation_iterator(
                                        p_x_prime._begin,
                                        target_key._begin))),
                            deref(unpacked_minus)), abs_func),
                    make_transform_iterator(
                        make_transform_iterator(
                            make_zip_iterator(
                                make_tuple2(
                                    make_permutation_iterator(
                                        p_y._begin,
                                        source_key._begin),
                                    make_permutation_iterator(
                                        p_y_prime._begin,
                                        target_key._begin))),
                            deref(unpacked_minus)), abs_func))),
            deref(unpacked_delay)), count, delay._begin)
    del unpacked_minus
    del delay_f
    del unpacked_delay


def connection_criticality(DeviceVectorViewFloat32 delay,
                           DeviceVectorViewFloat32 arrival_times,
                           DeviceVectorViewFloat32 departure_times,
                           DeviceVectorViewInt32 driver_key,
                           DeviceVectorViewInt32 sink_key,
                           DeviceVectorViewFloat32 criticality):
    cdef maximum[float] maximum_f

    cdef float critical_path = deref(max_element(arrival_times._begin,
                                                 arrival_times._end))

    cdef c_connection_criticality[float] *connection_criticality_f = \
        new c_connection_criticality[float](critical_path)
    cdef unpack_ternary_args[c_connection_criticality[float]] \
        *unpacked_connection_criticality = \
        new unpack_ternary_args[c_connection_criticality[float]] \
        (deref(connection_criticality_f))

    # Equivalent to:
    #
    #     a.v['criticality'][:] = ((self._arrival_times[a['source_key'].values] +
    #                                 a['delay'] +
    #                                 self._departure_times[a['target_key'].values])
    #                                 / self.critical_path)
    copy_n(
        make_transform_iterator(
            make_zip_iterator(
                make_tuple3(
                    make_permutation_iterator(arrival_times._begin,
                                              driver_key._begin),
                    delay._begin,
                    make_permutation_iterator(departure_times._begin,
                                              sink_key._begin))),
            deref(unpacked_connection_criticality)), delay._vector.size(),
        criticality._begin)

    return critical_path


def connection_cost(float criticality_exp,
                    DeviceVectorViewFloat32 delay,
                    DeviceVectorViewFloat32 arrival_times,
                    DeviceVectorViewFloat32 departure_times,
                    DeviceVectorViewInt32 driver_key,
                    DeviceVectorViewInt32 sink_key,
                    DeviceVectorViewFloat32 cost, 
                    float critical_path=-1):
    '''
    Compute the cost of each connection.  The result is equivalent to the
    following:

        criticality = ((arrival_times[target_key] + delay +
                        departure_times[source_key]) / critical_path)
        cost = ((criticality ** criticality_exp) * delay)

    where:

     - `arrival_times`, `delay`, and `departure_times` are arrays, with one
       entry per connection.
     - `criticality_exp` is the maximum delay of any path in the circuit.  This
       should be equivalent to the maximum arrival _(or departure)_ time.
     - `criticality_exp` is a normalizing exponent term.  The higher the
       exponent, the lower the resulting calculation, since the value under the
       exponent is always less than or equal to one.
    '''
    cdef maximum[float] maximum_f

    if critical_path < 0:
        critical_path = deref(max_element(arrival_times._begin,
                                          arrival_times._end))

    cdef c_connection_cost[float] *connection_cost_f = \
        new c_connection_cost[float](critical_path, criticality_exp)
    cdef unpack_ternary_args[c_connection_cost[float]] \
        *unpacked_connection_cost = \
        new unpack_ternary_args[c_connection_cost[float]] \
        (deref(connection_cost_f))

    # Equivalent to:
    #
    #     a.v['criticality'][:] = ((self._arrival_times[a['source_key'].values] +
    #                                 a['delay'] +
    #                                 self._departure_times[a['target_key'].values])
    #                                 / self.critical_path)
    copy_n(
        make_transform_iterator(
            make_zip_iterator(
                make_tuple3(
                    make_permutation_iterator(arrival_times._begin,
                                              driver_key._begin),
                    delay._begin,
                    make_permutation_iterator(departure_times._begin,
                                              sink_key._begin))),
            deref(unpacked_connection_cost)), delay._vector.size(),
        cost._begin)

    return critical_path


def block_delta_timing_cost(DeviceVectorViewInt32 arrival_target_key,
                            DeviceVectorViewFloat32 arrival_cost,
                            DeviceVectorViewInt32 departure_target_key,
                            DeviceVectorViewFloat32 departure_cost,
                            DeviceVectorViewInt32 arrival_reduced_keys,
                            DeviceVectorViewInt32 departure_reduced_keys,
                            DeviceVectorViewFloat32
                            arrival_reduced_target_cost,
                            DeviceVectorViewFloat32
                            departure_reduced_target_cost,
                            DeviceVectorViewFloat32
                            block_arrival_cost,
                            DeviceVectorViewFloat32
                            block_departure_cost):
    cdef size_t arrival_block_count = (
        <device_vector[int32_t].iterator>
        accumulate_by_key(arrival_target_key._begin, arrival_target_key._end,
                          arrival_cost._begin, arrival_reduced_keys._begin,
                          arrival_reduced_target_cost._begin).first -
        arrival_reduced_keys._begin)

    cdef size_t departure_block_count = (
        <device_vector[int32_t].iterator>
        accumulate_by_key(departure_target_key._begin,
                          departure_target_key._end, departure_cost._begin,
                          departure_reduced_keys._begin,
                          departure_reduced_target_cost._begin).first -
        departure_reduced_keys._begin)

    copy_n(
        arrival_reduced_target_cost._begin, arrival_block_count,
        make_permutation_iterator(block_arrival_cost._begin,
                                  arrival_reduced_keys._begin))
    copy_n(
        departure_reduced_target_cost._begin, departure_block_count,
        make_permutation_iterator(block_departure_cost._begin,
                                  departure_reduced_keys._begin))

    return arrival_block_count, departure_block_count
