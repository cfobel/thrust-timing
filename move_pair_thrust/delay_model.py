import numpy as np
from cythrust.device_vector import DeviceVectorInt32, DeviceVectorFloat32
from cythrust.device_vector.extrema import minmax_float32_by_key
from cythrust.device_vector.sort import sort_float32_by_int32_key, sort_int32
from cythrust.device_vector.count import count_int32_key
from cythrust.device_vector.sum import sum_n_float32_float32_stencil
from cythrust.device_vector.partition import (
    partition_int32_float32_stencil_non_negative,
    partition_int32_float32_stencil_negative,
    partition_n_int32_float32_stencil_non_negative)
from cythrust.device_vector.copy import permute_float32, permute_n_int32
from camip.device.CAMIP import sequence_int32
from .DELAY_MODEL import (fill_arrival_times, resolve_block_arrival_times,
                          move_unresolved_data_to_front)


try:
    profile
except:
    profile = lambda (f): f


def unit_delay(block_key_a, block_key_b):
    '''
    Unit delay for now.

    Notes
    =====

    This should likely be replaced with a position-based delay model,
    as used in [T-VPlace][1].

    [1]: http://dx.doi.org/10.1145/329166.329208
    '''
    return 1


class LevelTimingData(object):
    '''
    Vectors/arrays of per-connection data for a computing single level of
    arrival time calculations.

    The values are computed from scratch at each level _(i.e., no values
    persist from one iteration to the next)_.


    TODO
    ====

     - For the sake of efficiency, prevent the need to reallocate a new
       instance of this class for each level iteration when computing arrival
       times.
      - This can be done by allocating once with the connection count and the
        block count at their respective max values, and keeping a count to mark
        the "end" of the vectors for the current iteration.
    '''
    def __init__(self, connection_count, block_count):
        self.reduced_block_keys = DeviceVectorInt32(connection_count)
        self.min_arrivals = DeviceVectorFloat32(connection_count)
        self.max_arrivals = DeviceVectorFloat32(connection_count)
        self.d_idx = DeviceVectorInt32(connection_count)
        self.d_ready_block_keys = DeviceVectorInt32(block_count)
        self.d_ready_block_arrival_times = DeviceVectorFloat32(block_count)
        self.block_keys = DeviceVectorInt32(connection_count)

        self.driver_arrival_time = DeviceVectorFloat32(connection_count)
        self.delay = DeviceVectorFloat32(connection_count)
        self.connection_arrival_time = DeviceVectorFloat32(connection_count)


class UnresolvedConnectionEndpoints(object):
    '''
    Vectors/arrays of per-connection data for all unresolved connections in a
    net-list.

    The unresolved values in the arrays persist from one iteration to the next
    when computing arrival times.


    Notes
    =====

     - The `partition` method reorders all arrays at the same time, such that
       all entries for unresolved connection arrival times _(i.e., where
       `driver_arrival_time < 0`)_ are packed to the front of the arrays.
      - This makes it possible to only process unresolved connections during
        each level iteration of arrival times calculation.


    TODO
    ====

     - Update code using the data from this class to use the `connection_count`
       to mark the end of the vectors, rather than using the `end()` method of
       the vector.  This would:
      - Improve run-time performance, since the arrays would not need to be
        reallocated on every call to `partition`.
      - All data could remain in the arrays, but just reordered.  After one
        complete pass through arrival time calculations, this would result in
        the arrays in an order that could be sliced _(but not reordered)_
        to access all connections for a particular delay "level".
       - __NB__ This might not be that important, since the same thing could
         easily be accomplished by simply sorting the arrays by the unit
         arrival times _(i.e., "delay levels")_ after computing the arrival
         times according to each connection having a delay of 1.
    '''
    def __init__(self, arrival_connections):
        self.set_connection_data(arrival_connections)

    def set_connection_data(self, arrival_connections):
        '''
        Load initial connection data from a `pandas.DataFrame` instance with the following
        columns:

         - `block_key`: The key of the sink block of each connection.
         - `net_driver_block_key`: The key of the driver block of each connection.
        '''
        self.connection_count = len(arrival_connections)
        self.block_keys = DeviceVectorInt32.from_array(arrival_connections
                                                       .block_key.values)
        self.net_driver_block_key = DeviceVectorInt32.from_array(
            arrival_connections.net_driver_block_key.values)

    def partition_thrust(self, N, driver_arrival_time):
        '''
        Reorder all arrays at the same time, such that all entries for
        unresolved connection arrival times _(i.e., where `driver_arrival_time
        < 0`)_ are packed to the front of the arrays.

        This makes it possible to only process unresolved connections during
        each level iteration of arrival times calculation.
        '''
        move_unresolved_data_to_front(self.block_keys,
                                      self.net_driver_block_key,
                                      driver_arrival_time)
        self.connection_count = N
        self.block_keys = DeviceVectorInt32.from_array(self.block_keys[:N])
        self.net_driver_block_key = DeviceVectorInt32.from_array(
            self.net_driver_block_key[:N])

    def partition_numpy(self, not_ready_count, driver_arrival_time):
        '''
        `numpy`-based equivalent to the `partition_thrust` method.
        '''
        ready_to_calculate = (driver_arrival_time[:] >= 0)
        self.connection_count = not_ready_count
        self.block_keys = DeviceVectorInt32.from_array(
            self.block_keys[:][~ready_to_calculate])
        self.net_driver_block_key = DeviceVectorInt32.from_array(
            self.net_driver_block_key[:][~ready_to_calculate])

    def compute_connection_arrival_times(self, block_arrival_times):
        vectors = LevelTimingData(self.connection_count,
                                  block_arrival_times.size)

        # For each connection, gather the arrival time of the driver of the
        # corresponding net.
        #
        # Equivalent to:
        #
        #     vectors.driver_arrival_time = block_arrival_times[self.net_driver_block_key]
        permute_float32(block_arrival_times, self.net_driver_block_key,
                        vectors.driver_arrival_time)

        # Create mask, d_fining which connections are _not_ ready for
        # reduction. __NB__ A connection is ready for reduction once the
        # arrival time is available for the block that is d_iving the
        # respective net.
        sequence_int32(vectors.d_idx)

        # Pack the list of indexes of all "ready" connections to the beginning
        # of the `d_idx` vector.
        #
        # Equivalent to performing the following two lines at the same time:
        #
        #      ready = (vectors.driver_arrival_time >= 0)
        #      d_idx[:len(ready)] = d_idx[ready]
        #      d_idx[len(ready):] = d_idx[~ready]
        #
        # where `ready` is a list of connection indexes corresponding to
        # connections with resolved arrival times.
        ready_count = partition_int32_float32_stencil_non_negative(
            vectors.d_idx, vectors.driver_arrival_time)

        # ready_to_calculate = d_idx[N:]
        # not_ready_to_calculate = vectors.d_idx[N:]

        # Load the delay of each connection.
        # TODO: Once the timing calculations are working, use position-based
        # delay.
        vectors.delay[:] = 1.

        # Start by marking all connection arrival times as _unresolved_.
        vectors.connection_arrival_time[:] = -1

        # For each connection sourced by a driver block with a _resolved_
        # arrival time:
        #
        #  1. Add the arrival time of the driver block to the delay of the
        #     connection, storing the result .
        #
        # Equivalent to:
        #
        #      vectors.connection_arrival_time[ready_to_calculate] = \
        #          (vectors.driver_arrival_time[ready_to_calculate] +
        #           vectors.delay[ready_to_calculate])
        #
        # where:
        #
        #      ready_to_calculate = d_idx[N:]
        #
        # __NB__ Any connection _not_ included in `ready_to_calculate` will
        # hold a value of -1 to mark it as _unresolved_ for the current
        # iteration.
        sum_n_float32_float32_stencil(ready_count, vectors.d_idx,
                                      vectors.driver_arrival_time,
                                      vectors.delay,
                                      vectors.connection_arrival_time)
        return ready_count, vectors


@profile
def _update_arrival_times_thrust(timing_data, vectors, block_arrival_times):
    d_block_keys = DeviceVectorInt32.from_array(timing_data
                                                .block_keys[:])
    sort_float32_by_int32_key(d_block_keys,
                              vectors.connection_arrival_time)
    N_ = minmax_float32_by_key(d_block_keys,
                               vectors.connection_arrival_time,
                               vectors.reduced_block_keys,
                               vectors.min_arrivals, vectors.max_arrivals)

    # new code
    d_block_arrival_times = block_arrival_times
    sequence_int32(vectors.d_idx)

    block_count = partition_n_int32_float32_stencil_non_negative(
        vectors.d_idx, vectors.min_arrivals, N_)

    permute_n_int32(vectors.reduced_block_keys, vectors.d_idx, block_count,
                    vectors.d_ready_block_keys)

    permute_float32(d_block_arrival_times, vectors.d_ready_block_keys,
                    vectors.d_ready_block_arrival_times)

    unresolved_count = partition_int32_float32_stencil_negative(
        vectors.d_ready_block_keys, vectors.d_ready_block_arrival_times)

    sequence_int32(vectors.d_idx)
    partition_n_int32_float32_stencil_non_negative(vectors.d_idx,
                                                   vectors.min_arrivals, N_)

    resolve_block_arrival_times(unresolved_count, vectors.max_arrivals,
                                vectors.d_idx, d_block_arrival_times,
                                vectors.d_ready_block_keys)


class DelayModel(object):
    @profile
    def __init__(self, adjacency_list, delay=unit_delay):
        self.adjacency_list = adjacency_list
        self.net_drivers = (adjacency_list.driver_connections.block_key
                            .unique())
        self.block_count = adjacency_list.block_count
        if adjacency_list.ignore_clock:
            self.global_block_keys = np.array([], dtype='uint32')
        else:
            self.global_block_keys = (self.net_drivers
                                      [adjacency_list.connections
                                       [adjacency_list.connections['pin_key']
                                        == 5]['net_key'].unique().tolist()])

        # Assign an arrival time of zero to all:
        #
        #  - Input blocks
        #  - Clocked logic-blocks
        #
        # Assign an arrival time of -1 to all other blocks.
        self.clocked_driver_block_keys = DeviceVectorInt32.from_array(
            adjacency_list.clocked_driver_block_keys)
        self.d_global_block_keys = DeviceVectorInt32.from_array(
            self.global_block_keys)

        block_keys = DeviceVectorInt32.from_array(
            adjacency_list.connections.loc[adjacency_list.connections
                                           .block_type_key == 0].block_key
            .as_matrix())
        sort_int32(block_keys)

        reduced_block_keys = DeviceVectorInt32(block_keys.size)
        block_net_counts = DeviceVectorInt32(block_keys.size)
        N = count_int32_key(block_keys, reduced_block_keys, block_net_counts)
        block_net_counts = block_net_counts[:N]
        reduced_block_keys = reduced_block_keys[:N]

        # ## Set arrival-time to zero for blocks with only one net ##
        #
        # Some net-lists _(e.g. `clma`)_ have at least one logic block that is
        # only connected to a single net.  For each such block, either:
        #
        #  - The block has no inputs, so must be a constant-generator.
        #  - The block has no output, so is equivalent to no block at all.
        #
        # In either case, the arrival-time of the block can be set to zero.
        single_connection_blocks = reduced_block_keys[block_net_counts < 2]
        self.d_single_connection_blocks = DeviceVectorInt32.from_array(
            single_connection_blocks)

        self.max_arrival_time = -1

        self.delay_connections = adjacency_list.sink_connections.sort('block_key')
        self.delay_connections['net_driver_block_key'] = (
            self.net_drivers[self.delay_connections['net_key'].values])

        # Filter connections to only include connections that are not driving
        # the associated net.
        #
        # TODO: This might be redundant, since I think this should already be
        # taken care of by initializing the `delay_connections` from the
        # `sink_connections` of the `adjacency_list`.
        self.delay_connections = (
            self.delay_connections[self.delay_connections.block_key !=
                                   self.delay_connections
                                   .net_driver_block_key])
        # We compute the delays here, since with unit delay, the delays are
        # static and only need to be computed once.
        self.delay_connections['delay'] = 2.
        #self.connections = ConnectionTable.from_dataframe(
            #self.arrival_times.size, self.delay_connections)

    @profile
    def _compute_arrival_times(self, block_arrival_times):
        ready_count, vectors = (self.timing_data
                                .compute_connection_arrival_times(
                                    block_arrival_times))
        not_ready_to_calculate = vectors.d_idx[ready_count:]

        # Reduce the computed connection arrival-times by block-key, to collect
        # the _minimum_ and _maximum_ arrival-time for each block.
        #
        # __NB__ Each _resolved_ arrival-time will have a value that is either
        # positive or equal to zero.  Each _unresolved_ arrival-time will have
        # a value of -1.  Therefore, by collecting the minimum arrival-time
        # from the connections corresponding to each block, we can determine if
        # all respective arrival-times have been resolved. If the reduced
        # minimum arrival-time of all connections corresponding to a block is
        # negative, there is at least one _unresolved_ arrival-time for the
        # respective block. However, if the reduced minimum arrival-time for a
        # block is either zero or positive, all corresponding connection driver
        # arrival-times must be resolved.  In this case, the reduced maximum
        # arrival-time represents the arrival-time for the corresponding block.
        _update_arrival_times_thrust(self.timing_data, vectors,
                                     block_arrival_times)

        self.timing_data.partition_thrust(not_ready_to_calculate.size,
                                          vectors.driver_arrival_time)

        # __NB__ The `numpy` equivalent partition method can be run as shown
        # below:
        #
        #     self.timing_data.partition_numpy(not_ready_to_calculate.size,
        #                                      vectors.driver_arrival_time)

    @profile
    def compute_arrival_times(self, name):
        previous_connection_count = None
        block_arrival_times = DeviceVectorFloat32(self.block_count)
        fill_arrival_times(self.clocked_driver_block_keys,
                           self.d_global_block_keys,
                           self.d_single_connection_blocks,
                           block_arrival_times)

        self.timing_data = UnresolvedConnectionEndpoints(self
                                                         .delay_connections)

        connection_count = self.timing_data.connection_count

        i = 0
        while previous_connection_count != connection_count:
            connection_count = self.timing_data.connection_count
            self._compute_arrival_times(block_arrival_times)
            previous_connection_count = self.timing_data.connection_count
            i += 1
        self.max_arrival_time = block_arrival_times[:].max()
        #import pudb; pudb.set_trace()
        return block_arrival_times[:]
