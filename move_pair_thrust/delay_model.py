from itertools import izip
import cPickle as pickle

import numpy as np
import pandas as pd
from path_helpers import path
from cythrust.device_vector import DeviceVectorInt32, DeviceVectorFloat32
from cythrust.device_vector.extrema import minmax_float32_by_key
from cythrust.device_vector.sort import sort_float32_by_int32_key, sort_int32
from cythrust.device_vector.count import count_int32_key
from cythrust.device_vector.sum import sum_n_float32_float32_stencil
from cythrust.device_vector.partition import (
    partition_int32_float32_stencil_non_negative,
    partition_int32_float32_stencil_negative,
    partition_n_int32_float32_stencil_non_negative,
    partition_n_offset_int32_float32_stencil_non_negative)
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


@profile
def _update_arrival_times_pandas(arrival_connections, arrival_times):
    print 'using pandas'
    min_max_arrival = (arrival_connections.groupby('block_key')
                       .agg({'arrival_time': lambda x: (np.min(x),
                                                        np.max(x))}))

    for block_key, ((min_arrival, max_arrival), ) in (min_max_arrival
                                                      .iterrows()):
        if min_arrival >= 0 and arrival_times[block_key] < 0:
            # Minimum connection arrival-time for the block with key,
            # `block_key`, is zero or positive.  Therefore, the maximum
            # arrival-time is valid, and corresponds to the arrival-time of
            # the block.  Update the arrival-time for the block in the
            # `arrival_times` array.
            arrival_times[block_key] = max_arrival


class TempTimingData(object):
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


class PartitionTimingData(object):
    def __init__(self, block_count, clocked_driver_block_keys,
                 global_block_keys, single_connection_blocks):
        self.lengths = []

        self.block_arrival_times = DeviceVectorFloat32(block_count)
        fill_arrival_times(clocked_driver_block_keys, global_block_keys,
                           single_connection_blocks, self.block_arrival_times)

    def set_connection_data(self, arrival_connections):
        self.connection_count = len(arrival_connections)
        self.block_keys = DeviceVectorInt32.from_array(arrival_connections
                                                       .block_key.values)
        # `_compute_arrival_times`
        self.net_driver_block_key = DeviceVectorInt32.from_array(
            arrival_connections.net_driver_block_key.values)
        # `_update_arrival_times_thrust`
        # `_compute_arrival_times`

    def partition_thrust(self, N, driver_arrival_time):
        move_unresolved_data_to_front(self.block_keys,
                                      self.net_driver_block_key,
                                      driver_arrival_time)

    def partition_numpy(self, driver_arrival_time):
        #self.lengths.append(N)
        ready_to_calculate = (driver_arrival_time[:] >= 0)
        not_ready_count = driver_arrival_time.size - ready_to_calculate.sum()
        block_keys = np.empty(self.block_keys.size, dtype=np.int32)
        net_driver_block_key = np.empty_like(block_keys)
        block_keys[:not_ready_count] = self.block_keys[:][~ready_to_calculate]
        block_keys[not_ready_count:] = self.block_keys[:][ready_to_calculate]
        net_driver_block_key[:not_ready_count] = \
            self.net_driver_block_key[:][~ready_to_calculate]
        net_driver_block_key[not_ready_count:] = \
            self.net_driver_block_key[:][ready_to_calculate]
        return block_keys, net_driver_block_key

    @property
    def block_count(self):
        return self.block_arrival_times.size

    def compute_connection_arrival_times(self):
        vectors = TempTimingData(self.connection_count, self.block_count)

        # For each connection, gather the arrival time of the driver of the
        # corresponding net.
        #
        # Equivalent to:
        #
        #     vectors.driver_arrival_time = self.block_arrival_times[self.net_driver_block_key]
        permute_float32(self.block_arrival_times, self.net_driver_block_key,
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

    @property
    def size(self):
        return self.lengths[-1]


@profile
def _update_arrival_times_thrust(timing_data, vectors):
    d_block_keys = DeviceVectorInt32.from_array(timing_data
                                                .block_keys[:])
    sort_float32_by_int32_key(d_block_keys,
                              vectors.connection_arrival_time)
    N_ = minmax_float32_by_key(d_block_keys,
                               vectors.connection_arrival_time,
                               vectors.reduced_block_keys,
                               vectors.min_arrivals, vectors.max_arrivals)

    # new code
    d_block_arrival_times = timing_data.block_arrival_times
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
        self.net_drivers = (adjacency_list.driver_connections.drop_duplicates()
                            .block_key.values)
        self.delay = delay
        self.block_count = adjacency_list.block_count
        #self.arrival_times = np.empty(adjacency_list.block_count, dtype='f32')
        self.required_times = np.empty(adjacency_list.block_count, dtype='f32')
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

        # Assign an initial required-time of "infinity" to all blocks.
        self.required_times[:] = 1e7
        self.required_times[self.global_block_keys.tolist()] = 0.

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

        self.required_times[single_connection_blocks] = 0.

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
        self.delay_connections['delay'] = 1.
        #self.connections = ConnectionTable.from_dataframe(
            #self.arrival_times.size, self.delay_connections)

    @profile
    def compute_connection_delays(self):
        '''
        Update the delay of each connection.

        __NB__ For now, we are assuming a unit delay model, where each level of
        logic adds a unit delay.  However, we eventually want to use the
        current position of each block in the net-list to influence the
        associated delay, similar to [T-VPlace][1].

        [1]: http://dx.doi.org/10.1145/329166.329208
        '''
        return [self.delay(a, b) for a, b in
                izip(self.delay_connections['block_key'],
                     self.delay_connections
                     ['net_driver_block_key'])]

    @profile
    def _compute_arrival_times(self, arrival_connections, use_thrust=True):
        self.timing_data.set_connection_data(arrival_connections)
        ready_count, vectors = (self.timing_data
                                .compute_connection_arrival_times())
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
        if use_thrust:
            print 'using thrust: len(arrival_connections) = ', \
                len(arrival_connections)
            _update_arrival_times_thrust(self.timing_data, vectors)

        # TODO: Migrate away from using Pandas table for arrival_connections,
        # since the line below contributes ~75% of the run-time of this method.
        new_connections = arrival_connections.loc[arrival_connections.index
                                                  [not_ready_to_calculate]]
        #block_keys_, net_driver_block_key_ = self.timing_data.partition_numpy(
            #vectors.driver_arrival_time)
        self.timing_data.partition_thrust(ready_count,
                                          vectors.driver_arrival_time)
        assert(len(set(self.timing_data.net_driver_block_key
                       [:not_ready_to_calculate.size]) -
                   set(new_connections.net_driver_block_key)) == 0)
        assert(len(set(self.timing_data.block_keys
                       [:not_ready_to_calculate.size]) -
                   set(new_connections.block_key)) == 0)
        return new_connections

    @profile
    def compute_arrival_times(self, name, use_thrust=True):
        previous_connection_count = None
        arrival_connections = self.delay_connections.copy()
        connection_count = arrival_connections.shape[0]
        self.timing_data = PartitionTimingData(self.block_count,
                                               self.clocked_driver_block_keys,
                                               self.d_global_block_keys,
                                               self.d_single_connection_blocks)

        i = 0
        while previous_connection_count != connection_count:
            connection_count = arrival_connections.shape[0]
            arrival_connections = self._compute_arrival_times(
                arrival_connections, use_thrust)
            previous_connection_count = arrival_connections.shape[0]
            cached = path('%s-i%d-arrival_times.pickled' % (name, i))
            if use_thrust and cached.isfile():
                data = pickle.load(cached.open('rb'))
                if not (data == self.timing_data
                        .block_arrival_times[:]).all():
                    import pudb; pudb.set_trace()
                else:
                    print ' verified %d' % i
            elif not use_thrust:
                #pd.Series(self.arrival_times).to_pickle(cached)
                #print 'wrote cached arrival times to:', cached
                pass
            i += 1
        self.max_arrival_time = self.timing_data.block_arrival_times[:].max()
        return self.timing_data.block_arrival_times[:]

    @profile
    def compute_required_times(self):
        if self.max_arrival_time < 0:
            raise RuntimeError('Arrival times must be calculated first.')
        self.required_times[:] = 1e7
        self.required_times[self.adjacency_list
                            .clocked_sink_block_keys] = self.max_arrival_time
        self.required_times[self.global_block_keys.tolist()] = 0.
        previous_connection_count = None
        required_connections = self.delay_connections.copy()
        connection_count = required_connections.shape[0]

        while previous_connection_count != connection_count:
            connection_count = required_connections.shape[0]
            required_connections = self._compute_required_times(
                required_connections)
            previous_connection_count = required_connections.shape[0]
        return self.required_times, required_connections

    @profile
    def _compute_required_times(self, required_connections):
        required_connections['sink_required_time'] = (
            self.required_times[required_connections['block_key'].tolist()])

        # display(required_delays.arrival_connections)
        ready_to_calculate = (required_connections['sink_required_time'] < 1e7)
        required_connections['required_time'] = 1e7
        required_connections['required_time'][ready_to_calculate] = (
            required_connections['sink_required_time'][ready_to_calculate] -
            required_connections['delay'][ready_to_calculate])
        min_max_required = (required_connections.groupby
                            ('net_driver_block_key')
                            .agg({'required_time': lambda x: (np.min(x),
                                                              np.max(x))}))
        for driver_block_key, ((min_required,
                                max_required), ) in (min_max_required
                                                     .iterrows()):
            if max_required < 1e7:
                # Maximum connection required-time for the block with key,
                # `block_key`, is less than "infinity", so the minimum-required
                # time is valid, and corresponds to the required-time of the
                # block.  Update the required-time for the block in the
                # `required_times` array.
                self.required_times[driver_block_key] = min_required
        return required_connections[~ready_to_calculate].copy()
