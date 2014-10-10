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


@profile
def _update_arrival_times_thrust(arrival_connections, arrival_times):
    print 'using thrust'
    block_keys = DeviceVectorInt32.from_array(arrival_connections
                                              ['block_key'].as_matrix())
    reduced_block_keys = DeviceVectorInt32(block_keys.size)
    d_arrival_times = DeviceVectorFloat32.from_array(arrival_connections
                                                     ['arrival_time']
                                                     .as_matrix())
    min_arrivals = DeviceVectorFloat32(d_arrival_times.size)
    max_arrivals = DeviceVectorFloat32(d_arrival_times.size)

    sort_float32_by_int32_key(block_keys, d_arrival_times)
    N_ = minmax_float32_by_key(block_keys, d_arrival_times, reduced_block_keys,
                               min_arrivals, max_arrivals)

    # new code
    d_block_arrival_times = arrival_times
    d_idx = DeviceVectorInt32(len(arrival_connections))
    sequence_int32(d_idx)

    block_count = partition_n_int32_float32_stencil_non_negative(
        d_idx, min_arrivals, N_)

    d_ready_block_keys = DeviceVectorInt32(block_count)
    permute_n_int32(reduced_block_keys, d_idx, block_count, d_ready_block_keys)

    d_ready_block_arrival_times = DeviceVectorFloat32(block_count)
    permute_float32(d_block_arrival_times, d_ready_block_keys,
                    d_ready_block_arrival_times)

    unresolved_count = partition_int32_float32_stencil_negative(
        d_ready_block_keys, d_ready_block_arrival_times)

    #sequence_int32(d_idx)
    #N = partition_n_int32_float32_stencil_non_negative(d_idx, min_arrivals,
                                                       #block_count)

    #arrival_times[d_ready_block_keys[:unresolved_count]] = \
        #d_max_arrivals[:][d_idx[:N]]
    pass
    #unresolved_block_keys = ready_block_keys[arrival_times
                                             #[ready_block_keys] < 0]
    unresolved_block_keys = d_ready_block_keys[:unresolved_count]

    temp = arrival_times[:]
    temp[unresolved_block_keys] = max_arrivals[:N_][min_arrivals[:N_] >= 0]
    arrival_times[:] = temp


class DelayModel(object):
    @profile
    def __init__(self, adjacency_list, delay=unit_delay):
        self.adjacency_list = adjacency_list
        self.net_drivers = (adjacency_list.driver_connections.drop_duplicates()
                            .block_key.values)
        self.delay = delay
        #self.arrival_times = np.empty(adjacency_list.block_count, dtype='f32')
        self.arrival_times = DeviceVectorFloat32(adjacency_list.block_count)
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
        self.arrival_times[:] = -1
        arrival_times = self.arrival_times[:]
        arrival_times[adjacency_list.clocked_driver_block_keys] = 0.
        arrival_times[self.global_block_keys.tolist()] = 0.
        self.arrival_times[:] = arrival_times
        del arrival_times
        self.max_arrival_time = -1

        # Assign an initial required-time of "infinity" to all blocks.
        self.required_times[:] = 1e7
        self.required_times[self.global_block_keys.tolist()] = 0.

        block_keys = DeviceVectorInt32.from_array(adjacency_list.connections
                                                  .loc[adjacency_list
                                                       .connections
                                                       .block_type_key ==
                                                       0].block_key
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

        arrival_times = self.arrival_times[:]
        arrival_times[single_connection_blocks] = 0.
        self.arrival_times[:] = arrival_times
        del arrival_times
        self.required_times[single_connection_blocks] = 0.

        self.delay_connections = adjacency_list.sink_connections.copy()
        self.delay_connections['net_driver_block_key'] = (
            self.net_drivers[self.delay_connections['net_key'].tolist()])
        # Filter connections to only include connections that are not driving
        # the associated net.
        #
        # TODO: This might be redundant, since I think this should already be
        # taken care of by initializing the `delay_connections` from the
        # `sink_connections` of the `adjacency_list`.
        self.delay_connections = (self.delay_connections
                                  [self.delay_connections['block_key'] !=
                                   self.delay_connections
                                   ['net_driver_block_key']])
        # We compute the delays here, since with unit delay, the delays are
        # static and only need to be computed once.
        self.delay_connections['delay'] = 1.

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
        d_block_arrival_times = self.arrival_times
        d_net_driver_block_key = DeviceVectorInt32.from_array(
            arrival_connections.net_driver_block_key.values)
        d_driver_arrival_time = DeviceVectorFloat32(len(arrival_connections))

        # For each connection, gather the arrival time of the driver of the
        # corresponding net.
        permute_float32(d_block_arrival_times, d_net_driver_block_key,
                        d_driver_arrival_time)

        # Create mask, defining which connections are _not_ ready for
        # reduction. __NB__ A connection is ready for reduction once the
        # arrival time is available for the block that is driving the
        # respective net.
        d_idx = DeviceVectorInt32(len(arrival_connections))
        sequence_int32(d_idx)

        N = partition_int32_float32_stencil_non_negative(d_idx,
                                                         d_driver_arrival_time)
        d_delay = DeviceVectorFloat32.from_array(arrival_connections.delay
                                                 .values)
        # ready_to_calculate = d_idx[N:]
        not_ready_to_calculate = d_idx[N:]

        d_arrival_times = DeviceVectorFloat32(len(arrival_connections))
        d_arrival_times[:] = -1
        sum_n_float32_float32_stencil(N, d_idx, d_driver_arrival_time, d_delay,
                                      d_arrival_times)

        # For each connection where the driver arrival time has been resolved,
        # compute the arrival time of the corresponding block, based on the
        # arrival time of the driver and the delay between the driver block and
        # the connection block.
        arrival_connections['arrival_time'] = d_arrival_times[:]

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
            _update_arrival_times_thrust(arrival_connections,
                                         self.arrival_times)
        else:
            _update_arrival_times_pandas(arrival_connections,
                                         self.arrival_times)

        # TODO: Migrate away from using Pandas table for arrival_connections,
        # since the line below contributes ~75% of the run-time of this method.
        return arrival_connections.loc[arrival_connections
                                       .index[not_ready_to_calculate]]

    @profile
    def compute_arrival_times(self, name, use_thrust=True):
        previous_connection_count = None
        arrival_connections = self.delay_connections.copy()
        connection_count = arrival_connections.shape[0]

        i = 0
        while previous_connection_count != connection_count:
            connection_count = arrival_connections.shape[0]
            arrival_connections = self._compute_arrival_times(
                arrival_connections, use_thrust)
            previous_connection_count = arrival_connections.shape[0]
            cached = path('%s-i%d-arrival_times.pickled' % (name, i))
            if use_thrust and cached.isfile():
                data = pickle.load(cached.open('rb'))
                if not (data == self.arrival_times[:]).all():
                    import pudb; pudb.set_trace()
                print ' verified %d' % i
            elif not use_thrust:
                pd.Series(self.arrival_times).to_pickle(cached)
                print 'wrote cached arrival times to:', cached
            i += 1
        self.max_arrival_time = self.arrival_times[:].max()
        return self.arrival_times[:]

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
