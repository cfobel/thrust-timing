from itertools import izip
from cythrust.device_vector import DeviceVectorInt32, DeviceVectorFloat32
from cythrust.device_vector.sort import sort_float32_by_int32_key
from cythrust.device_vector.extrema import minmax_float32_by_key
from cythrust.device_vector.sort import sort_float32_by_int32_key, sort_int32
from cythrust.device_vector.count import count_int32_key
import numpy as np


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
    N = minmax_float32_by_key(block_keys, d_arrival_times, reduced_block_keys,
                              min_arrivals, max_arrivals)

    ready_block_keys = reduced_block_keys[:N][(min_arrivals[:N] >= 0)]
    unresolved_block_keys = ready_block_keys[arrival_times
                                             [ready_block_keys] < 0]
    arrival_times[unresolved_block_keys] = \
        max_arrivals[:N][min_arrivals[:N] >= 0]


class DelayModel(object):
    @profile
    def __init__(self, adjacency_list, delay=unit_delay):
        self.adjacency_list = adjacency_list
        self.net_drivers = np.array(adjacency_list.driver_connections
                                    .drop_duplicates()['block_key'])
        self.delay = delay
        self.arrival_times = np.empty(adjacency_list.block_count, dtype='f32')
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
        self.arrival_times[adjacency_list.clocked_driver_block_keys] = 0.
        self.arrival_times[self.global_block_keys.tolist()] = 0.
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

        self.arrival_times[single_connection_blocks] = 0.
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
        self.delay_connections['delay'] = self.compute_connection_delays()

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
        # For each connection, gather the arrival time of the driver of the
        # corresponding net.
        arrival_connections['driver_arrival_time'] = (
            self.arrival_times[arrival_connections
                               ['net_driver_block_key'].tolist()])
        # Create mask, defining which connections are ready for reduction.
        # __NB__ A connection is ready for reduction once the arrival time is
        # available for the block that is driving the respective net.
        ready_to_calculate = (arrival_connections
                              ['driver_arrival_time'] >= 0)
        arrival_connections['arrival_time'] = -1
        # For each connection where the driver arrival time has been resolved,
        # compute the arrival time of the corresponding block, based on the
        # arrival time of the driver and the delay between the driver block and
        # the connection block.
        arrival_connections['arrival_time'][ready_to_calculate] = (
            arrival_connections['driver_arrival_time']
            [ready_to_calculate] +
            arrival_connections['delay'][ready_to_calculate])

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

        return arrival_connections[~ready_to_calculate].copy()

    @profile
    def compute_arrival_times(self, use_thrust=True):
        previous_connection_count = None
        arrival_connections = self.delay_connections.copy()
        connection_count = arrival_connections.shape[0]

        while previous_connection_count != connection_count:
            connection_count = arrival_connections.shape[0]
            arrival_connections = self._compute_arrival_times(
                arrival_connections, use_thrust)
            previous_connection_count = arrival_connections.shape[0]
        self.max_arrival_time = self.arrival_times.max()
        return self.arrival_times

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
