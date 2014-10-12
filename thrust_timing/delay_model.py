import numpy as np
import pandas as pd

from cyplace_experiments.data.connections_table import (CONNECTION_DRIVER,
                                                        CONNECTION_SINK)
from cythrust.device_vector import DeviceVectorInt32, DeviceVectorFloat32
from cythrust.device_vector.extrema import minmax_float32_by_key
from cythrust.device_vector.sort import sort_float32_by_int32_key
from cythrust.device_vector.sum import sum_n_float32_float32_stencil
from cythrust.device_vector.partition import (
    partition_int32_float32_stencil_non_negative,
    partition_int32_float32_stencil_negative,
    partition_n_int32_float32_stencil_non_negative)
from cythrust.device_vector.copy import permute_float32, permute_n_int32
from camip.device.CAMIP import sequence_int32
from .DELAY_MODEL import (fill_longest_paths, resolve_block_longest_paths,
                          move_unresolved_data_to_front)


try:
    profile
except:
    profile = lambda (f): f


class LevelConnectionData(object):
    '''
    Vectors/arrays of per-connection data for a computing single level of
    longest path calculations.

    The values are computed from scratch at each level _(i.e., no values
    persist from one iteration to the next)_.


    TODO
    ====

     - For the sake of efficiency, prevent the need to reallocate a new
       instance of this class for each level iteration when computing longest
       paths.
      - This can be done by allocating once with the connection count and the
        block count at their respective max values, and keeping a count to mark
        the "end" of the vectors for the current iteration.
    '''
    def __init__(self, connection_count, block_count):
        self.reduced_block_keys = DeviceVectorInt32(connection_count)
        self.min_longest_path = DeviceVectorFloat32(connection_count)
        self.max_longest_path = DeviceVectorFloat32(connection_count)
        self.d_idx = DeviceVectorInt32(connection_count)
        self.d_ready_block_keys = DeviceVectorInt32(block_count)
        self.d_ready_block_longest_paths = DeviceVectorFloat32(block_count)
        self.block_keys = DeviceVectorInt32(connection_count)

        self.source_longest_path = DeviceVectorFloat32(connection_count)
        self.delay = DeviceVectorFloat32(connection_count)
        self.connection_longest_path = DeviceVectorFloat32(connection_count)


class UnresolvedConnectionData(object):
    '''
    Vectors/arrays of per-connection data for all unresolved connections in a
    net-list.

    The unresolved values in the arrays persist from one iteration to the next
    when computing longest paths.


    Notes
    =====

     - The `partition` method reorders all arrays at the same time, such that
       all entries for unresolved connection longest paths _(i.e., where
       `source_longest_path < 0`)_ are packed to the front of the arrays.
      - This makes it possible to only process unresolved connections during
        each level iteration of longest paths calculation.


    TODO
    ====

     - Update code using the data from this class to use the `connection_count`
       to mark the end of the vectors, rather than using the `end()` method of
       the vector.  This would:
      - Improve run-time performance, since the arrays would not need to be
        reallocated on every call to `partition`.
      - All data could remain in the arrays, but just reordered.  After one
        complete pass through longest path calculations, this would result in
        the arrays in an order that could be sliced _(but not reordered)_
        to access all connections for a particular delay "level".
       - __NB__ This might not be that important, since the same thing could
         easily be accomplished by simply sorting the arrays by the unit
         longest paths _(i.e., "delay levels")_ after computing the longest
         paths according to each connection having a delay of 1.
    '''
    def __init__(self, sink_connections, source=CONNECTION_DRIVER):
        self.source = source
        assert(source == CONNECTION_DRIVER or source == CONNECTION_SINK)
        self._set_connection_data(sink_connections)

    def _set_connection_data(self, sink_connections):
        '''
        Load initial connection data from a `pandas.DataFrame` instance with the following
        columns:

         - `block_key`: The key of the sink block of each connection.
         - `driver_block_key`: The key of the driver block of each connection.
        '''
        self.connection_count = len(sink_connections)
        if self.source == CONNECTION_DRIVER:
            source_keys = sink_connections.driver_block_key.values
            target_keys = sink_connections.block_key.values
        else:
            source_keys = sink_connections.block_key.values
            target_keys = sink_connections.driver_block_key.values

        self.source_key = DeviceVectorInt32.from_array(source_keys)
        self.target_key = DeviceVectorInt32.from_array(target_keys)

    def partition_thrust(self, N, source_longest_path):
        '''
        Reorder all arrays at the same time, such that all entries for
        unresolved connection longest paths _(i.e., where `source_longest_path
        < 0`)_ are packed to the front of the arrays.

        This makes it possible to only process unresolved connections during
        each level iteration of longest paths calculation.
        '''
        move_unresolved_data_to_front(self.target_key, self.source_key,
                                      source_longest_path)
        self.connection_count = N
        self.target_key = DeviceVectorInt32.from_array(self.target_key[:N])
        self.source_key = DeviceVectorInt32.from_array(self.source_key[:N])

    def partition_numpy(self, not_ready_count, source_longest_path):
        '''
        `numpy`-based equivalent to the `partition_thrust` method.
        '''
        ready_to_calculate = (source_longest_path[:] >= 0)
        self.connection_count = not_ready_count
        self.target_key = DeviceVectorInt32.from_array(
            self.target_key[:][~ready_to_calculate])
        self.source_key = DeviceVectorInt32.from_array(
            self.source_key[:][~ready_to_calculate])

    def compute_connection_longest_paths(self, block_longest_paths):
        vectors = LevelConnectionData(self.connection_count,
                                      block_longest_paths.size)

        # For each connection, gather the longest path of the source of the
        # corresponding net.
        #
        # Equivalent to:
        #
        #     vectors.source_longest_path = \
        #         block_longest_paths[self.source_key]
        permute_float32(block_longest_paths, self.source_key,
                        vectors.source_longest_path)

        # Create mask, d_fining which connections are _not_ ready for
        # reduction. __NB__ A connection is ready for reduction once the
        # longest path is available for the block that is d_iving the
        # respective net.
        sequence_int32(vectors.d_idx)

        # Pack the list of indexes of all "ready" connections to the beginning
        # of the `d_idx` vector.
        #
        # Equivalent to performing the following two lines at the same time:
        #
        #      ready = (vectors.source_longest_path >= 0)
        #      d_idx[:len(ready)] = d_idx[ready]
        #      d_idx[len(ready):] = d_idx[~ready]
        #
        # where `ready` is a list of connection indexes corresponding to
        # connections with resolved longest paths.
        ready_count = partition_int32_float32_stencil_non_negative(
            vectors.d_idx, vectors.source_longest_path)

        # ready_to_calculate = d_idx[N:]
        # not_ready_to_calculate = vectors.d_idx[N:]

        # Load the delay of each connection.
        # TODO: Once the timing calculations are working, use position-based
        # delay.
        vectors.delay[:] = 1.

        # Start by marking all connection longest paths as _unresolved_.
        vectors.connection_longest_path[:] = -1

        # For each connection sourced by a source block with a _resolved_
        # longest path:
        #
        #  1. Add the longest path of the source block to the delay of the
        #     connection, storing the result .
        #
        # Equivalent to:
        #
        #      vectors.connection_longest_path[ready_to_calculate] = \
        #          (vectors.source_longest_path[ready_to_calculate] +
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
                                      vectors.source_longest_path,
                                      vectors.delay,
                                      vectors.connection_longest_path)
        return ready_count, vectors


@profile
def _update_longest_paths_thrust(timing_data, vectors, block_longest_paths):
    # Make local copy of `timing_data.target_key` vector since we need to sort
    # it.
    target_keys = DeviceVectorInt32.from_array(timing_data.target_key[:])
    sort_float32_by_int32_key(target_keys,
                              vectors.connection_longest_path)
    N_ = minmax_float32_by_key(target_keys,
                               vectors.connection_longest_path,
                               vectors.reduced_block_keys,
                               vectors.min_longest_path,
                               vectors.max_longest_path)

    # new code
    d_block_longest_paths = block_longest_paths
    sequence_int32(vectors.d_idx)

    block_count = partition_n_int32_float32_stencil_non_negative(
        vectors.d_idx, vectors.min_longest_path, N_)

    permute_n_int32(vectors.reduced_block_keys, vectors.d_idx, block_count,
                    vectors.d_ready_block_keys)

    permute_float32(d_block_longest_paths, vectors.d_ready_block_keys,
                    vectors.d_ready_block_longest_paths)

    unresolved_count = partition_int32_float32_stencil_negative(
        vectors.d_ready_block_keys, vectors.d_ready_block_longest_paths)

    sequence_int32(vectors.d_idx)
    partition_n_int32_float32_stencil_non_negative(vectors.d_idx,
                                                   vectors.min_longest_path,
                                                   N_)

    resolve_block_longest_paths(unresolved_count, vectors.max_longest_path,
                                vectors.d_idx, d_block_longest_paths,
                                vectors.d_ready_block_keys)


class DeviceBlockData(object):
    @profile
    def __init__(self, connections_table):
        # ## Identify all level zero blocks ##
        #
        # We need to identify the following blocks, since they must be assigned
        # an longest path of 0:
        #
        #  - Input blocks
        #  - Clocked logic-blocks
        #  - Clock drivers
        #
        # The keys of input and clocked logic-blocks are available as the
        # `input_block_keys` and `sync_logic_block_keys` methods of the
        # provided `connections_table`, which is an instance of the class
        # `cyplace_experiments.data.connections_table.ConnectionsTable`.
        #
        # All other blocks must be assigned an longest path of -1.
        self.input_block_keys = DeviceVectorInt32.from_array(
            connections_table.input_block_keys())
        self.output_block_keys = DeviceVectorInt32.from_array(
            connections_table.output_block_keys())
        self.sync_logic_block_keys = DeviceVectorInt32.from_array(
            connections_table.sync_logic_block_keys())
        # Logic blocks that only have a single connection, either an input or
        # an output.
        self.single_connection_blocks = DeviceVectorInt32.from_array(
            connections_table.single_connection_blocks)
        self.block_count = connections_table.block_count

    def set_longest_paths(self, connections, block_longest_paths,
                          source=CONNECTION_DRIVER):
        '''
        Populate columns on the provided connections based on the supplied
        longest paths.
        '''
        pd.set_option('line_width', 200)
        if source == CONNECTION_DRIVER:
            label = 'arrival'
        else:
            label = 'departure'

        connections['sync_driver'] = \
            connections.driver_block_key.isin(self.sync_logic_block_keys[:])
        connections['sync_sink'] = \
            connections.block_key.isin(self.sync_logic_block_keys[:])
        connections['driver_%s_level' % label] = \
            block_longest_paths[connections.driver_block_key.values]
        connections['sink_%s_level' % label] = \
            block_longest_paths[connections.block_key.values]


class LongestPath(object):
    @profile
    def compute_longest_paths(self, device_netlist_data, connections,
                              source=CONNECTION_DRIVER):
        block_longest_paths = DeviceVectorFloat32(device_netlist_data.
                                                  block_count)
        if source == CONNECTION_DRIVER:
            external_blocks = device_netlist_data.input_block_keys
        else:
            external_blocks = device_netlist_data.output_block_keys
        fill_longest_paths(external_blocks,
                           device_netlist_data.sync_logic_block_keys,
                           device_netlist_data.single_connection_blocks,
                           block_longest_paths)

        timing_data = UnresolvedConnectionData(connections, source)

        i = 0

        # __NB__ Assuming the input connection data is well-formed, the while
        # loop statement could be:
        #
        #    while timing_data.connection_count > 0:
        #        ...
        #
        # However, in the case of ill-formed input data, it is possible that
        # the loop would never exit.  To protect against this, we continue
        # looping until the number of unresolved connections does not change
        # between iterations.
        unresolved_connection_count = None
        while timing_data.connection_count != unresolved_connection_count:
            unresolved_connection_count = timing_data.connection_count
            self.compute_resolved_longest_paths(timing_data,
                                                block_longest_paths)
            i += 1
        assert(timing_data.connection_count == 0)

        return block_longest_paths

    @profile
    def compute_arrival_times(self, device_netlist_data, connections):
        block_longest_paths = self.compute_longest_paths(device_netlist_data,
                                                         connections,
                                                         CONNECTION_DRIVER)

        # Make one final pass to find the maximum incoming edge delay for
        # synchronous blocks, since they currently have an longest path of
        # zero.
        # TODO: Use Thrust to perform this reduction.
        connections['driver_arrival_level'] = \
            block_longest_paths[:][connections.driver_block_key.values]
        sync_block_longest_path = (connections[connections.block_key
                                                .isin(device_netlist_data
                                                      .sync_logic_block_keys[:]
                                                      )]
                                    .groupby(['block_key'])
                                    .agg({'driver_arrival_level': np.max}) + 1)
        _block_longest_paths = block_longest_paths[:]
        _block_longest_paths[sync_block_longest_path.index] = \
            sync_block_longest_path.values
        block_longest_paths[:] = _block_longest_paths

        return block_longest_paths[:]

    @profile
    def compute_departure_times(self, device_netlist_data, connections):
        block_longest_paths = self.compute_longest_paths(device_netlist_data,
                                                         connections,
                                                         CONNECTION_SINK)

        # Make one final pass to find the maximum incoming edge delay for
        # synchronous blocks, since they currently have an longest path of
        # zero.
        # TODO: Use Thrust to perform this reduction.
        connections['sink_departure_level'] = \
            block_longest_paths[:][connections.block_key.values]
        sync_block_longest_path = (connections[connections.driver_block_key
                                                .isin(device_netlist_data
                                                      .sync_logic_block_keys[:]
                                                      )]
                                    .groupby(['driver_block_key'])
                                    .agg({'sink_departure_level': np.max}) +
                                   1)
        _block_longest_paths = block_longest_paths[:]
        _block_longest_paths[sync_block_longest_path.index] = \
            sync_block_longest_path.values
        block_longest_paths[:] = _block_longest_paths

        return block_longest_paths[:]


    @profile
    def compute_resolved_longest_paths(self, timing_data, block_longest_path):
        ready_count, vectors = (timing_data.compute_connection_longest_paths(
            block_longest_path))
        not_ready_to_calculate = vectors.d_idx[ready_count:]

        # Reduce the computed connection longest-paths by block-key, to collect
        # the _minimum_ and _maximum_ longest-path for each block.
        #
        # __NB__ Each _resolved_ longest-path will have a value that is either
        # positive or equal to zero.  Each _unresolved_ longest-path will have
        # a value of -1.  Therefore, by collecting the minimum longest-path
        # from the connections corresponding to each block, we can determine if
        # all respective longest-paths have been resolved. If the reduced
        # minimum longest-path of all connections corresponding to a block is
        # negative, there is at least one _unresolved_ longest-path for the
        # respective block. However, if the reduced minimum longest-path for a
        # block is either zero or positive, all corresponding connection source
        # longest-paths must be resolved.  In this case, the reduced maximum
        # longest-path represents the longest-path for the corresponding block.
        _update_longest_paths_thrust(timing_data, vectors, block_longest_path)

        timing_data.partition_thrust(not_ready_to_calculate.size,
                                     vectors.source_longest_path)

        # __NB__ The `numpy` equivalent partition method can be run as shown
        # below:
        #
        #     timing_data.partition_numpy(not_ready_to_calculate.size,
        #                                 vectors.source_longest_path)
