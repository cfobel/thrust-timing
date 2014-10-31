import cythrust.device_vector as dv
from cythrust import DeviceDataFrame
from cythrust import DeviceVectorCollection
from thrust_timing.sort_timing import (compute_arrival_times,
                                       compute_departure_times)
from cyplace_experiments.data.connections_table import (CONNECTION_DRIVER,
                                                        CONNECTION_SINK)
from cyplace_experiments.data import open_netlists_h5f
import numpy as np
import pandas as pd
import thrust_timing.SORT_TIMING as SORT_TIMING
import cyplace_experiments.data.CONNECTIONS_TABLE as CONNECTIONS_TABLE
try:
    import thrust_timing.cuda.SORT_TIMING as cuSORT_TIMING
    import cyplace_experiments.data.cuda.CONNECTIONS_TABLE as cuCONNECTIONS_TABLE
except ImportError:
    pass

pd.set_option('display.width', 300)


try:
    profile
except:
    profile = lambda (f): f


def get_arch_data(nrows, ncols, allocator=dv):
    h5f = open_netlists_h5f()

    arch = getattr(h5f.root.architectures.vpr__k4_n1, 'x%04d_by_y%04d' %
                (nrows, ncols))

    arch_data = DeviceVectorCollection({'delays':
                                        np.empty(np.sum([np.prod(c.shape)
                                                        for c in arch]),
                                                 dtype=np.float32)},
                                       allocator=allocator)

    offset = 0

    for name in ('fb_to_fb', 'fb_to_io', 'io_to_fb', 'io_to_io'):
        c = getattr(arch, name)
        c_size = np.prod(c.shape)
        arch_data.v['delays'][offset:offset + c_size] = c[:].ravel()
        offset += c_size
    del arch
    h5f.close()
    arch_data.nrows = nrows
    arch_data.ncols = ncols
    return arch_data


class PathTimingData(object):
    def __init__(self, arch_data, connections_table, source=CONNECTION_DRIVER,
                 allocator=dv):
        if allocator == dv:
            self.SORT_TIMING = SORT_TIMING
            self.CONNECTIONS_TABLE = CONNECTIONS_TABLE
        else:
            self.SORT_TIMING = cuSORT_TIMING
            self.CONNECTIONS_TABLE = cuCONNECTIONS_TABLE
        self.allocator = allocator
        self.arch_data = arch_data
        if source == CONNECTION_DRIVER:
            self.special_blocks, block_data, connections = \
                compute_arrival_times(connections_table)
            driver_label = 'source_key'
            sink_label = 'target_key'
        elif source == CONNECTION_SINK:
            self.special_blocks, block_data, connections = \
                compute_departure_times(connections_table)
            sink_label = 'source_key'
            driver_label = 'target_key'
        else:
            raise ValueError('Invalid source type.  Must be one of '
                             '(CONNECTION_DRIVER, CONNECTION_SINK).')
        self.block_data = DeviceDataFrame(block_data[:], allocator=allocator)
        self.connections = DeviceDataFrame(
            connections[0].base()[['source_key', 'sync_source', 'target_key',
                                   'delay', 'source_longest_path',
                                   'target_longest_path',
                                   'max_target_longest_path', 'reduced_keys']],
            allocator=allocator)
        self.connections.add('delay_type', dtype=np.uint8)
        self.connections.add('sink_type', dtype=np.uint8)
        self.connections.add('driver_type', dtype=np.uint8)

        c = self.connections.v
        block_types = self.allocator.from_array(connections_table
                                                .block_data['type'].values)
        block_types_v = self.allocator.view_from_vector(block_types)
        self.CONNECTIONS_TABLE.driver_and_sink_type(c[driver_label],
                                                    c[sink_label],
                                                    c['driver_type'],
                                                    c['sink_type'],
                                                    block_types_v)
        self.CONNECTIONS_TABLE.connection_delay_type(c['driver_type'],
                                                     c['sink_type'],
                                                     c['delay_type'])

        # Construct views/slices of the connection data frame, one view per
        # delay level, in order.
        self.views = [self.connections.view(*v.index_bounds())
                      for v in connections]
        self.arrival_levels = (self.block_data['longest_paths'].values
                               .astype(np.int32))

    @profile
    def update_delays_from_positions(self, positions_device_vector_view):
        b = positions_device_vector_view
        c = self.connections.v
        self.SORT_TIMING.look_up_delay(c['source_key'], c['target_key'],
                                       c['delay_type'], b['p_x'], b['p_y'],
                                       self.arch_data.v['delays'],
                                       self.arch_data.nrows,
                                       self.arch_data.ncols, c['delay'])

    @profile
    def update_position_based_longest_paths(self, positions_device_vector_view):
        # Reset source_longest_path to sentinel value.
        self.connections.v['source_longest_path'][:] = -1e6
        self.update_delays_from_positions(positions_device_vector_view)

        self.block_data.v['longest_paths'][:] = 0

        for view in self.views:
            self.SORT_TIMING.step1(self.block_data.v['longest_paths'],
                                   view.v['source_key'],
                                   view.v['source_longest_path'])
            self.SORT_TIMING.step2(view.v['sync_source'],
                                   view.v['source_longest_path'])
            self.SORT_TIMING.step8(view.v['delay'],
                                   view.v['target_longest_path'],
                                   view.v['source_longest_path'], view.size)
            resolved_block_count = self.SORT_TIMING.step9(
                view.v['target_key'], view.v['target_longest_path'],
                view.v['max_target_longest_path'], view.v['reduced_keys'],
                view.size)
            self.SORT_TIMING.step10(self.block_data.v['longest_paths'],
                                    view.v['max_target_longest_path'],
                                    view.v['reduced_keys'],
                                    resolved_block_count)
        return self.block_data.v['longest_paths']
