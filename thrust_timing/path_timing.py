from cythrust.device_vector import DeviceVectorFloat32, DeviceVectorViewFloat32
import cythrust.device_vector as dv
from cythrust import DeviceDataFrame
from cythrust import DeviceVectorCollection
from thrust_timing.SORT_TIMING import (look_up_delay, step1, step2, step8,
                                       step9, step10)
from thrust_timing.sort_timing import (compute_arrival_times,
                                       compute_departure_times,
                                       reset_timing_data)
from cyplace_experiments.data.CONNECTIONS_TABLE import (driver_and_sink_type,
                                                        connection_delay_type)
from cyplace_experiments.data.connections_table import (CONNECTION_DRIVER,
                                                        CONNECTION_SINK)
from cyplace_experiments.data import open_netlists_h5f
import numpy as np
import pandas as pd

pd.set_option('display.width', 300)


try:
    profile
except:
    profile = lambda (f): f


def get_arch_data(nrows, ncols):
    h5f = open_netlists_h5f()

    arch = getattr(h5f.root.architectures.vpr__k4_n1, 'x%04d_by_y%04d' %
                (nrows, ncols))

    arch_data = DeviceVectorCollection({'delays':
                                        np.empty(np.sum([np.prod(c.shape)
                                                        for c in arch]),
                                                dtype=np.float32)})

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
    def __init__(self, arch_data, connections_table, source=CONNECTION_DRIVER):
        self.arch_data = arch_data
        if source == CONNECTION_DRIVER:
            self.special_blocks, self.block_data, connections = \
                compute_arrival_times(connections_table)
            driver_label = 'source_key'
            sink_label = 'target_key'
        elif source == CONNECTION_SINK:
            self.special_blocks, self.block_data, connections = \
                compute_departure_times(connections_table)
            sink_label = 'source_key'
            driver_label = 'target_key'
        else:
            raise ValueError('Invalid source type.  Must be one of '
                             '(CONNECTION_DRIVER, CONNECTION_SINK).')
        self.connections = DeviceDataFrame(connections[0].base()[
            ['source_key', 'sync_source', 'target_key', 'delay',
             'source_longest_path', 'target_longest_path',
             'max_target_longest_path',
             'reduced_keys']])
        self.connections.add('delay_type', dtype=np.uint8)
        self.connections.add('sink_type', dtype=np.uint8)
        self.connections.add('driver_type', dtype=np.uint8)

        c = self.connections.v
        b = connections_table.block_data.v
        driver_and_sink_type(c[driver_label], c[sink_label], c['driver_type'],
                             c['sink_type'], b['type'])
        connection_delay_type(c['driver_type'], c['sink_type'],
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
        look_up_delay(c['source_key'], c['target_key'], c['delay_type'],
                      b['p_x'], b['p_y'], self.arch_data.v['delays'],
                      self.arch_data.nrows, self.arch_data.ncols, c['delay'])

    @profile
    def update_position_based_longest_paths(self,
                                            positions_device_vector_view):
        '''
        BUG
        ===

        We are only copying into the intermediate `longest_paths` `numpy` array
        here because when we read from `self.block_data.v['longest_paths']`,
        there seem to be inconsistent entries that are overwritten with zero.

        See the comments in the code below for details.
        '''
        # Reset source_longest_path to sentinel value.
        self.connections.v['source_longest_path'][:] = -1e6
        #reset_timing_data(self.special_blocks, self.block_data,
                          #self.connections)
        self.update_delays_from_positions(positions_device_vector_view)

        self.block_data.v['longest_paths'][:] = 0
        longest_paths = np.zeros(self.block_data.v['longest_paths'].size,
                                 dtype=np.float32)

        for view in self.views:
            step1(self.block_data.v['longest_paths'], view.v['source_key'],
                  view.v['source_longest_path'])
            step2(view.v['sync_source'], self.block_data.v['longest_paths'])
            step8(view.v['delay'], view.v['target_longest_path'],
                  view.v['source_longest_path'], view.size)
            resolved_block_count = step9(view.v['target_key'],
                                         view.v['target_longest_path'],
                                         view.v['max_target_longest_path'],
                                         view.v['reduced_keys'], view.size)
            step10(self.block_data.v['longest_paths'],
                   view.v['max_target_longest_path'], view.v['reduced_keys'],
                   resolved_block_count)

            # TODO: __BUG__
            #
            # We are only copying into the `longest_paths` `numpy` array here
            # because when we read from `self.block_data.v['longest_paths']`,
            # there seem to be inconsistent entries that are overwritten with
            # zero.
            #
            # Notes
            # =====
            #
            #  - Since the particular entries _(including the number of
            #    entries)_ that are overwritten varies from run to run, it
            #    seems that there is some type of write hazard.
            #  - However, the longest path for the blocks at each level are
            #    dependent on the longest path of the blocks from previous
            #    levels.  This implies that the longest path values are being
            #    read from the `self.block_data.v['longest_paths']` array by
            #    the Thrust C code, where things seem to be working ok.
            #
            # TODO
            # ====
            #
            # The following is a potential Thrust-based workaround for the bug
            # described above.
            #
            #  - Create two new level-ordered arrays:
            #
            #   * `max_target_longest_path`
            #   * `target_key`
            #
            #  - Write a Thrust function to copy the following array contents
            #    from each iteration to a _contiguous section_ of the
            #    corresponding new, level-ordered array:
            #
            #   * `views.v['max_target_longest_path'][:resolved_block_count]`
            #   * `views.v['reduced_keys'][:resolved_block_count]`
            #
            #  - After all levels are processed, do a single scatter from the
            #    level-ordered `max_target_longest_path` to the output
            #    `longest_paths` array, using the `target_key` level-ordered
            #    array as the scatter index map.
            longest_paths[view.v['reduced_keys'][:resolved_block_count]] = \
                view.v['max_target_longest_path'][:resolved_block_count]
        return longest_paths
