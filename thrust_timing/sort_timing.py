from thrust_timing.DELAY_MODEL import fill_longest_paths
from cythrust.device_vector import DeviceVectorFloat32, DeviceVectorInt32

import numpy as np
import pandas as pd


try:
    profile
except:
    profile = lambda (f): f


@profile
def compute_departure_times(connections_table):
    connections = connections_table.sink_connections()[['block_key',
                                                        'driver_block_key']]
    connections.driver_block_key = (connections.driver_block_key.values
                                    .astype(np.int32))
    connections.block_key = connections.block_key.values.astype(np.int32)
    connections.rename(columns={'block_key': 'source_key',
                                'driver_block_key': 'target_key'}, inplace=True)

    # Mark whether or not each source/target block is a synchronous logic block.
    block_is_sync = pd.Series(np.zeros(connections_table.block_count,
                                       dtype=np.uint8))
    block_is_sync[connections_table.sync_logic_block_keys()] = True

    # Initialize constant values.
    connections['sync_source'] = block_is_sync[connections.source_key].values
    connections['sync_target'] = block_is_sync[connections.target_key].values

    d_longest_paths = DeviceVectorFloat32(connections_table.block_count)

    fill_longest_paths(DeviceVectorInt32.from_array(connections_table.output_block_keys()),
                       DeviceVectorInt32.from_array(connections_table.sync_logic_block_keys()),
                       DeviceVectorInt32.from_array(connections_table.single_connection_blocks),
                       d_longest_paths)
    longest_paths = d_longest_paths[:]

    return compute_longest_target_paths(connections, longest_paths)


@profile
def compute_arrival_times(connections_table):
    connections = connections_table.sink_connections()[['driver_block_key',
                                                        'block_key']]
    connections.driver_block_key = (connections.driver_block_key.values
                                    .astype(np.int32))
    connections.block_key = connections.block_key.values.astype(np.int32)
    connections.rename(columns={'driver_block_key': 'source_key',
                                'block_key': 'target_key'}, inplace=True)

    # Mark whether or not each source/target block is a synchronous logic block.
    block_is_sync = pd.Series(np.zeros(connections_table.block_count,
                                       dtype=np.uint8))
    block_is_sync[connections_table.sync_logic_block_keys()] = True

    # Initialize constant values.
    connections['sync_source'] = block_is_sync[connections.source_key].values
    connections['sync_target'] = block_is_sync[connections.target_key].values

    d_longest_paths = DeviceVectorFloat32(connections_table.block_count)

    fill_longest_paths(DeviceVectorInt32.from_array(connections_table.input_block_keys()),
                       DeviceVectorInt32.from_array(connections_table.sync_logic_block_keys()),
                       DeviceVectorInt32.from_array(connections_table.single_connection_blocks),
                       d_longest_paths)
    longest_paths = d_longest_paths[:]

    return compute_longest_target_paths(connections, longest_paths)


@profile
def do_iteration(v_block_min_source_longest_path, v_longest_paths,
                 d_connections):
    from thrust_timing.SORT_TIMING import (step1, step2, step3, step4, step5,
                                           step6, step7, step8, step9, step10)

    step1(v_block_min_source_longest_path, v_longest_paths, *d_connections._view_dict.values())
    step2(v_block_min_source_longest_path, v_longest_paths, *d_connections._view_dict.values())
    unique_block_count = step3(v_block_min_source_longest_path, v_longest_paths, *d_connections._view_dict.values())
    step4(v_block_min_source_longest_path, v_longest_paths, *(d_connections._view_dict.values() + [unique_block_count]))
    step5(v_block_min_source_longest_path, v_longest_paths, *d_connections._view_dict.values())
    ready_connection_count = step6(v_block_min_source_longest_path, v_longest_paths, *d_connections._view_dict.values())
    step7(v_block_min_source_longest_path, v_longest_paths, *(d_connections._view_dict.values() + [ready_connection_count]))
    step8(v_block_min_source_longest_path, v_longest_paths, *(d_connections._view_dict.values() + [ready_connection_count]))
    resolved_block_count = step9(v_block_min_source_longest_path, v_longest_paths, *(d_connections._view_dict.values() + [ready_connection_count]))
    step10(v_block_min_source_longest_path, v_longest_paths, *(d_connections._view_dict.values() + [resolved_block_count]))
    return unique_block_count, ready_connection_count, resolved_block_count


@profile
def compute_longest_target_paths(connections, longest_paths):
    from cythrust import DeviceDataFrame
    import cythrust.device_vector as dv

    # Initialize values that will be updated.
    init_value = -1e6

    d_connections = DeviceDataFrame(connections)
    d_connections.add('delay', dtype=np.float32)
    d_connections.add('target_longest_path', dtype=np.float32)
    d_connections.add('min_source_longest_path', dtype=np.float32)
    d_connections.add('source_longest_path', dtype=np.float32)
    d_connections.add('max_target_longest_path', dtype=np.float32)
    d_connections.add('reduced_keys', dtype=np.int32)
    d_connections.v['delay'][:] = init_value
    d_connections.v['target_longest_path'][:] = init_value
    d_connections.v['min_source_longest_path'][:] = init_value

    d_block_data = DeviceDataFrame()
    d_block_data.add('longest_paths', longest_paths, dtype=np.float32)
    d_block_data.add('min_source_longest_path', dtype=np.float32)

    v_connections = d_connections.view(d_connections.size)

    resolved_block_count = 1
    while resolved_block_count > 0:
        unique_block_count, ready_connection_count, resolved_block_count = \
            do_iteration(d_block_data.v['min_source_longest_path'],
                         d_block_data.v['longest_paths'], v_connections)
        v_connections = v_connections.view(ready_connection_count,
                                           v_connections.size)
        pass

    return d_connections, d_block_data


@profile
def compute_longest_level_target_paths(connections, longest_paths):
    block_count = longest_paths.size

    block_min_source_longest_path = pd.Series(np.empty(block_count),
                                              dtype=np.float32)
    block_min_source_longest_path[:] = float(-1e6)

    connections['source_longest_path'] = longest_paths[connections
                                                       .loc[:, 'source_key']
                                                       .values]
    connections.loc[connections.sync_source > 0, 'source_longest_path'] = 0

    # This sort is to represent the sort that is necessary before using `thrust::reduce_by_key`.
    connections.sort('target_key', inplace=True)
    connections.index = range(connections.shape[0])

    min_source_longest_path = connections.groupby('target_key').agg({'source_longest_path': np.min})
    block_min_source_longest_path[min_source_longest_path.index] = min_source_longest_path.values

    connections.min_source_longest_path = \
        block_min_source_longest_path[connections.target_key.values].values

    # This sort can be done using `thrust::partition`, which also returns
    # the partition point _(i.e., the number of ready connections)_.  Also,
    # since we need to do another reduce-by-key, we can avoid doing another
    # sort by the `target_key` if we use `thrust::stable_partition`.
    connections.sort(['min_source_longest_path', 'target_key'], inplace=True,
                     ascending=False)
    ready_connection_count = (connections.min_source_longest_path >= 0).sum()

    # This would be replaced by the actual delay look-up for each corresponding
    # connection.
    connections[:ready_connection_count].delay = 1.
    connections[:ready_connection_count].target_longest_path = \
        (connections[:ready_connection_count].delay +
         connections[:ready_connection_count].source_longest_path)

    max_target_longest_path = \
        (connections[:ready_connection_count].groupby('target_key')
         .agg({'target_longest_path': np.max}))
    longest_paths[max_target_longest_path.index] = max_target_longest_path.values
#     display(connections)

    return ready_connection_count
