from cyplace_experiments.data.connections_table import (get_simple_net_list,
                                                        ConnectionsTable)
from thrust_timing.DELAY_MODEL import fill_longest_paths
from cythrust.device_vector import DeviceVectorFloat32, DeviceVectorInt32

import numpy as np
import pandas as pd



def compute_departure_times(connections_table):
    connections = connections_table.sink_connections()[['driver_block_key', 'block_key']]
    connections.rename(columns={'block_key': 'source_key',
                                'driver_block_key': 'target_key'}, inplace=True)

    # Mark whether or not each source/target block is a synchronous logic block.
    block_is_sync = pd.Series(np.zeros(connections_table.block_count,
                                       dtype=np.bool))
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

    compute_longest_target_paths(connections, longest_paths)

    return pd.Series(longest_paths)


def compute_arrival_times(connections_table):
    connections = connections_table.sink_connections()[['driver_block_key', 'block_key']]
    connections.rename(columns={'driver_block_key': 'source_key',
                                'block_key': 'target_key'}, inplace=True)

    # Mark whether or not each source/target block is a synchronous logic block.
    block_is_sync = pd.Series(np.zeros(connections_table.block_count,
                                       dtype=np.bool))
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

    compute_longest_target_paths(connections, longest_paths)

    return pd.Series(longest_paths)


# In[ ]:

def compute_longest_target_paths(connections, longest_paths):
    # Initialize values that will be updated.
    connections['delay'] = float(-1e6)
    connections['target_longest_path'] = float(-1e6)
    connections['min_source_longest_path'] = float(-1e6)

    previous_count = None
    while previous_count != connections.shape[0]:
        proccessed_connection_count = compute_longest_level_target_paths(connections,
                                                                         longest_paths)
        previous_count = connections.shape[0]
        connections = connections[proccessed_connection_count:]
    return longest_paths


# In[498]:

def compute_longest_level_target_paths(connections, longest_paths):
    block_count = longest_paths.size

    block_min_source_longest_path = pd.Series(np.empty(block_count),
                                              dtype=np.float32)
    block_min_source_longest_path[:] = float(-1e6)

    connections['source_longest_path'] = longest_paths[connections.loc[:, 'source_key'].values]
    connections.loc[connections.sync_source, 'source_longest_path'] = 0

    # This sort is to represent the sort that is necessary before using `thrust::reduce_by_key`.
    connections.sort('target_key', inplace=True)
    connections.index = range(connections.shape[0])

    min_source_longest_path = connections.groupby('target_key').agg({'source_longest_path': np.min})
    block_min_source_longest_path[min_source_longest_path.index] = min_source_longest_path.values

    connections.min_source_longest_path =         block_min_source_longest_path[connections.target_key.values].values
    # This sort can be done using `thrust::partition`, which also returns
    # the partition point _(i.e., the number of ready connections)_.  Also,
    # since we need to do another reduce-by-key, we can avoid doing another
    # sort by the `target_key` if we use `thrust::stable_partition`.
    connections.sort(['min_source_longest_path', 'target_key'], inplace=True, ascending=False)
    ready_connection_count = (connections.min_source_longest_path >= 0).sum()

    # This would be replaced by the actual delay look-up for each corresponding connection.
    connections[:ready_connection_count].delay = 1.
    connections[:ready_connection_count].target_longest_path =         (connections[:ready_connection_count].delay +
         connections[:ready_connection_count].source_longest_path)

    max_target_longest_path =         connections[:ready_connection_count].groupby('target_key').agg({'target_longest_path': np.max})
    longest_paths[max_target_longest_path.index] = max_target_longest_path.values
#     display(connections)

    return ready_connection_count
