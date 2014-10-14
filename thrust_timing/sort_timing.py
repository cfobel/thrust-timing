from collections import OrderedDict
from thrust_timing.SORT_TIMING import fill_longest_paths
from cythrust import DeviceDataFrame, DeviceVectorCollection
from cyplace_experiments.data.connections_table import (CONNECTION_DRIVER,
                                                        CONNECTION_SINK)
from .SORT_TIMING import (reset_block_min_source_longest_path, step1, step2,
                          step3, step4, step5, step6, step7, step8, step9,
                          step10, sort_by_target_key)


import numpy as np
import pandas as pd


try:
    profile
except:
    profile = lambda (f): f


@profile
def prepare_device_timing_data(connections_table, source):
    '''
    Extract relevant information for timing calculations from a
    `ConnectionsTable` instance.  The data is copied into
    `cythrust.DeviceVector` instances, which provide a means to interface with
    Thrust code.

    The extracted data includes:

     - Per-block data, including:
      * The longest path delay of connections entering _either_ the inputs or
        outputs of each block.
      * The longest path delay of connections entering _either_ the inputs or
        outputs of each block.


    Arguments
    =========

     - `connections_table`:
      * A `ConnectionsTable` instance, as, _e.g., returned by
        `ConnectionsTable.from_net_list_name(<ex5p,clma,etc.>)`.

     - `source`:
      * The type of block to treat as a source. One of:
       - `CONNECTION_DRIVER`: Driver blocks are treated as the source _(arrival
         times)_.
       - `CONNECTION_SINK`: Sink blocks are treated as the source _(departure
         times)_.
    '''
    source_label = 'driver_key'
    target_label = 'sink_key'
    external_blocks = 'input_block_keys'

    if source != CONNECTION_DRIVER:
        # Assume `source == CONNECTION_SINK`
        source_label, target_label = target_label, source_label
        external_blocks = 'output_block_keys'

    d_block_data = DeviceDataFrame()
    d_block_data.add('longest_paths', np.empty(connections_table.block_count),
                     dtype=np.float32)
    d_block_data.add('min_source_longest_path', dtype=np.float32)

    d_special_blocks = DeviceVectorCollection(
        OrderedDict([('external_block_keys', getattr(connections_table,
                                                     external_blocks)()
                      .astype(np.int32)),
                     ('sync_logic_block_keys',
                      connections_table.sync_logic_block_keys()
                      .astype(np.int32)),
                     ('single_connection_blocks', connections_table
                      .single_connection_blocks)]))

    connections = connections_table.sink_connections()[[source_label,
                                                        target_label]]
    for label in (source_label, target_label):
        connections[label] = connections[label].values.astype(np.int32)
    connections.rename(columns={source_label: 'source_key',
                                target_label: 'target_key'}, inplace=True)

    # Mark whether or not each source/target block is a synchronous logic
    # block.
    block_is_sync = pd.Series(np.zeros(connections_table.block_count,
                                       dtype=np.uint8))
    block_is_sync[connections_table.sync_logic_block_keys()] = True

    # Initialize constant values.
    connections['sync_source'] = block_is_sync[connections.source_key].values
    connections['sync_target'] = block_is_sync[connections.target_key].values

    d_connections = DeviceDataFrame(connections)
    d_connections.add('delay', dtype=np.float32)
    d_connections.add('target_longest_path', dtype=np.float32)
    d_connections.add('min_source_longest_path', dtype=np.float32)
    d_connections.add('source_longest_path', dtype=np.float32)
    d_connections.add('max_target_longest_path', dtype=np.float32)
    d_connections.add('reduced_keys', dtype=np.int32)

    v_connections = d_connections.view(d_connections.size)

    return d_special_blocks, d_block_data, v_connections


def reset_timing_data(d_special_blocks, d_block_data, v_connections):
    # Initialize values that will be updated.
    init_value = -1e6

    v_connections.v['delay'][:] = init_value
    v_connections.v['target_longest_path'][:] = init_value
    if 'min_source_longest_path' in v_connections.v:
        v_connections.v['min_source_longest_path'][:] = init_value

    fill_longest_paths(d_special_blocks.v['external_block_keys'],
                       d_special_blocks.v['sync_logic_block_keys'],
                       d_special_blocks.v['single_connection_blocks'],
                       d_block_data.v['longest_paths'])


@profile
def compute_departure_times(connections_table):
    d_special_blocks, d_block_data, v_connections = prepare_device_timing_data(
        connections_table, CONNECTION_SINK)
    reset_timing_data(d_special_blocks, d_block_data, v_connections)
    d_block_data, views = compute_longest_target_paths(d_block_data,
                                                       v_connections)
    return d_special_blocks, d_block_data, views


@profile
def compute_arrival_times(connections_table):
    d_special_blocks, d_block_data, v_connections = prepare_device_timing_data(
        connections_table, CONNECTION_DRIVER)
    reset_timing_data(d_special_blocks, d_block_data, v_connections)
    d_block_data, views = compute_longest_target_paths(d_block_data,
                                                       v_connections)
    return d_special_blocks, d_block_data, views


@profile
def do_iteration(d_block_data, v_connections):
    reset_block_min_source_longest_path(
        d_block_data.v['min_source_longest_path'])
    step1(d_block_data.v['longest_paths'], v_connections.v['source_key'],
          v_connections.v['source_longest_path'])
    step2(v_connections.v['sync_source'],
          v_connections.v['source_longest_path'])
    unique_block_count = step3(v_connections.v['source_key'],
                               v_connections.v['target_key'],
                               v_connections.v['sync_source'],
                               v_connections.v['sync_target'],
                               v_connections.v['delay'],
                               v_connections.v['target_longest_path'],
                               v_connections.v['min_source_longest_path'],
                               v_connections.v['source_longest_path'],
                               v_connections.v['reduced_keys'])
    step4(d_block_data.v['min_source_longest_path'],
          v_connections.v['min_source_longest_path'],
          v_connections.v['reduced_keys'], unique_block_count)
    step5(d_block_data.v['min_source_longest_path'],
          v_connections.v['target_key'],
          v_connections.v['min_source_longest_path'])
    ready_connection_count = step6(v_connections.v['source_key'],
                                   v_connections.v['target_key'],
                                   v_connections.v['sync_source'],
                                   v_connections.v['sync_target'],
                                   v_connections.v['target_longest_path'],
                                   v_connections.v['min_source_longest_path'],
                                   v_connections.v['source_longest_path'])
    step7(v_connections.v['delay'], ready_connection_count)
    step8(v_connections.v['delay'], v_connections.v['target_longest_path'],
          v_connections.v['source_longest_path'],
          ready_connection_count)
    resolved_block_count = step9(v_connections.v['target_key'],
                                 v_connections.v['target_longest_path'],
                                 v_connections.v['max_target_longest_path'],
                                 v_connections.v['reduced_keys'],
                                 ready_connection_count)
    step10(d_block_data.v['longest_paths'],
           v_connections.v['max_target_longest_path'],
           v_connections.v['reduced_keys'], resolved_block_count)

    return unique_block_count, ready_connection_count, resolved_block_count


@profile
def compute_longest_target_paths(d_block_data, v_connections):
    views = []

    # Sort connection data by target block key, since we perform reductions by
    # the target key in step 3 and step 9 in `do_iteration`.
    sort_by_target_key(v_connections.v['source_key'],
                       v_connections.v['target_key'],
                       v_connections.v['sync_source'],
                       v_connections.v['sync_target'],
                       v_connections.v['delay'])

    resolved_block_count = 1
    while resolved_block_count > 0:
        unique_block_count, ready_connection_count, resolved_block_count = \
            do_iteration(d_block_data, v_connections)
        if v_connections.size > 0:
            views.append(v_connections.view(0, ready_connection_count))
        v_connections = v_connections.view(ready_connection_count,
                                           v_connections.size)
        pass

    return d_block_data, views