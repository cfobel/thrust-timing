# coding: utf-8
import sys
from cyplace_experiments.data import get_net_list_connection_counts
from cyplace_experiments.data.connections_table import (get_simple_net_list,
                                                        ConnectionsTable)
from thrust_timing.sort_timing import compute_arrival_times, compute_departure_times

import numpy as np
import pandas as pd

pd.set_option('line_width', 300)


def test_simple_netlist():
    connections_table, arrival_times, departure_times =  get_simple_net_list()

    block_data, connections = compute_arrival_times(connections_table)
    _arrival_times = block_data['longest_paths'].values
    block_data, connections = compute_departure_times(connections_table)
    _departure_times = block_data['longest_paths'].values

    np.testing.assert_array_equal(_arrival_times, arrival_times)
    np.testing.assert_array_equal(_departure_times, departure_times)


def _test_net_list_by_name(net_list_name):
    connections_table = ConnectionsTable.from_net_list_name(net_list_name)

    block_data, connections = compute_arrival_times(connections_table)
    arrival_times = block_data['longest_paths'].values
    block_data, connections = compute_departure_times(connections_table)
    departure_times = block_data['longest_paths'].values
    np.testing.assert_array_equal(arrival_times[connections_table
                                                .input_block_keys()], 0)
    np.testing.assert_array_equal(departure_times[connections_table
                                                .output_block_keys()], 0)
    np.testing.assert_array_less(np.repeat(-1, arrival_times.size),
                                 departure_times)
    np.testing.assert_array_less(np.repeat(-1, departure_times.size),
                                 departure_times)


def test_arrival_times_quick():
    netlists = ['clma', 'ex5p', 'tseng', ]

    for n in netlists:
        yield _test_net_list_by_name, n


def test_arrival_times_full():
    names = get_net_list_connection_counts().name.values

    for name in names:
        yield _test_net_list_by_name, name


def parse_args(argv=None):
    '''Parses arguments, returns (options, args).'''
    from argparse import ArgumentParser

    if argv is None:
        argv = sys.argv

    names = get_net_list_connection_counts().name.values
    parser = ArgumentParser(description='Compute arrival times for netlist.')
    parser.add_argument(dest='net_file_namebase', choices=names)

    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parse_args()
    _test_net_list_by_name(args.net_file_namebase)
