# coding: utf-8
from cyplace_experiments.data import get_net_list_connection_counts
from cyplace_experiments.data.connections_table import (get_simple_net_list,
                                                        ConnectionsTable)
from thrust_timing.sort_timing import compute_arrival_times, compute_departure_times

import numpy as np


def test_simple_netlist():
    connections_table, arrival_times, departure_times =  get_simple_net_list()

    _arrival_times = compute_arrival_times(connections_table)
    _departure_times = compute_departure_times(connections_table)

    np.testing.assert_array_equal(_arrival_times, arrival_times)
    np.testing.assert_array_equal(_departure_times, departure_times)


def _test_net_list_by_name(net_list_name):
    connections_table = ConnectionsTable.from_net_list_name(net_list_name)

    arrival_times = compute_arrival_times(connections_table)
    departure_times = compute_departure_times(connections_table)
    np.testing.assert_array_equal(arrival_times[connections_table
                                                .input_block_keys()], 0)
    np.testing.assert_array_equal(departure_times[connections_table
                                                .output_block_keys()], 0)


def test_arrival_times_quick():
    netlists = ['clma', 'ex5p', 'tseng', ]

    for n in netlists:
        yield _test_net_list_by_name, n


def test_arrival_times_full():
    names = get_net_list_connection_counts().name.values

    for name in names:
        yield _test_net_list_by_name, name
