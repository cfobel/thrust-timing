# coding: utf-8
import sys

import pandas as pd
import numpy as np
from nose.tools import ok_
from path_helpers import path


try:
    profile
except:
    profile = lambda (f): f


from cyplace_experiments.data.connections_table import (ConnectionsTable,
                                                        get_simple_net_list)
from thrust_timing.delay_model import DeviceBlockData, LongestPath

@profile
def _test_arrival_times_from_hdf(netlist_namebase, verify_file=None,
                                 save_file=None, append=False):
    if verify_file is not None and save_file is not None:
        raise RuntimeError('Cannot save result and verify result in the same '
                           'run.  Please provide only one or the other.')
    if verify_file is not None:
        assert(verify_file.isfile())
    if save_file is not None and save_file.isfile() and not append:
        raise RuntimeError('Output file exists.  Specify `-a` to append.')

    connections_table = ConnectionsTable.from_net_list_name(netlist_namebase)

    if verify_file is not None:
        data = pd.HDFStore(str(verify_file), 'r')
        if 'arrival_times' in getattr(data.root, netlist_namebase):
            gold_arrival_times = (getattr(data.root,
                                          netlist_namebase).arrival_times
                                  .values[:])
        else:
            gold_arrival_times = None
        if 'departure_times' in getattr(data.root, netlist_namebase):
            gold_departure_times = (getattr(data.root,
                                            netlist_namebase).departure_times
                                    .values[:])
        else:
            gold_departure_times = None
        data.close()
    else:
        gold_arrival_times = None
        gold_departure_times = None

    result = _test_arrival_times(connections_table, gold_arrival_times,
                                 gold_departure_times)
    device_netlist_data, sink_connections, arrival_times = result

    if verify_file is None and save_file is not None:
        pd.Series(arrival_times).to_hdf(str(save_file), '/%s/arrival_times' %
                                        netlist_namebase, complevel=2,
                                        complib='blosc')
        print 'wrote results to: %s' % save_file


def _test_arrival_times(connections_table, gold_arrival_times=None,
                        gold_departure_times=None):
    device_netlist_data = DeviceBlockData(connections_table)

    # TODO: Implement departure times calculation (I think this should just
    # involve swapping role of net_driver_block_key and block_key for each
    # connection, and using delay_model.clocked_sink_block_keys)
    sink_connections = connections_table.sink_connections().sort('block_key')
    #del connections_table
    longest_path = LongestPath()

    arrival_times = longest_path.compute_arrival_times(device_netlist_data,
                                                       sink_connections)
    departure_times = longest_path.compute_departure_times(device_netlist_data,
                                                           sink_connections)

    device_netlist_data.set_longest_paths(sink_connections, arrival_times[:])

    if gold_arrival_times is not None:
        np.testing.assert_equal(arrival_times, gold_arrival_times)
        print 'verified arrival'
    if gold_departure_times is not None:
        np.testing.assert_equal(departure_times, gold_departure_times)
        print 'verified departure'

    ok_(sink_connections[(sink_connections.sink_arrival_level <
                          sink_connections.driver_arrival_level)]
           .sync_driver.all())
    async_driver_connections = sink_connections.loc[~sink_connections
                                                    .sync_driver]
    ok_((async_driver_connections.sink_arrival_level >=
         async_driver_connections.driver_arrival_level).all())
    return device_netlist_data, sink_connections, arrival_times


def test_arrival_times_quick():
    fixtures_path = path(__file__).parent.joinpath('fixtures',
                                                   'arrival_times.dat')
    netlists = ['clma', 'ex5p', 'tseng', ]

    for n in netlists:
        yield _test_arrival_times_from_hdf, n, fixtures_path


def test_arrival_times_full():
    fixtures_path = path(__file__).parent.joinpath('fixtures',
                                                   'arrival_times.dat')

    h5f = pd.HDFStore(str(fixtures_path), 'r')
    names = h5f.root._v_children.keys()
    h5f.close()
    for name in names:
        yield _test_arrival_times_from_hdf, name, fixtures_path


def test_simple_netlist():
    # Load simple net-list with arrival times and departure times to check
    # against.
    connections_table, arrival_times, departure_times = get_simple_net_list()

    _test_arrival_times(connections_table, arrival_times, departure_times)


def parse_args(argv=None):
    '''Parses arguments, returns (options, args).'''
    from argparse import ArgumentParser

    if argv is None:
        argv = sys.argv

    parser = ArgumentParser(description='Compute arrival times for netlist.')
    parser.add_argument(dest='net_file_namebase')
    result_options = parser.add_mutually_exclusive_group()
    result_options.add_argument('-S', '--save-path', type=path, help='Path '
                                'to save results to.')
    result_options.add_argument('-V', '--verify-path', type=path, help='Path '
                                'containing data to verify against.')
    parser.add_argument('-a', '--append', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device_netlist_data, sink_connections, arrival_times = _test_arrival_times(
        args.net_file_namebase, verify_file=args.verify_path,
        save_file=args.save_path, append=args.append)
