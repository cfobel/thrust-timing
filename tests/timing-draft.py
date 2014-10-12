# coding: utf-8
import re
import sys
import cPickle as pickle

import pandas as pd
from nose.tools import ok_
from cyplace_experiments.data import open_netlists_h5f
from path_helpers import path


try:
    profile
except:
    profile = lambda (f): f


from cyplace_experiments.data.connections_table import ConnectionsTable
from thrust_timing.delay_model import DelayModel


@profile
def _test_arrival_times(netlist_namebase, verify_file=None, save_file=None,
                        append=False):
    if verify_file is not None and save_file is not None:
        raise RuntimeError('Cannot save result and verify result in the same '
                           'run.  Please provide only one or the other.')
    if verify_file is not None:
        assert(verify_file.isfile())
    if save_file is not None and save_file.isfile() and not append:
        raise RuntimeError('Output file exists.  Specify `-a` to append.')

    connections_table = ConnectionsTable(netlist_namebase)
    delay_model = DelayModel(connections_table)

    # TODO: Implement departure times calculation (I think this should just
    # involve swapping role of net_driver_block_key and block_key for each
    # connection, and using delay_model.clocked_sink_block_keys)
    sink_connections = connections_table.sink_connections().sort('block_key')
    del connections_table
    arrival_times = delay_model.compute_arrival_times(sink_connections)

    if verify_file is not None:
        data = pd.HDFStore(str(verify_file), 'r')
        ok_((getattr(data.root, netlist_namebase).arrival_times.values[:] ==
             arrival_times).all())
        data.close()
        print 'verified'
    elif save_file is not None:
        pd.Series(arrival_times).to_hdf(str(save_file), '/%s/arrival_times' %
                                        netlist_namebase, complevel=2,
                                        complib='blosc')
        print 'wrote results to: %s' % save_file
    delay_model.set_arrival_times(sink_connections, arrival_times[:])
    ok_(sink_connections[(sink_connections.sink_arrival_level <
                          sink_connections.driver_arrival_level)]
           .sync_driver.all())
    async_driver_connections = sink_connections.loc[~sink_connections
                                                    .sync_driver]
    ok_((async_driver_connections.sink_arrival_level >=
         async_driver_connections.driver_arrival_level).all())
    return delay_model, sink_connections, arrival_times


def test_arrival_times_quick():
    fixtures_path = path(__file__).parent.joinpath('fixtures',
                                                   'arrival_times.dat')
    netlists = ['clma', 'ex5p', 'tseng', ]

    for n in netlists:
        yield _test_arrival_times, n, fixtures_path


def test_arrival_times_full():
    fixtures_path = path(__file__).parent.joinpath('fixtures',
                                                   'arrival_times.dat')

    h5f = pd.HDFStore(str(fixtures_path), 'r')
    names = h5f.root._v_children.keys()
    h5f.close()
    for name in names:
        yield _test_arrival_times, name, fixtures_path


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
    delay_model, sink_connections, arrival_times = _test_arrival_times(
        args.net_file_namebase, verify_file=args.verify_path,
        save_file=args.save_path, append=args.append)
