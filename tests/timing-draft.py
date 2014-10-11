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


## Arrival time calculation using structured parallel primitives

# ## TODO ##
#
#  - Write textual description of algorithm.

# In[2]:

from move_pair_thrust.adjacency_list import NetAdjacencyList
from move_pair_thrust.delay_model import DelayModel


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

    netlist_h5f = open_netlists_h5f()

    netlist_group = getattr(netlist_h5f.root.netlists, netlist_namebase)
    adjacency_list = NetAdjacencyList.from_hdf_group(netlist_group)
    netlist_h5f.close()
    delay_model = DelayModel(adjacency_list)
    #import pudb; pudb.set_trace()

    # TODO: Make one final pass to find the maximum incoming edge delay for
    # synchronous blocks, since they currently have an arrival time of zero.

    # TODO: Implement departure times calculation (I think this should just
    # involve swapping role of net_driver_block_key and block_key for each
    # connection, and using delay_model.clocked_sink_block_keys)
    arrival_times = delay_model.compute_arrival_times(netlist_namebase)

    if verify_file is not None:
        data = pd.HDFStore(str(verify_file), 'r')
        assert((getattr(data.root.arrival_times, netlist_namebase).values[:] ==
                arrival_times).all())
        data.close()
        print 'verified'
    elif save_file is not None:
        pd.Series(arrival_times).to_hdf(str(save_file), '/arrival_times/%s' %
                                        netlist_namebase, complevel=2,
                                        complib='blosc')
        print 'wrote results to: %s' % save_file
    return adjacency_list, delay_model, arrival_times


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
    names = h5f.root.arrival_times._v_children.keys()
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
    adjacency_list, delay_model, arrival_times = _test_arrival_times(
        args.net_file_namebase, verify_file=args.verify_path,
        save_file=args.save_path, append=args.append)
