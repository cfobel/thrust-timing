# coding: utf-8
import sys
import cPickle as pickle

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
def _test_arrival_times(netlist_namebase):
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

    cached = path('%s-arrival_times.pickled' % netlist_namebase)
    if cached.isfile():
        data = pickle.load(cached.open('rb'))
        assert((data == arrival_times).all())
        print 'verified'
    return adjacency_list, delay_model, arrival_times


def test_arrival_times():
    netlists = ['clma', 'ex5p', 'tseng', ]

    for n in netlists:
        yield _test_arrival_times, n


def parse_args(argv=None):
    '''Parses arguments, returns (options, args).'''
    from argparse import ArgumentParser

    if argv is None:
        argv = sys.argv

    parser = ArgumentParser(description='Compute arrival times for netlist.')
    parser.add_argument(dest='net_file_namebase')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    adjacency_list, delay_model, arrival_times = _test_arrival_times(
        args.net_file_namebase)
