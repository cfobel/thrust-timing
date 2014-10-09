# coding: utf-8
import sys
import cPickle as pickle

from IPython.display import display, Image
import pandas as pd
import numpy as np
from cyplace_experiments.data import get_data_directory, open_netlists_h5f
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
def test(netlist_namebase, use_thrust):
    netlist_h5f = open_netlists_h5f()

    #netlist_namebase = 'clma'
    #netlist_namebase = 'ex5p'
    #netlist_namebase = 'uoft_raytracer_low_fanout_latch-1x4lut'
    #netlist_namebase = 'b19_1_new_mapped-1x4lut'
    netlist_group = getattr(netlist_h5f.root.netlists, netlist_namebase)
    adjacency_list = NetAdjacencyList.from_hdf_group(netlist_group)
    netlist_h5f.close()
    delay_model = DelayModel(adjacency_list)
    arrival_times = delay_model.compute_arrival_times(use_thrust=use_thrust)

    #import pudb; pudb.set_trace()
    cached = path('%s-arrival_times.pickled' % netlist_namebase)
    if use_thrust and cached.isfile():
        data = pickle.load(cached.open('rb'))
        assert((data == arrival_times).all())
        print 'verified'
    elif not use_thrust:
        pd.Series(arrival_times).to_pickle(cached)
        print 'wrote cached arrival times to:', cached


def parse_args(argv=None):
    '''Parses arguments, returns (options, args).'''
    from argparse import ArgumentParser

    if argv is None:
        argv = sys.argv

    parser = ArgumentParser(description='Compute arrival times for netlist.')
    parser.add_argument('-d', '--disable-thrust', action='store_true')
    parser.add_argument(dest='net_file_namebase')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #import pudb; pudb.set_trace()
    args = parse_args()
    test(args.net_file_namebase, use_thrust=(not args.disable_thrust))
