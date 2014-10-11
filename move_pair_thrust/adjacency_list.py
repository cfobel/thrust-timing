import numpy as np
import pandas as pd


try:
    profile
except:
    profile = lambda (f): f


class NetAdjacencyList(object):
    @profile
    def __init__(self, net_count, block_count, connections,
                 ignore_clock=False):
        self.net_count = net_count
        self.block_count = block_count
        self.connections = connections
        self.ignore_clock = ignore_clock

        # Input blocks
        # ============
        #
        # Pin 0 -> output _(i.e., driver)_

        # Output blocks
        # ============
        #
        # Pin 0 -> input _(i.e., sink)_

        # Logic blocks
        # ============
        #
        # Pins 0-3 -> input _(i.e., sink)_
        # Pin 4 -> output _(i.e., driver)_
        # Pin 5 -> clock _(i.e., block is synchronous)_

        # All driver connections
        # ======================
        #
        # Include:
        #
        #  - All input block connections
        query = (connections['block_type'] == '.input')
        #  - All logic block connections to _pin 4_.
        query |= ((connections['block_type'] == '.clb') &
                  (connections['pin_key'] == 4))

        self.driver_connections = connections[query]

        # All sink connections
        # ====================
        #
        # Include:
        #
        #  - All input block connections
        query = (connections['block_type'] == '.output')
        #  - All logic block connections to _pin 4_.
        query |= ((connections['block_type'] == '.clb') &
                  (connections['pin_key'] < 4))

        self.sink_connections = connections[query]

        # All synchronous/clock connections
        # =================================
        #
        # Include:
        #
        #  - All logic block connections to _pin 5_.
        query = ((connections['block_type'] == '.clb') &
                 (connections['pin_key'] == 5))
        self.clock_connections = connections[query]

        # Notes
        # =====
        #
        # All connections should be one of the following:
        #
        #  - Driver connection
        #  - Sink connection
        #  - Clock connection
        #
        # Verify this is the case, by ensuring the sum of driver, sink and
        # clock connection counts is equal to the total number of connections.
        typed_connection_count = (self.driver_connections.shape[0] +
                                  self.sink_connections.shape[0] +
                                  self.clock_connections.shape[0])
        assert(typed_connection_count == self.total_connection_count)

        # Construct _(static)_ list of clocked-driver block keys, containing
        # the key of each block that is either a) an input block, or b) a logic
        # block that is connected to the clock and is driving a net _(all logic
        # blocks should be driving a net)_.
        self.clocked_driver_block_keys = (
            self.driver_connections[self.driver_connections['block_type']
                                    == '.input']['block_key'].unique())
        if not ignore_clock:
            self.clocked_driver_block_keys = np.concatenate(
                [self.clocked_driver_block_keys,
                 self.clock_connections['block_key'].unique()])

        # Gather the key of each block that is either:
        #
        #  - An output.
        #  - A synchonous logic block, _i.e., a logic block that is connected
        #    to a clock signal on pin 5_.
        self.clocked_sink_block_keys = (self.connections
                                        [(self.connections['block_type'] ==
                                          '.output')]['block_key'].unique())
        if not ignore_clock:
            self.clocked_sink_block_keys = np.concatenate(
                [self.clocked_sink_block_keys,
                 self.connections[(self.connections['block_type'] == '.clb') &
                                  (self.connections['pin_key'] ==
                                   5)]['block_key'].unique()])

    @property
    def total_connection_count(self):
        return self.connections.shape[0]

    @classmethod
    @profile
    def from_hdf_group(self, netlist_group, **kwargs):
        connections = pd.DataFrame(np.array([
            getattr(netlist_group.connections.cols, r)[:]
            for r in netlist_group.connections.colnames]).T,
            columns=netlist_group.connections.colnames)
        connections['block_label'] = (netlist_group.block_labels[:]
                                      [connections['block_key'].as_matrix()])
        connections['net_label'] = (netlist_group.net_labels[:]
                                    [connections['net_key'].as_matrix()])
        connections['block_type_key'] = (netlist_group.block_types[:]
                                         [connections['block_key']
                                          .as_matrix()])
        connections['block_type'] = (netlist_group.block_type_counts.cols
                                     .label[:][connections['block_type_key']
                                               .as_matrix()])
        return NetAdjacencyList(netlist_group._v_attrs.net_count,
                                netlist_group._v_attrs.block_count,
                                connections, **kwargs)
