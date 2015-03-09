FPGA timing delay calculations using NVIDIA Thrust
==================================================

This repository contains an implementation of [FPGA][4] netlist timing delay
calculations using parallel patterns in the NVIDIA [Thrust][9] library. The
delay model is inspired by the cost model used in the timing-driven simulated
annealing placement method in the [VPR][7], which is now part of the popular
FPGA CAD suite, [VTR][5] (published [here][6]).

The parallel method described here algorithm was designed and developed as part
of Christian Fobel's PhD research in the [School of Computer Science][2] at the
[University of Guelph][3] in Guelph, Ontario, Canada, with funding support from
[NSERC][8].


# License #

The source code in this project is licensed as described in the contained
`COPYING` file.

Copyright (c) 2015 Christian Fobel


[2]: http://www.socs.uoguelph.ca/
[3]: http://www.uoguelph.ca/
[4]: http://en.wikipedia.org/wiki/Field-programmable_gate_array
[5]: https://code.google.com/p/vtr-verilog-to-routing/
[6]: http://dx.doi.org/10.1145/2617593
[7]: http://dx.doi.org/10.1145/2068716.2068718
[8]: http://www.nserc-crsng.gc.ca/index_eng.asp
[9]: http://thrust.github.io/
