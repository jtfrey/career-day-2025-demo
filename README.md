# career-day-2025-demo

Materials associated with a demo presented at the Tome School 2025 Career Day.  This project uses a simple integer-based neural network to identify colored polygons:

1. Inputs are the number of sides to a polygon, _s_ and its color, _c_.
2. The color, _c_, is expressed as a 3-bit value where:
   - 0 = black
   - 1 = red
   - 2 = blue
   - 3 = 2 + 1 = red + blue = purple
   - 4 = green
   - 5 = 4 + 1 = green + red = yellow
   - 6 = 4 + 2 = green + blue = cyan
   - 7 = 4 + 2 + 1 = red + blue + green = white
3. Output is a single integer value index into descriptive names for the figures.

Presentation materials include several placards of colored polygons, neuron description sheets for three (3) student helpers, and a descriptive name table for each of two parameterizations of the network.

# basic-nnet.py

The code in the basic script has no external dependencies and implements the network layers as Python functions.  Lacking any command-line arguments, the script iterates over all combinations of _s_ and _c_ to display their mappings via the neural network.

A single command-line argument is interpreted as an alternative value for the second weighting parameter, _w2_ in the output layer.  The default, 5, produces some mistaken mappings in the network; the value of 8 successfully distinguishes all combinations.

With two command-line arguments, an explicit singular value of _s_ and _c_ are assumed.  This pair alone is presented to the network and analyzed.

A third command-line argument augments the explicit _s_ and _c_ with a value for _w2_.

A fourth command-line argument augments the explicit _s_ and _c_ with values for _w1_ and _w2_, respectively.

The script produces a help screen when the `-h` or `--help` argument is specified:

```
$ python ./basic-nnet.py -h 
usage:

    ./basic-nnet.py | <s> <c> {{<w1>} <w2>} | <w2>
    
    <s>  : number of polygon sides
    <c>  : color value, [0...7]
    <w1> : weight of output from hidden layer neuron 1 (default: 3)
    <w2> : weight of output from hidden layer neuron 2 (default: 5)
    
    With no arguments the full range of <s> and <c> are enumerated.

```

With output layer weights _w1_ and _w2_ of -2 and 7, a yellow (5) triangle (3) maps to 45:

```
$ python ./basic-nnet.py 3 5 -2 7
( 3, 5) = <-2,7>.[-5, 5]      =   45
  45 => yellow triangle @ (3,5)
Repeats:
  45 => [(3, 5, 'yellow triangle')]
Number of pairs is 1 vs. unique pairs 1
```

# torch-nnet.py

The function of the basic neural network is reproduced within the PyTorch framework in this script.  All of the same command-line arguments are present:

```
$ python3 ./torch-nnet.py 3 5 -2 7
( 3, 5) =   45
  45 => yellow triangle @ (3,5)
Repeats:
  45 => [(3, 5, 'yellow triangle')]
Number of pairs is 1 vs. unique pairs 1
```

Note that the output of the hidden layer is not captured in this variant, hence the first line of output differs.
