# career-day-2025-demo

Materials associated with a demo presented at the Tome School 2025 Career Day.  This project uses a simple integer-based neural network to identify colored polygons:

1. Inputs are the number of sides to a polygon, ___s___ and its color, ___c___.
2. The color, ___c___, is expressed as a 3-bit value where:
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

The code in the basic script has no external dependencies and implements the network layers as Python functions.  Lacking any command-line arguments, the script iterates over all combinations of ___s___ and ___c___ to display their mappings via the neural network.

A single command-line argument is interpreted as an alternative value for the second weighting parameter, ___w2___ in the output layer.  The default, 5, produces some mistaken mappings in the network; the value of 8 successfully distinguishes all combinations.

With two command-line arguments, an explicit singular value of ___s___ and ___c___ are assumed.  This pair alone is presented to the network and analyzed.

A third command-line argument augments the explicit ___s___ and ___c___ with a value for ___w2___.

A fourth command-line argument augments the explicit ___s___ and ___c___ with values for ___w1___ and ___w2___, respectively.

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

With output layer weights ___w1___ and ___w2___ of -2 and 7, a yellow (5) triangle (3) maps to 45:

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

# Full mapping

## With miscategorization (w2 = 5)

The full enumeration of ___s___ and ___c___ mappings when ___w2___ is 5:

```
$ python3 ./torch-nnet.py 5
   :
   0 => [(3, 0, 'black triangle')]
   2 => [(3, 1, 'red triangle')]
   4 => [(3, 2, 'blue triangle')]
   6 => [(3, 3, 'purple triangle')]
   8 => [(3, 4, 'green triangle'), (4, 0, 'black square')]
  10 => [(3, 5, 'yellow triangle'), (4, 1, 'red square')]
  12 => [(3, 6, 'cyan triangle'), (4, 2, 'blue square')]
  14 => [(3, 7, 'white triangle'), (4, 3, 'purple square')]
  16 => [(4, 4, 'green square'), (5, 0, 'black pentagon')]
  18 => [(4, 5, 'yellow square'), (5, 1, 'red pentagon')]
  20 => [(4, 6, 'cyan square'), (5, 2, 'blue pentagon')]
  22 => [(4, 7, 'white square'), (5, 3, 'purple pentagon')]
  24 => [(5, 4, 'green pentagon'), (6, 0, 'black hexagon')]
  26 => [(5, 5, 'yellow pentagon'), (6, 1, 'red hexagon')]
  28 => [(5, 6, 'cyan pentagon'), (6, 2, 'blue hexagon')]
  30 => [(5, 7, 'white pentagon'), (6, 3, 'purple hexagon')]
  32 => [(6, 4, 'green hexagon'), (7, 0, 'black heptagon')]
  34 => [(6, 5, 'yellow hexagon'), (7, 1, 'red heptagon')]
  36 => [(6, 6, 'cyan hexagon'), (7, 2, 'blue heptagon')]
  38 => [(6, 7, 'white hexagon'), (7, 3, 'purple heptagon')]
  40 => [(7, 4, 'green heptagon'), (8, 0, 'black octagon')]
  42 => [(7, 5, 'yellow heptagon'), (8, 1, 'red octagon')]
  44 => [(7, 6, 'cyan heptagon'), (8, 2, 'blue octagon')]
  46 => [(7, 7, 'white heptagon'), (8, 3, 'purple octagon')]
  48 => [(8, 4, 'green octagon')]
  50 => [(8, 5, 'yellow octagon')]
  52 => [(8, 6, 'cyan octagon')]
  54 => [(8, 7, 'white octagon')]
Number of pairs is 48 vs. unique pairs 28
```

## No miscategorization (w2 = 8)

The full enumeration of ___s___ and ___c___ mappings when ___w2___ is 8:

```
$ python3 ./torch-nnet.py 8
   :
   0 => [(3, 0, 'black triangle')]
   5 => [(3, 1, 'red triangle')]
  10 => [(3, 2, 'blue triangle')]
  11 => [(4, 0, 'black square')]
  15 => [(3, 3, 'purple triangle')]
  16 => [(4, 1, 'red square')]
  20 => [(3, 4, 'green triangle')]
  21 => [(4, 2, 'blue square')]
  22 => [(5, 0, 'black pentagon')]
  25 => [(3, 5, 'yellow triangle')]
  26 => [(4, 3, 'purple square')]
  27 => [(5, 1, 'red pentagon')]
  30 => [(3, 6, 'cyan triangle')]
  31 => [(4, 4, 'green square')]
  32 => [(5, 2, 'blue pentagon')]
  33 => [(6, 0, 'black hexagon')]
  35 => [(3, 7, 'white triangle')]
  36 => [(4, 5, 'yellow square')]
  37 => [(5, 3, 'purple pentagon')]
  38 => [(6, 1, 'red hexagon')]
  41 => [(4, 6, 'cyan square')]
  42 => [(5, 4, 'green pentagon')]
  43 => [(6, 2, 'blue hexagon')]
  44 => [(7, 0, 'black heptagon')]
  46 => [(4, 7, 'white square')]
  47 => [(5, 5, 'yellow pentagon')]
  48 => [(6, 3, 'purple hexagon')]
  49 => [(7, 1, 'red heptagon')]
  52 => [(5, 6, 'cyan pentagon')]
  53 => [(6, 4, 'green hexagon')]
  54 => [(7, 2, 'blue heptagon')]
  55 => [(8, 0, 'black octagon')]
  57 => [(5, 7, 'white pentagon')]
  58 => [(6, 5, 'yellow hexagon')]
  59 => [(7, 3, 'purple heptagon')]
  60 => [(8, 1, 'red octagon')]
  63 => [(6, 6, 'cyan hexagon')]
  64 => [(7, 4, 'green heptagon')]
  65 => [(8, 2, 'blue octagon')]
  68 => [(6, 7, 'white hexagon')]
  69 => [(7, 5, 'yellow heptagon')]
  70 => [(8, 3, 'purple octagon')]
  74 => [(7, 6, 'cyan heptagon')]
  75 => [(8, 4, 'green octagon')]
  79 => [(7, 7, 'white heptagon')]
  80 => [(8, 5, 'yellow octagon')]
  85 => [(8, 6, 'cyan octagon')]
  90 => [(8, 7, 'white octagon')]
Number of pairs is 48 vs. unique pairs 48
```
