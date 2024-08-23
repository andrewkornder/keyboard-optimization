ABOUT
-----

This is a program that optimizes a keyboard layout for a given corpus of text. It does so using any one of three metrics:
1. A metric based purely on finger movement developed by me.
2. An implementation of [Michael Dickens' metric](https://github.com/michaeldickens/Typing)
3. An implementation of [Carpalx's metric](https://mk.bcgsc.ca/carpalx/)


INSTALLATION
-----
This program is made with CUDA on windows. If you have an NVIDIA device and CUDA installed, you can build this project with CMake and nvcc.
The project provides six buildable targets which all use the same interface, the only difference being the metric and whether it de-duplicates the keyboards.
- dist: the first metric
- dist-ddup: the first metric with additional code to ensure that each keyboard in a generation is unique
- okp: the second metric
- okp-ddup: you get the idea
- carpalx
- carpalx-ddup


CONFIGURATION
-----
This program needs a configuration file to properly run. This is just a text file with a few fields that need to be filled out.
Each field is declared with an opening tag "<name>", a value, and a closing tag "</name>". For example,
```
<corpus>  
./text  
</corpus>
```
declares that the field `corpus` should be `./text/`.
Valid fields are
- `size` (required, >512): how many keyboards per generation. (will be rounded to power of 2)
- `survivors` (default = 0.5 * sqrt(size)): how many of those keyboards are used to create the next generation.
- `cache`: a directory to place cached corpora in. if not included, turns off caching.
- `generations` (default = 30): how many generations to run until finishing the evolution.
- `plateauLength` (default = 2): after this many generations without improvement, stop evolving.
- `rounds` (required): when using the `evolve` mode, decides how many keyboards to generate.
- `output` (default = "./output"): when using the `evolve` mode, decides where to put generated keyboard files.
- `seed` (default = random): the seed which drives the pRNG of the program (can be written in hex).
- `exportTables`: if provided, creates two files in this directory with the corpus and metric used.
- `corpus` (required): a folder to read text from. all files and subfolders will be read as text and
  used to train the keyboards.
- `movable` (default = all movable): which keys are allowed to be modified on the QWERTY keyboard.
  this should be 30 ones or zeros (1 = movable, 0 = left in place).

Of these 10 fields, only `corpus`, `size`, `rounds`, and `output` must be provided.  
The fields `size` and `surviving` will both get rounded up to the nearest power of two.

USAGE
-----
Once you have the executable, you can run it from the command line.
The first argument to the executable must be the path to your configuration file.
Any other arguments will be treated as modes to run.
There are 8 possibles modes:
- `help`: show the possible modes/configurable fields
- `perf`: profile various functions in the code (mostly here for debugging reasons)
- `rand`: compute the average score of a random keyboard
- `qwerty`: compute the score of the QWERTY layout
- `evolve`: evolve new keyboards and place them in the configured output directory
- `lock`: compute the cost of leaving each single key in its place
- `test`: test user keyboards given through stdin
- `test-common`: test a few alternative keyboards and QWERTY

Any mode can be repeated multiple times by following it with a number (e.g. "rand 100" runs "rand" 100 times).  
For example:  
`dist.exe configuration.txt qwerty rand 100` will do the following:
1. Load the configuration from `configuration.txt`
2. Print the score for QWERTY.
3. Print the average score of `size` random keyboards 100 times.
