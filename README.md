ABOUT
-----

This is a program that optimizes a keyboard layout for a given corpus of text. It does so using any one of three metrics:
1. A metric based purely on finger movement developed by me.
2. An implementation of [Michael Dickens' metric](https://github.com/michaeldickens/Typing)
3. An implementation of [Carpalx's metric](https://mk.bcgsc.ca/carpalx/)


INSTALLATION
-----
This program is made with CUDA on windows. If you have an NVIDIA device and CUDA installed, you can build this project with CMake and nvcc.
The project provides three buildable targets which all use the same interface, the only difference being the metric used.
- distance
- dickens
- carpalx  

The names should be self-explanatory.


CONFIGURATION
-----
This program needs a configuration file to properly run. This is just a text file with a few fields that need to be filled out.
Each field is declared with the name and a value separated by an equals sign. Every field should be placed on different lines.  

For example, the line `corpus = ./text/` declares that the field `corpus` should be `"./text/"`.  
A line can be commented out by starting it with '#'. Ending a line with a backslash ('\') continues the line.

A field can be one of the following types of values:
1. A boolean value represented by "on" or "off".
2. An integer represented by a series of digits (anything besides the digits 0-9 will be ignored).
3. A string of arbitrary text.
4. A path-like string (e.g. "./folder/file.txt" or "C:/folder/*.txt").

Valid fields are
- `size` (required, >512): How many keyboards per generation.
- `survivors` (default = sqrt(0.5 * size)): How many of those keyboards are used to create the next generation.
- `cache`: A directory to place cached corpora in. If not included, turns off caching.
- `generations` (default = 30): How many generations to run until finishing the evolution.
- `plateauLength` (default = 2): After this many generations without improvement, stop evolving.
- `rounds` (required): When using the `evolve` mode, decides how many keyboards to generate.
- `duplicates` (default = on): Whether to tolerate duplicate keyboards in each generation. Deduplication is slow,
  and offers only marginal improvements, so I suggest leaving duplicates on.
- `output` (default = "./output"): When using the `evolve` mode, decides where to put generated keyboard files.
- `seed` (default = random): The seed which drives the pRNG of the program (can be written in hex).
- `exportTables`: If provided, creates two files in this directory with the corpus and metric used.
- `corpus` (required): A file or glob pattern to read text from. If this is a pattern, all matching files will be read
  as text and used to train the keyboards. Otherwise, it will read the entire file.
- `movable` (default = all movable): Which keys are allowed to be modified on the QWERTY keyboard.
  this should be a list of every key which is allowed to move (e.g. "abcdef,.").
- `showOutput` (default = off): Whether to show each generated keyboard while running `evolve`.

Of these fields, only `corpus`, `size`, `rounds`, and `output` must be provided.  
The fields `size` and `surviving` will both get rounded up to the nearest power of two.

USAGE
-----
Once you have the executable, you can run it from the command line.
The first argument to the executable must be the path to your configuration file.
Any other arguments will be treated as modes to run.
There are 7 possibles modes:
- `help`: Show the possible modes/configurable fields.
- `perf`: Profile various functions in the code (mostly here for debugging reasons).
- `rand`: Compute the average score of a random keyboard.
- `evolve`: Evolve new keyboards and place them in the configured output directory.
- `lock`: Compute the cost of leaving each single key in its place.
- `test`: Test user keyboards given through stdin.
- `test-all`: Test a few alternative keyboards and QWERTY.

Additionally, using the name of a common keyboard layout will compute the score of that layout on this metric.
Valid layouts are:
- qwerty
- alphabet
- dvorak
- colemak
- carpalx
- arensito
- asset
- capewell

Any mode can be repeated multiple times by following it with a number (e.g. "rand 100" runs "rand" 100 times).  
For example:  
`dist.exe configuration.txt qwerty rand 3` will do the following:
1. Load the configuration from `configuration.txt`
2. Print the score for QWERTY.
3. Show the average score of `size` random keyboards for three trials.
