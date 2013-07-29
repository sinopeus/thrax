"""
Split a corpus into a training corpus and a validation corpus. Takes five
arguments: an input file, a training output file, a validation output file, a
splitting ratio, and a seed for the random number generator.
"""
import sys
import random

input_file = sys.argv[1]
output_file1 = sys.argv[2]
output_file2 = sys.argv[3]

try:
    P = float( sys.argv[4] )
except IndexError:
    P = 0.9
try:
    seed = sys.argv[5]
except IndexError:
    seed = None

if seed:
    random.seed( seed )

i = open(input_file)
o1 = open(output_file1, 'a+b')
o2 = open(output_file2, 'a+b')

for line in iter(i):
    r = random.random()
    if r > P:
        o2.write(bytes(line, "utf-8"))
    else:
        o1.write(bytes(line, "utf-8"))

