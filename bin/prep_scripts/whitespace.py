import sys, re

inp = open(sys.argv[1], 'r')
out = open(sys.argv[2], 'a')

line = inp.readline()

while line:
    out.write(' '.join(line.split()) + '\n')
    line = inp.readline()

inp.close()
out.close()
