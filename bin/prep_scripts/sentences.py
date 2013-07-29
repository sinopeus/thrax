import sys, re

inp = open(sys.argv[1], "r")
out = open(sys.argv[2], "a")

pat = re.compile('\.')
line = inp.readline()
while line:
  segments = pat.split(line.rstrip('\n') + ' ')
  joined = '.\n'.join(segments)
  out.write(joined)
  line = inp.readline()
