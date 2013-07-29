from bs4 import BeautifulSoup
import sys

soup = BeautifulSoup(open(sys.argv[1]))
dest = open(sys.argv[1] + ".txt", 'w')
dest.write(soup.body.get_text())
dest.close()
