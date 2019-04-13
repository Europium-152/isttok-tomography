import sys
sys.path.append("../")
import reconstructPlasma
import reconstructPhantom

reconstructPhantom.phantom(7)

# Run in terminal: kernprof -l -v wrapper.py
# To read again: python -m line_profiler wrapper.py.lprof