import runpy
import sys
from io import StringIO

import importlib_resources
import pkg_resources
from importlib_resources import path

from benchmarks.benchmark_utils import swap_attr

print("GRRRRRRRR run_all_examples.py")

for entry in [entry for entry in importlib_resources.contents('examples') if
              not pkg_resources.resource_isdir('examples', entry)
              and entry.endswith(".py")
              and '__init__' not in entry
              and 'demo.py' not in entry
              and 'DELETE' not in entry
              ]:
    with path('examples', entry) as file_in:
        print("")
        print("")

        print("")
        print("*****************************************************************")
        print("Running {}".format(entry))
        print("*****************************************************************")
        with swap_attr(sys, "stdin", StringIO("q")):
            try:
                runpy.run_path(file_in, run_name="__main__", init_globals={
                    'argv': ['--sleep-for-animation=False']
                })
            except Exception as e:
                print(e)
            print("runpy done.")
        print("Done with {}".format(entry))
