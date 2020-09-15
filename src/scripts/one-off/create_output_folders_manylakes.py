import os
import re
directory = '../../../data/raw/figure3' #unprocessed data directory
lnames = set()
os.mkdir("../manylakes/outputs1")
os.mkdir("../manylakes/outputs2")
os.mkdir("../manylakes/outputs3")
os.mkdir("../manylakes/labels")
for filename in os.listdir(directory):
    #parse lakename from file
    m = re.search(r'^nhd_(\d+)_test_train.*', filename)
    if m is None:
        continue
    name = m.group(1)
    if name not in lnames:
        os.mkdir("../manylakes/outputs1/"+name)
        os.mkdir("../manylakes/outputs2/"+name)
        os.mkdir("../manylakes/outputs3/"+name)
        os.mkdir("../manylakes/labels/"+name)