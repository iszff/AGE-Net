import numpy as np
# Usage:
# => Build a new folder named ClusterLabel
# => Move this code into the new Folder
# => Using "python Cluster.py" to run it.
# => Remove this code from the new Folder.
path = "/data/Eyediap_3D_face_rgb/Label/"
#length = 1181
length = 17692

filelists = [9, 6, 2, 15, 1, 4, 7, 16, 5, 11, 14, 10, 8, 3]
filelength = [3, 3, 4, 4]

begin = 0
for i in range(4):
  curlists = filelists[begin: begin+filelength[i]]
  begin = begin + filelength[i]
  print(curlists)
  contents = []
  for j in range(len(curlists)):
    with open( path + f"p{curlists[j]}.label") as infile:
      lines = infile.readlines()
      header = lines.pop(0)
      lines = lines[:length]
      contents += lines
  with open(path + "ClusterLabel/"  f"Cluster{i}.label", 'w') as outfile:
    outfile.write(header)
    for content in contents:
      outfile.write(content)