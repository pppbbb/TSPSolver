#!/usr/bin/python
import os

VC_BIN_DIR = '"C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\\bin"'

files_list = os.listdir("./")
cpp_list = list()
cu_list = list()

for file in files_list:
    if ".cpp" in str(file):
        cpp_list.append(str(file))
    if ".cu" in str(file):
        cu_list.append(str(file))

print(cpp_list)
print(cu_list)

f = open("make.bat", "w")
for file in cu_list:
    f.write('nvcc -ptx -ccbin ' + VC_BIN_DIR + ' -o ' + file.split('.')[0] + '.ptx ' + file + "\n")

f.write("nvcc -o a.exe")
for file in cpp_list:
    f.write(" " + file)
f.write(' cuda.lib -ccbin '+ VC_BIN_DIR +'\n')