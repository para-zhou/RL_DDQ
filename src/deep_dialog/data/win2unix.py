'''
Convert win version data to unix version
'''
import pickle


def win2unix(file_path):
    original = file_path
    destination = file_path[:-2] + 'unix.p'
    
    content = ''
    outsize = 0
    with open(original, 'rb') as infile:
        content = infile.read()
    with open(destination, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))
    
    return destination
