import re
import os

gnaf_trig = 'C:/Users/u62231/Desktop/Projects/loci/gnaf_current.trig'
gnaf_turtle = 'C:/Users/u62231/Desktop/Projects/loci/datasetsLinksetsTurtle/gnaf_current.ttl'

asgs2016_trig = "C:/Users/u62231/Desktop/Projects/loci/original_datasetsLinksets/asgs2016.trig"
asgs2016_turtle = "C:/Users/u62231/Desktop/Projects/loci/datasetsLinksetsTurtle/asgs2016.ttl"

def TrigToTurtleSeekMethod(filename):
    with open(filename, 'rb+') as filehandler:
        filehandler.seek(-1, os.SEEK_END)
        filehandler.truncate()


def TrigToTurtleSeekMethod(number, file):
    count = 0

    with open(file, 'r+b', buffering=0) as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        while f.tell() > 0:
            f.seek(-1, os.SEEK_CUR)
            print(f.tell())
            char = f.read(1)
            if char != b'\n' and f.tell() == end:
                print("No change: file does not end with a newline")
                exit(1)
            if char == b'\n':
                count += 1
            if count == number + 1:
                f.truncate()
                print("Removed " + str(number) + " lines from end of file")
                exit(0)
            f.seek(-1, os.SEEK_CUR)

    if count < number + 1:
        print("No change: requested removal would leave empty file")
        exit(3)


def TrigToTurtle(input_file, output_file):
    file_last_line_index = GetFileLastLineIndex(input_file)
    print("last line is: ")
    print(file_last_line_index)
    with open(input_file, 'r') as input:
        with open(output_file, 'w') as output:
            print("now writting..")
            i = 0
            for line in input:
                if i == 0:
                    print("skipping first line")
                    print(line)
                    pass
                elif i == file_last_line_index:
                    print("skipping last line")
                    print(line)
                    pass
                else:
                    output.writelines(line)
                i = i + 1



        # data = fin.read().splitlines(True)
        # print(data[0])
        # print(data[-1])
        #
        #
        # fout.writelines(data[1:-1])
        # print(data[0])
        # print(data[-1])


def GetFileLastLineIndex(input_file):
    with open(input_file) as f:
        print("getting file last line index...")
        for i, l in enumerate(f):
            pass
    return i

def CheckFirstLine():
    with open('C:/Users/u62231/Desktop/Projects/loci/original_datasetsLinksets/gnaf_201605_addressSites_instances.nt', 'r') as fin:
        for i, line in enumerate(fin):
            if i > 2:
                pass
            else:
                print(line)


def add_synthetic_triples_to_datasets(input_name, output_name):
    with open(input_name, 'r') as input:
        with open(output_name, 'w') as output:
            for line in input:
                # write existing triple
                #output.write(line)

                # search for obj or sub to duplicate triple
                if re.search(r"http://linked.data.gov.au/dataset", line):
                    replace = (re.search(r'<http://linked.data.gov.au/dataset(.*?)>', line).group())
                    replace = replace[:-1]
                    replace = replace + "_1>"
                    line = re.sub(r'<http://linked.data.gov.au/dataset(.*?)>', replace, line)
                elif re.search(r"http://linked.data.gov.au/linkset", line):
                    replace = (re.search(r'<http://linked.data.gov.au/linkset(.*?)>', line).group())
                    replace = replace[:-1]
                    replace = replace + "_1>"
                    line = re.sub(r'<http://linked.data.gov.au/dataset(.*?)>', replace, line)
                elif re.search(r"_:", line):
                    replace = (re.search(r'_:(.*?) ', line).group())
                    replace = replace[:-1]
                    replace = replace + "_1 "
                    line = re.sub(r'_:(.*?) ', replace, line)

                # write new additional triple
                output.write(line)

            print("new")
            print(line)

def add_synthetic_triples_to_linksets(input_name, output_name):
    with open(input_name, 'r') as input:
        with open(output_name, 'w') as output:
            for line in input:
                # write existing triple
                # output.write(line)

                # search for obj or sub to duplicate triple
                if re.search(r":[a-zA-Z0-9_](.*?) ", line):
                    replace = (re.search(r':[a-zA-Z0-9_](.*?) ', line).group())
                    replace = replace[:-1]
                    replace = replace + "_1 "
                    line = re.sub(r':[a-zA-Z0-9_](.*?) ', replace, line)
                elif re.search(r"@prefix l: <http://linked.data.gov.au/dataset/(.*?)> ", line):
                    replace = (re.search(r'@prefix l: <http://linked.data.gov.au/dataset/(.*?)>', line).group())
                    replace = replace[:-1]
                    replace = replace + "_1> "
                    line = re.sub(r'@prefix l: <http://linked.data.gov.au/dataset/(.*?)> ', replace, line)
                else:
                    pass

                # write new additional triple
                output.write(line)

    print("new")
    print(line)
            # if re.search("http://linked.data.gov.au/dataset", line):
            #     print(line)
            #     newline = line.find(">")
            #     line[newline               print(newline)


            #     print(line)
            #     match = re.match("http://linked.data.gov.au/dataset[.*?]", line)
            #     print("match")
            #     print(match)
            #     print(line)
            #     #new_line = re.sub("http://linked.data.gov.au/dataset\.+")
            #     print(new_line)
            # #
            # line_as_list = line.split()
            # print(line_as_list)
            # for
            # for i in line_as_list:
            #     if re.search("http://linked.data.gov.au/dataset", i):
            #         print(i)
            #         newi = i[:-1]
            #         i = "{}_1>".format(newi)
            #         print(i)
            #     if re.search("_:", i):
            #         new_i = "{}_1>".format(i)
            #         print('yes')
            # print("brillant")
            # print(line_as_list)
            #  re.sub('http://linked.data.gov.au/dataset')

def ReadFirstFewLines():
    with open('C:/Users/u62231/Desktop/Projects/loci/gnaf_current.trig', 'r') as fin:
        line = fin.readlines(0)
        print(line)
        print(fin.readlines(1))


#TrigToTurtle(asgs2016_trig, asgs2016_turtle)

#GetFileLen(gnaf_trig)
#CheckFirstLine()
#add_synthetic_triples()

#directory_input = "C:/Users/u62231/Desktop/Projects/loci/original_datasetsLinksets/test"
#directory_output = "C:/Users/u62231/Desktop/Projects/loci/original_datasetsLinksets/test"
#add_synthetic_triples_to_linksets(directory_input, directory_inputdirectory_inputdirectory_input)
directory_input = "C:\\Users\\u62231\\Desktop\\Projects\\loci\\datasetsLinksetsTurtle"
directory_output = "C:\\Users\\u62231\\Desktop\\Projects\\loci\\additonal_triples"


for filename in os.listdir(directory_input):
    if filename.endswith(".nt") or filename.endswith(".ttl") or filename.endswith(".trig"):
        print("################################################################")
        input_name = "{}/{}".format(directory_input, filename)
        print(input_name)
        output_name = "{}/trips_plus_{}".format(directory_output, filename)
        print(output_name)
        add_synthetic_triples_to_linksets(input_name, output_name)

         # print(os.path.join(directory, filename))



#
# for line in test_string:
#     print(line)
#     if re.search(r":(.*?) ", line):
#         print("found")
#         replace = (re.search(r':(.*?) ', line).group())
#         replace = replace[:-1]
#         replace = replace + "_1 "
#         line = re.sub(r':(.*?) ', replace, line)
#         print(line)
#     else:
#         pass
