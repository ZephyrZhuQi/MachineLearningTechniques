from sys import argv

res = []
with open(argv[1],"r") as f:
    data = f.read().splitlines()
    for line in data:
        line = [x for x in str(line).split()]
        res_row = []
        if((float(line[0]))==int(argv[3])):
            res_row.append(1)
        else:
            res_row.append(-1)
        res_row.append(line[1])
        res_row.append(line[2])
        res.append(res_row)

with open(argv[2],"w") as format_data:
    for i in range(len(res)):
        format_data.write(str(res[i][0]))
        format_data.write(" ")
        format_data.write(str(res[i][1]))
        format_data.write(" ")
        format_data.write(str(res[i][2])+"\n")