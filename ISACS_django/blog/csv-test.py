import csv

mycsv = csv.reader(open("./ISACS/labels.csv"))
next(mycsv)

for row in mycsv:
    print(row[1])