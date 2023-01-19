import numpy as np
import pandas as pd

def Saveexcel(data,filename):
    '''
        save excel
    '''
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1",float_format="%.2f")
    writer.save()

if __name__ == "__main__":
    #####problem 1#####
    sheet1 = pd.read_csv("influence_data.csv")  #sorted by influer
    data1 = np.array(sheet1)
    print(data1)

    num0 = 335
    count = 1
    for i in range(len(data1)):
        if data1[i][0] != num0:
            num0 = data1[i][0]
            count += 1
    print(count)

    infl = []
    wtl = np.zeros([count,4])

    type0 = set()
    num0 = 0
    count = -1
    #influer data
    for i in range(len(data1)):
        if i == 0:
            num0 = data1[i][0]
            count += 1
            type0 = set()
        elif data1[i][0] != num0:
            inf0 = []
            inf0.append(data1[i - 1][0])
            inf0.append(data1[i - 1][1])
            inf0.append(data1[i - 1][2])
            inf0.append(data1[i - 1][3])
            inf0.append(len(type0))
            infl.append(inf0)
            num0 = data1[i][0]
            count += 1
            type0 = set()
            time0 = 0
        #type
        type0.add(data1[i][6])
        #last
        if i == len(data1) - 1:
            inf0 = []
            inf0.append(data1[i][0])
            inf0.append(data1[i][1])
            inf0.append(data1[i][2])
            inf0.append(data1[i][3])
            inf0.append(len(type0))
            infl.append(inf0)

    infl = np.array(infl)
    print(infl)

    #calculate weight
    num0 = 0
    count = -1
    weight = []
    for i in range(len(data1)):
        if i == 0:
            num0 = data1[i][0]
            count += 1
            w = 0
        elif data1[i][0] != num0:
            num0 = data1[i][0]
            count += 1
            weight.append(w)
            w = 0
        #genre
        if data1[i][6] == infl[count][2]:
            t = 1
        else:
            t = int(infl[count][4])
        #time span
        l = abs(int(data1[i][7]) - int(infl[count][3]))
        l = 1.4 ** (l / 10)
        w = int(t) * l + w
        #last
        if i == len(data1) - 1:
            weight.append(w)

    print(weight)
    Saveexcel(weight,"weight.xlsx")
    Saveexcel(infl,"influence.xlsx")

    #calculate edge
    edge = []
    num0 = 0
    count = -1
    for i in range(len(data1)):
        if i == 0:
            num0 = data1[i][0]
            count += 1
        elif data1[i][0] != num0:
            num0 = data1[i][0]
            count += 1
        # genre
        if data1[i][6] == infl[count][2]:
            t = 1
        else:
            t = infl[count][4]
        # time span
        l = abs(int(data1[i][7]) - int(infl[count][3]))
        l = 1.4 ** (l / 10)
        w = int(t) * l
        # save
        edge0 = []
        edge0.append(data1[i][0])
        edge0.append(data1[i][4])
        edge0.append(w)
        edge.append(edge0)

    Saveexcel(edge,"edge.xlsx")
