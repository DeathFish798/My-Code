import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def Saveexcel(data,filename):
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1", float_format='%.3f')
    writer.save()

if __name__ == "__main__":
    data = pd.read_excel("附件1.xlsx")
    data =np.array(data)

    data = pd.DataFrame(data[:,2:])
    print("原始数据：",data,end="\n\n")

    #####归一化#####
    scaler = MinMaxScaler() #实例化
    result = scaler.fit_transform(data) #max_min归一化
    print("归一化：",result,end="\n\n")

    Saveexcel(result,"result_max_min.xlsx")

    #####反归一化#####
    result = scaler.inverse_transform(result)
    print("反归一化：",result,end="\n\n")

    Saveexcel(result,"result_invmax_min.xlsx")

    #####标准化#####
    scaler = StandardScaler() #实例化
    result = scaler.fit_transform(data)  # z_score标准化
    print("标准化：",result,end="\n\n")

    Saveexcel(result,"result_std.xlsx")

    #####反标准化#####
    result = scaler.inverse_transform(result)
    print("反标准化：",result,end="\n\n")

    Saveexcel(result,"result_invstd.xlsx")

    #####教材费#####
    a = 23.5 + 25.5 + 49 + 49+  38 + 49 + 56
    a = a* 0.85
    b = 23
    c = a + b
    print("教材费",c)