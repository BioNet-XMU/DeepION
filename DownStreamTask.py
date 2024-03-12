from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import distance
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

i = None
cu_peak = None
adj_matrix_weigh = None
adj_matrix = None
return_charge_signal = None
return_signal = None

# 深度遍历
def graph_travel(G,x,marked):
    result=[]
    result.append(x)
    marked[x] = 1
    if sum(G[x,:])==1:
        return result
    else:
        for i in range(G.shape[1]):
            if G[x,i] != 0 and x!=i and marked[i]==0:
                result.extend(graph_travel(G,i,marked))
    return result

def output_max_ConnectMatrix(adj_matrix):

    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i, j] == 1:
                adj_matrix[j, i] = 1
    le = adj_matrix.shape[0]

    for i in range(le):
        adj_matrix[i, i] = 1
    # print(G[0])

    marked = [0] * le
    # print(marked)

    results = []
    for i in range(le):
        if marked[i] == 0:
            result = graph_travel(adj_matrix, i,marked)
            # print(result)
            results.append(result)
    return results

def intensity_thre(data1, data2):
    data1 = np.where(data1 > np.percentile(data1, 95), 1, 0)
    data2 = np.where(data2 > np.percentile(data2, 95), 1, 0)

    data3 = data1 + data2
    score = len(np.where(data3 == 2)[0]) / len(np.where(data3 != 0)[0])
    return score

def detect_iso(data, peak,peak_plus, mode, tolerate, type, charge,feature):

    global i,cu_peak,adj_matrix_weigh,adj_matrix,return_charge_signal,return_signal

    num = 0

    if mode == 'ppm':
        ind = np.where((peak <= (peak[i] + peak_plus) * (1000000 + tolerate) / 1000000) & (
                        peak >= (peak[i] + peak_plus) * (1000000 - tolerate) / 1000000))[0]

    if mode == 'Da':
        ind = np.where((peak <= (peak[i] + peak_plus) + tolerate) & (peak >= (peak[i] + peak_plus) - tolerate))[0]

    if len(ind) != 0:
        for j in ind:

            # weight = intensity_thre(data_filter[:, i], data_filter[:, j])
            # print(feature.shape)
            # weight = distance.euclidean(feature[i],feature[j])
            weight = distance.euclidean(feature[i],feature[j])
            if peak[j] in cu_peak and np.sum(data[:, i]) > np.sum(data[:, j]) and weight <= 0.25:
                print(peak[i],peak[j])
                print(weight)

                adj_matrix_weigh[i, j] = weight
                adj_matrix[i,j] = 1
                num = 1

                cu_peak.remove(peak[j])

                if charge == 1:
                    return_charge_signal[j] = 1
                if charge == 2:
                    return_charge_signal[j] = 2

                if type == 'C':
                    return_signal[j] = 1
                if type == 'S':
                    return_signal[j] = 2
                if type == 'Cl':
                    return_signal[j] = 3
                if type == 'K':
                    return_signal[j] = 4
                if type == 'Br':
                    return_signal[j] = 5

    return num

def return_adj(data,peak,ion_mode,feature):

    global i,cu_peak,adj_matrix_weigh,adj_matrix,return_charge_signal,return_signal

    adj_matrix = np.zeros((len(peak), len(peak)))
    adj_matrix_weigh = np.zeros((len(peak), len(peak)))

    cu_peak = list(peak)
    return_signal = np.zeros_like(peak)
    return_charge_signal = np.zeros_like(peak)
    for i in range(len(adj_matrix)):
    # for i in range(uu[0], uu[0]+2):

        for charge in [1, 2]:
            if charge == 1:
                if peak[i] in cu_peak:

                    num = detect_iso(data, peak, peak_plus=1.00336, mode='ppm', tolerate=10, type='C', charge=charge,feature = feature)

                    if num == 1:
                        num2 = detect_iso(data, peak, peak_plus=2.0067, mode='ppm', tolerate=10, type='C', charge=charge,feature= feature)

                        if num2 == 1:
                            num3 = detect_iso(data, peak, peak_plus=3.01008, mode='ppm', tolerate=10, type='C', charge=charge,feature= feature)

                            if num3 == 1:
                                num4 = detect_iso(data, peak, peak_plus=4.01344, mode='ppm', tolerate=10, type='C', charge=charge,feature= feature)

                    num = detect_iso(data, peak, peak_plus=1.9958, mode='ppm', tolerate=10, type='S', charge=charge,feature= feature)

                    if num == 1:
                        num2 = detect_iso(data, peak, peak_plus=1.9958 * 2, mode='ppm', tolerate=10, type='S', charge=charge,feature= feature)

                    if ion_mode == 'positive':

                        num = detect_iso(data, peak, peak_plus=1.99812, mode='ppm', tolerate=10, type='K', charge=charge,feature= feature)

                        if num == 1:
                            num2 = detect_iso(data, peak, peak_plus=1.99812 * 2, mode='ppm', tolerate=10, type='K', charge=charge,feature= feature)

                    if ion_mode == 'negative':

                        num = detect_iso(data, peak, peak_plus=1.99705, mode='ppm', tolerate=10, type='Cl', charge=charge,feature= feature)

                        if num == 1:
                            num2 = detect_iso(data, peak, peak_plus=1.99705 * 2, mode='ppm', tolerate=10, type='Cl', charge=charge,feature= feature)

                        num = detect_iso(data, peak, peak_plus=1.99795, mode='ppm', tolerate=10, type='Br', charge=charge,feature= feature)

                        if num == 1:
                            num2 = detect_iso(data, peak, peak_plus=1.99795 * 2, mode='ppm', tolerate=10, type='Br', charge=charge,feature= feature)

            if charge == 2 :
                if peak[i] in cu_peak:

                    num = detect_iso(data, peak, peak_plus=0.50168,mode='ppm', tolerate=10, type='C', charge=charge,feature= feature)

                    if num == 1:
                        num2 = detect_iso(data, peak, peak_plus=0.50168 * 2, mode='ppm', tolerate=10, type='C', charge=charge,feature= feature)

                        if num2 == 1:
                            num3 = detect_iso(data, peak, peak_plus=0.50168 * 3, mode='ppm', tolerate=10, type='C', charge=charge,feature= feature)

                            if num3 == 1:
                                num4 = detect_iso(data, peak, peak_plus=0.50168 * 4, mode='ppm', tolerate=10, type='C',charge=charge,feature= feature)

        if peak[i] in cu_peak:
            cu_peak.remove(peak[i])

    return adj_matrix,return_signal,return_charge_signal,adj_matrix_weigh


def output_results(filename,results,peak,adj_matrix,return_signal,return_charge_signal,adj_matrix_weigh):

    adj_iso = adj_matrix

    peak_list = peak
    peak_list = peak_list[np.argsort(peak_list)]

    max_num = np.max([len(i) for i in results])

    for i in range(max_num):
        globals()['df%d' % i] = []

    for kk in results:
        for num in range(1, max_num + 1):
            if len(kk) == num:
                for i in range(len(kk)):
                    globals()['df%d' % i].append(peak_list[kk][i])
                for j in range(len(kk), max_num):
                    # print(peak_list[kk])
                    # print(peak_list[kk][j])
                    globals()['df%d' % j].append(0)

    data = np.zeros((len(df0), max_num + (max_num - 1) * 2))

    for i in range(max_num):
        data[:, i] = globals()['df%d' % i]

    for i in range(max_num, max_num + (max_num - 1) * 2):
        globals()['df%d' % i] = []

    for i in range(max_num - 1):
        for j in range(len(df0)):
            if globals()['df%d' % (i + 1)][j] != 0:
                signal = return_signal[np.where(peak_list == globals()['df%d' % (i + 1)][j])[0]]

                # if signal == 1:
                #     locals()['df%d'%(i + max_num)].append('C')
                # if signal == 2:
                #     locals()['df%d' % (i + max_num)].append('S')
                # if signal == 3:
                #     locals()['df%d' % (i + max_num)].append('C')
                # if signal == 4:
                #     locals()['df%d' % (i + max_num)].append('K')
                # if signal == 5:
                #     locals()['df%d' % (i + max_num)].append('B')

                globals()['df%d' % (i + max_num)].append(signal)
            else:

                globals()['df%d' % (i + max_num)].append(0)

    for i in range(max_num - 1):
        for j in range(len(df0)):
            if globals()['df%d' % (i + 1)][j] != 0:
                signal = return_charge_signal[np.where(peak_list == globals()['df%d' % (i + 1)][j])[0]]

                # if signal == 1:
                #     locals()['df%d'%(i + max_num)].append('C')
                # if signal == 2:
                #     locals()['df%d' % (i + max_num)].append('S')
                # if signal == 3:
                #     locals()['df%d' % (i + max_num)].append('C')
                # if signal == 4:
                #     locals()['df%d' % (i + max_num)].append('K')
                # if signal == 5:
                #     locals()['df%d' % (i + max_num)].append('B')

                globals()['df%d' % (i + max_num + max_num - 1)].append(signal)
            else:

                globals()['df%d' % (i + max_num + max_num - 1)].append(0)

    for i in range(max_num, max_num + (max_num - 1) * 2):
        data[:, i] = globals()['df%d' % i]

    columns = ['monoisotope']
    for i in range(1, max_num):
        columns.append('isotope_%d' % i)

    for i in range(max_num, max_num * 2 - 1):
        columns.append('isotope_type_%d' % (i - max_num + 1))

    for i in range(max_num * 2 - 1, max_num * 2 - 1 + (max_num - 1)):
        columns.append('charge_%d' % (i - max_num * 2 + 1 + 1))

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(filename, index=False)

def output_moin(results,data,peak,filename_peak,filename_data):

    moin = np.array([k[0] for k in results])

    data_moin = data[:, moin]

    peak = peak[moin]
    np.savetxt('uuu', moin)
    data_moin = sparse.bsr_matrix(data_moin)
    sparse.save_npz(filename_data, data_moin)
    np.savetxt(filename_peak, peak, delimiter=',')


def Co_localizedIONSearching(return_feature, peak,num,output_file):
    peak = np.loadtxt(peak)
    tree = KDTree(return_feature)
    num = int(num)
    columns = ['querry ion']
    for i in range(num):
        columns.append('Co-localized ion %d' % i)

    data = np.zeros((len(peak), num+1))

    for i in range(len(peak)):
        dist, ind = tree.query(return_feature[np.where(peak == peak[i])[0]].reshape(1, -1), k=num + 1)
        data[i] = peak[ind[0]]

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(output_file, index=False)

def IsotopeDiscovery(oridata, return_feature, peak, ion_mode,output_file):

    data = np.loadtxt(oridata)
    peak = np.loadtxt(peak)

    adj_matrix, return_signal, return_charge_signal, adj_matrix_weigh = return_adj(data, peak, ion_mode,
                                                                                   return_feature)

    results = output_max_ConnectMatrix(adj_matrix)

    output_results(output_file, results, peak, adj_matrix, return_signal, return_charge_signal, adj_matrix_weigh)

# return_feature = np.loadtxt('Pos_feature_COL')
# peak =np.loadtxt('DATASET/Pos_brain_data_peak.csv')
# output_file = 'output.csv'
# Co_localizedIONSearching(return_feature, peak,5,output_file)

