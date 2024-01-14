import os
import argparse
from DeepION import DeepION_training,DeepION_predicting,DimensionalityReduction
from DownStreamTask import Co_localizedIONSearching,IsotopeDiscovery
import numpy as np

parser = argparse.ArgumentParser(
    description='DeepION for ion image representation of mass spectrometry imaging')

parser.add_argument('--input_Matrix',required= True,help = 'path to inputting MSI data matrix')
parser.add_argument('--input_PeakList',required= True,help = 'path to inputting MSI peak list')
parser.add_argument('--input_shape',required= True,type = int, nargs = '+', help='inputting MSI file shape')
parser.add_argument('--mode',
                    help = 'COL mode for co-localized ion searching, ISO mode for isotope ion discovery',
                    default= 'COL')
parser.add_argument('--ion_mode',
                    help = 'positive or negative ion mode that MSI experiment used',
                    required = True)
parser.add_argument('--num',
                    help = 'The number of searched co-localized ions for each ion',
                    default = 5)
parser.add_argument('--output_file', default='output/',help='output file name')

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    args = parser.parse_args()

    DeepION_training(args.input_Matrix, args.input_shape,args.mode)

    features = DeepION_predicting(args.input_Matrix, args.input_shape,args.mode)

    LD_features = DimensionalityReduction(features)

    # np.savetxt(args.ion_mode + '_feature_' + args.mode +'.txt',LD_features)

    if args.mode == 'COL':

        Co_localizedIONSearching(LD_features, args.input_PeakList, args.num, args.output_file)

    if args.mode == 'ISO':

        IsotopeDiscovery(args.input_Matrix, LD_features, args.input_PeakList, args.ion_mode, args.output_file)


