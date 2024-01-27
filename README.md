# DeepION
DeepION is a deep learning-based low-dimensional representation model of ion images for mass spectrometry imaging. In this model, two modes of DeepION, denoted as “COL” and “ISO” are designed for the cases of regular co-localized ions from different molecules and isotope ions from a same molecule respectively. Developer is Lei Guo from Laboratory of Biomedical Network, Department of Electronic Science, Xiamen University of China.

# Overview of DeepION
<div align=center>
<img src="https://github.com/gankLei-X/DeepION/assets/70273368/7aab5832-d2e9-448c-838c-6f697993aa2f" width="800" height="480" /><br/>
</div>

__Schematic overview of DeepION consisting of four modules.__ (1) Data Augmentation module. The original ion image is first imported into the data augmentation module "T"  to generate two augmented images, where the T_COL including color jitter, filtering, Poisson noise, and random missing value is carried in COL mode, while T_ISO introduces an additional process of intensity-dependent missing value in ISO mode. (2) Encoder module. Two augmented images are propagated through a pair of ResNet18-based encoders that shared parameters, then output two 512-dimensional representation vectors. (3) Projection module and Prediction module are used to avoid collapsing solutions during the optimization process of maximizing the similarity between two augmentations from a same image and ensure to learn the meaningful representation vectors. A contrastive loss is employed to maximize similarity with a stop-gradient operation to prevent collapsing during training. (4) Dimensionality Reduction module. This module is applied to further reduce the dimensionality of ion image representation to a 20-dimensional vector O for downstream tasks. 


# Requirement

    python == 3.5, 3.6 or 3.7

    scikit-learn == 0.19.1
    
    pytorch == 1.8.2
        
    numpy >= 1.8.0
    
    umap == 0.5.1

    typing == 3.6.4

    torchvision == 0.11.3
    
    kornia == 0.4.1 
    
    boly_pytorch

# Quickly start

## Input
(1) The preprocessed MSI matrix data with two-dimensional shape [X*Y,P], where X and Y represent the pixel numbers of horizontal and vertical coordinates of MSI data, and P represents the number of ions.

(2) The preprocessed MSI peak data with one-dimensional shape [P,1], where P represents the number of ions.

(3) The inputing MSI data shape.

(4) "COL" mode or "ISO" mode that selected.

(5) "positive" or "negative" ion mode that MSI experiments used.

(6) The number of searched co-locolized ions for each ion.

(7) The filename of output.

## Run DeepION model

cd to the DeepION fold

If you want to perfrom DeepION for co-localized ion searching, taking rat brain section in positive mode as an example, run:

    python run.py --input_Matrix .../DATASET/Pos_brain_data_matrix.txt --input_PeakList .../DATASET/Pos_brain_data_peak.csv --input_shape 198 422 --mode COL --ion_mode positive --num 5 --output_file Pos_COL_result
    
If you want to perfrom DeepION for isotope discovery, taking rat brain section in positive mode as an example, run:

    python run.py --input_Matrix .../DATASET/Pos_brain_data_matrix.txt --input_PeakList .../DATASET/Pos_brain_data_peak.csv --input_shape 198 422 --mode ISO --ion_mode positive --output_file Pos_ISO_result

# Acknowledge

We thank the package developers of "boly_pytorch" and "kornia"

# Contact

Please contact me if you need any help: gl5121405@gmail.com
