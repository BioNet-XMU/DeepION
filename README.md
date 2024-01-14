# DeepION
DeepION is a deep learning-based low-dimensional representation model of ion images for mass spectrometry imaging. In this model, two modes of DeepION, denoted as “COL” and “ISO” are designed for the cases of regular co-localized ions from different molecules and isotope ions from a same molecule respectively. Developer is Lei Guo from Laboratory of Biomedical Network, Department of Electronic Science, Xiamen University of China.

# Overview of DeepION
<div align=center>
<img src="https://github.com/gankLei-X/DeepION/assets/70273368/e03431ce-3d39-4277-86b9-66a98c3eea73" width="800" height="480" /><br/>
</div>

__Schematic overflow of the DeepION model__. The input original ion image is first augmented to two views using the augmentation operation "T" , where “COL” mode used the augmentation combination of color jitter, filtering, Poisson noise, random missing, ‘ISO’ mode conduct the intensity-dependent missing additively. Then, encoder module utilizes ResNet18 to capture the relevant spatial patterns from two augmented views to generate the vector representation. The projection module and prediction module ensure the encoder module achieving stable and meaningful ion image representation, in which multi-layer perceptron is applied. These three modules are trained to maximize similarity using the contrastive loss. After training is completed, the dimensionality reduction module is implemented by the UMAP algorithm to achieve the final representation O for downstream tasks. 

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
