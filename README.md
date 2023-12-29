# DeepION
DeepION is a deep learning-based low-dimensional representation model of ion images for mass spectrometry imaging. In this model, two modes of DeepION, denoted as “COL” and “ISO” are designed for the cases of regular co-localized ions from different molecules and isotope ions from a same molecule respectively. Developer is Lei Guo from Laboratory of Biomedical Network, Department of Electronic Science, Xiamen University of China.

# Overview of DeepION
<div align=center>
<img src="https://github.com/gankLei-X/DeepION/assets/70273368/e03431ce-3d39-4277-86b9-66a98c3eea73" width="800" height="480" /><br/>
</div>

__Schematic overflow of the DeepION model__. The input original ion image is first augmented to two views using the augmentation operation "T" , where “COL” mode used the augmentation combination of color jitter, filtering, Poisson noise, random missing, ‘ISO’ mode conduct the intensity-dependent missing additively. Then, encoder module utilizes ResNet18 to capture the relevant spatial patterns from two augmented views to generate the vector representation. The projection module and prediction module ensure the encoder module achieving stable and meaningful ion image representation, in which multi-layer perceptron is applied. These three modules are trained to maximize similarity using the contrastive loss. After training is completed, the dimensionality reduction module is implemented by the UMAP algorithm to achieve the final representation O for downstream tasks. 

# Requirement

    python == 3.5, 3.6 or 3.7
    
    pytorch == 1.8.2
    
    opencv == 4.5.3
    
    matplotlib == 2.2.2

    numpy >= 1.8.0
    
    umap == 0.5.1

# Quickly start

## Input
