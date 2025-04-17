# Seq2Seq-IR
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 [![GitHub contributors](https://img.shields.io/github/contributors/Lin-Group-at-UMass/Seq2Seq-IR.svg)](https://github.com/Lin-Group-at-UMass/Seq2Seq-IR/graphs/contributors/)
 [![SLSA 3](https://slsa.dev/images/gh-badge-level3.svg)](https://slsa.dev)

**Revolutionizing Spectroscopic Analysis Using Sequence-to-Sequence Models I: From Infrared Spectra to Molecular Structures**\
_Ethan J. French, Xianqi Deng, Siqi Chen, Cheng-Wei Ju, Xi Cheng, Lijun Zhang, Xiao Liu, Hui Guan, and Zhou Lin_

---

## Abstract

Infrared (IR) spectroscopy reveals molecular and material features via their characteristic vibrational frequencies in an efficient and sensitive style and has thus become one of the most popular analytical tools in broad areas involving chemical discovery. These fields include material synthesis, drug design, pharmacokinetics, safety screening, pollutant sensing, and observational astronomy. However, in situ molecular or material identification from spectral signals remains a resource-intensive challenge and requires professional training due to its complexity in tracking effects to causes. Motivated by the recent success of sequence-to-sequence (Seq2Seq) models from deep learning, we developed a direct, accurate, effortless and physics-informed protocol to realize such a in-situ spectrum-to-structure translation, and provided the proof-of-concept of our models using IR spectra. We expressed both the input IR spectrum and the output molecular structure as alphanumerical sequences, treated them as two sentences describing the same molecule from two different languages, and translated them into each other using Seq2Seq models from recurrent neural networks (RNNs) and Transformers. Trained and validated using a curated data set of 198,091 organic molecules from the QM9 and PC9 databases, our Seq2Seq models achieved state-of-the-art accuracy of up to 0.611, 0.850, 0.804, and > 0.972 in generating target molecular identities, chemical formulas, structural frameworks, and functional groups from only IR spectra. Our study sets the stage for a revolutionary way to analyze molecular or material spectra by replacing human labor with rapid and accurate deep learning approaches.

---

## Installation
```
conda create -n ir python=3.12
cd Seq2Seq-IR/Spectrum2Structure/
pip install -r requirements.txt
```

---

## Usage


---

## Data
The data is available on Figshare at: [10.6084/m9.figshare.28754678](https://doi.org/10.6084/m9.figshare.28754678) 

or can be downloaded by running:

```bash
XXXXXXXXXX
```
The data is also temporarily available on Google Drive at https://drive.google.com/drive/folders/1UMvwrLYZU5D3FcrdzxSXH_fPjSDE0uyb
---
## Model Availability

The pre-trained models can be obtained via PyTorch Hub
```python
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
```
or via FigShare at: [10.6084/m9.figshare.28795676](https://doi.org/10.6084/m9.figshare.28795676)
## Questions, Comments, or Concerns?

For code related issues, please open a issue. For all other discussion, please open in the discussion tab.

---

## Citation

---

## Acknowledgements

H.G. and Z.L. acknowledge the financial support from from the University of Massachusetts
Amherst under their Start-Up Funds and the ADVANCE Collaborative Grant. All au-
thors acknowledge UMass/URI Unity Cluster and MIT SuperCloud for providing high-
performance computing (HPC) resources.
