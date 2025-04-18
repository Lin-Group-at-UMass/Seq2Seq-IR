# Seq2Seq-IR
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 [![GitHub contributors](https://img.shields.io/github/contributors/Lin-Group-at-UMass/Seq2Seq-IR.svg)](https://github.com/Lin-Group-at-UMass/Seq2Seq-IR/graphs/contributors/)
 [![SLSA 3](https://slsa.dev/images/gh-badge-level3.svg)](https://slsa.dev)

**[Revolutionizing Spectroscopic Analysis Using Sequence-to-Sequence Models I: From Infrared Spectra to Molecular Structures](https://chemrxiv.org/engage/chemrxiv/article-details/67f601ef81d2151a029f5a2e)**\
_Ethan J. French, Xianqi Deng, Siqi Chen, Cheng-Wei Ju, Xi Cheng, Lijun Zhang, Xiao Liu, Hui Guan, and Zhou Lin_

<p align="center">
<img src="./figures/Figure_1_encoder-decoder_neurips_v10.jpg">
</p>

---

## Abstract

Infrared (IR) spectroscopy reveals molecular and material features via their characteristic vibrational frequencies in an efficient and sensitive style and has thus become one of the most popular analytical tools in broad areas involving chemical discovery. These fields include material synthesis, drug design, pharmacokinetics, safety screening, pollutant sensing, and observational astronomy. However, in situ molecular or material identification from spectral signals remains a resource-intensive challenge and requires professional training due to its complexity in tracking effects to causes. Motivated by the recent success of sequence-to-sequence (Seq2Seq) models from deep learning, we developed a direct, accurate, effortless and physics-informed protocol to realize such a in-situ spectrum-to-structure translation, and provided the proof-of-concept of our models using IR spectra. We expressed both the input IR spectrum and the output molecular structure as alphanumerical sequences, treated them as two sentences describing the same molecule from two different languages, and translated them into each other using Seq2Seq models from recurrent neural networks (RNNs) and Transformers. Trained and validated using a curated data set of 198,091 organic molecules from the QM9 and PC9 databases, our Seq2Seq models achieved state-of-the-art accuracy of up to 0.611, 0.850, 0.804, and > 0.972 in generating target molecular identities, chemical formulas, structural frameworks, and functional groups from only IR spectra. Our study sets the stage for a revolutionary way to analyze molecular or material spectra by replacing human labor with rapid and accurate deep learning approaches.

---

## Installation
```
conda create -n ir python=3.12
conda activate ir
cd Spectrum2Structure/
pip install -r requirements.txt
```

---

## Data
The data is available on Figshare at: [10.6084/m9.figshare.28754678](https://doi.org/10.6084/m9.figshare.28754678) 

or can be downloaded the [data](https://drive.google.com/drive/folders/1cnhPv3j5suJ9ZkO9w6gxyMYXCxSbyR1k) from Google Drive

and save data to this dir
```bash
cd Spectrum2Structure/data
# download or copy data to this default directory
```

### The [code and data](https://drive.google.com/drive/folders/1UMvwrLYZU5D3FcrdzxSXH_fPjSDE0uyb) are also temporarily available on Google Drive.

---
## Model Availability

The pre-trained models can be obtained via PyTorch Hub
```python
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
```
or via FigShare at: [10.6084/m9.figshare.28795676](https://doi.org/10.6084/m9.figshare.28795676)

or you can download the weights from
* [Google Drive](https://drive.google.com/drive/folders/1Wqoa6ORUxERydX8EVyIyzWl03dBqyarf)

and save model weights to this dir
```bash
cd Spectrum2Structure/
mkdir checkpoints
cd checkpoints/
# download or copy model weight to this default directory
unzip smiles_checkpoints.zip
```
#### Seq2Seq Models
* [LSTM](https://drive.google.com/file/d/18h9KhxCqPs8dqfkvVkXmlpzUHZcvXVG3/view?usp=drive_link)
* [GRU](https://drive.google.com/file/d/1yvieeRp4zAPLxwbEXy2r_y-KKDQtpuoE/view?usp=drive_link)
* [GPT](https://drive.google.com/file/d/1aZ4LimRDZQdO6-nbl2L7pFX_1tf7bcb5/view?usp=drive_link)
* [Transformer](https://drive.google.com/file/d/1GEKui9gihHuNBLjL39D7RFfIPul5FBBf/view?usp=drive_link)

#### Seq2Seq Models - Mixture
* [LSTM-Mixture](https://drive.google.com/file/d/1SR_eywrGnizKsq3pj90MIOsoJUXYmXyi/view?usp=drive_link)
* [GRU-Mixture](https://drive.google.com/file/d/1gnLu4cNLegIQ_VY02NHMEHdprG7vvooN/view?usp=drive_link)
* [GPT-Mixture](https://drive.google.com/file/d/1MEW_AM3cALkOGscMi5OxXYdUKQECDlny/view?usp=drive_link)
* [Transformer-Mixture](https://drive.google.com/file/d/1BvmMF_TV3AM2rN7n5v-x8-M_gF-I_PMk/view?usp=drive_link)

#### Seq2Seq Models - SMILES
* [smiles_checkpoints](https://drive.google.com/file/d/1OHjNAwHIZGW89V9PlxsNk0SL0lJaL5hw/view?usp=drive_link)

---

## Usage
### Model Training

Available models for different datasets

Single  Molecule Dataset: ```lstm```, ```gru```, ```gpt```, ```transformer```

Mixture Molecule Dataset: ```lstm-mixture```, ```gru-mixture ```, ```gpt-mixture```, ```transformer-mixture```

SMILES  Format   Dataset: ```lstm-smiles```, ```gru-smiles```, ```gpt-smiles```, ```transformer-smiles```

```bash
cd Spectrum2Structure/
chmod +x run_train.sh
./run_train.sh <model>
```

### Model Testing
```bash
cd Spectrum2Structure/
chmod +x run_test.sh
```

Get Models Results
```bash
./run_test.sh <model> <checkpoint> eval [output_file]
```

Generate Sequences
```bash
./run_test.sh <model> <checkpoint> generation [output_file]
```

Get Models Top-k Results
```bash
./run_test.sh <model> <checkpoint> topk [output_file]
```


### Model Evaluation

---

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
