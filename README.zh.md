# README 中文版

## Abstract - 摘要

<p align="justify">
红外（IR）光谱借助分子特有的振动频率，以高效而灵敏的方式揭示分子与材料特征，因此已成为材料合成、药物设计、药代动力学、安全性筛查、污染物检测及天文观测等诸多领域中最受欢迎的分析工具之一。然而，仅凭光谱信号进行原位分子或材料识别仍然是一项资源密集型的难题，并因溯因推果的复杂性而需要专业培训。受深度学习序列到序列（Seq2Seq）模型近期成功的启发，我们开发了一种直接、准确、省力且融合物理先验的流程，实现了原位“光谱‑结构”翻译，并以 IR 光谱验证了概念可行性。我们将输入的 IR 光谱与输出的分子结构均表示为字母数字序列，将二者视为用两种不同语言描述同一分子的两句话，并利用循环神经网络（RNN）和 Transformer 的 Seq2Seq 模型进行互译。在来自 QM9 与 PC9 数据库、共 198 091 个有机分子的精选数据集上训练与验证后，我们的 Seq2Seq 模型在仅依赖 IR 光谱的情况下，实现了生成目标分子身份、化学式、骨架结构和官能团的最先进准确率，分别达到 0.611、0.850、0.804 以及 > 0.972。我们的研究为通过快速且精准的深度学习方法替代人工分析分子或材料光谱开辟了革命性的新途径。
</p>

## Installation - 下载
```bash
conda create -n ir python=3.12
conda activate ir
cd Spectrum2Structure/
# pip install pip==24.0
pip install -r requirements.txt
```
#### 注意：请使用```pip ≤ 24.0```，如果你的```pip > 24.0```会导致```pytorch_lightning == 1.6.3```无法下载。如果你下载 的```pytorch_lightning != 1.6.3```，请修改```train.py```文件中的```'gpus'```设置。

---
## Data - 数据
可以通过Figshare来访问我们的数据集：[10.6084/m9.figshare.28754678](https://doi.org/10.6084/m9.figshare.28754678) 

然后请将数据保存到data目录当中
```bash
# 下载并复制数据到该目录当中
cd Spectrum2Structure/data
```

---
## Model Availability - 可使用的模型

可以通过PyTorch Hub来获取我们的预训练模型
```python
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
```
或者通过Figshare来获取我们的预训练模型: [10.6084/m9.figshare.28795676](https://doi.org/10.6084/m9.figshare.28795676)

然后将模型权重保存到checkpoints目录当中
```bash
cd Spectrum2Structure/
mkdir checkpoints
cd checkpoints/
# 下载并复制数据到该目录当中
unzip smiles_checkpoints.zip
```
#### Seq2Seq Models - SELFIES Format 通过SELFIES格式数据集训练的模型
* [LSTM-SELFIES](https://drive.google.com/file/d/18h9KhxCqPs8dqfkvVkXmlpzUHZcvXVG3/view?usp=drive_link)
* [GRU-SELFIES](https://drive.google.com/file/d/1yvieeRp4zAPLxwbEXy2r_y-KKDQtpuoE/view?usp=drive_link)
* [GPT-SELFIES](https://drive.google.com/file/d/1aZ4LimRDZQdO6-nbl2L7pFX_1tf7bcb5/view?usp=drive_link)
* [Transformer-SELFIES](https://drive.google.com/file/d/1GEKui9gihHuNBLjL39D7RFfIPul5FBBf/view?usp=drive_link)

#### Seq2Seq Models - Mixture Molecule 通过混合分子格式数据集训练的模型
* [LSTM-Mixture](https://drive.google.com/file/d/1SR_eywrGnizKsq3pj90MIOsoJUXYmXyi/view?usp=drive_link)
* [GRU-Mixture](https://drive.google.com/file/d/1gnLu4cNLegIQ_VY02NHMEHdprG7vvooN/view?usp=drive_link)
* [GPT-Mixture](https://drive.google.com/file/d/1MEW_AM3cALkOGscMi5OxXYdUKQECDlny/view?usp=drive_link)
* [Transformer-Mixture](https://drive.google.com/file/d/1BvmMF_TV3AM2rN7n5v-x8-M_gF-I_PMk/view?usp=drive_link)

#### Seq2Seq Models - SMILES Format 通过SMILES格式数据集训练的模型
* [SMILES_Checkpoints](https://drive.google.com/file/d/1OHjNAwHIZGW89V9PlxsNk0SL0lJaL5hw/view?usp=drive_link)

---

## Usage - 使用方法
### Model Training - 模型训练

对于不同数据集的可用的模型：

SELFIES格式数据集: ```lstm```, ```gru```, ```gpt```, ```transformer```

混合分子格式数据集: ```lstm-mixture```, ```gru-mixture ```, ```gpt-mixture```, ```transformer-mixture```

SMILES格式数据集: ```lstm-smiles```, ```gru-smiles```, ```gpt-smiles```, ```transformer-smiles```

```bash
cd Spectrum2Structure/
chmod +x run_train.sh
./run_train.sh <model>
```

或者，你也可以使用自定义设置来训练模型。
```bash
cd Spectrum2Structure/
python train.py --model <model> --mode <mode> \
    --hidden_dim <hidden_dim> --dropout <dropout> --layers <layers> --heads <heads> \
    --batch_size <batch_size> --max_epochs <max_epochs> --lr <lr> --weight_decay <weight_decay>
    --use_gpu <use_gpu> --calculate_prediction <calculate_prediction>
```

可选参数:
```bash
--model                Model architecture: LSTM, GRU, GPT, Transformer       (default: Transformer)
--mode                 Choose encoding mode: selfies, smiles, or mixture     (default: selfies)
--hidden_dim           size of input hidden units                            (default: 768)
--dropout              dropout probability                                   (default: 0.1)
--layers               number of hidden layers                               (default: 6)
--heads                number of attention heads                             (default: 6)
--batch_size           Batch size for training                               (default: 256)
--max_epochs           number of epochs to train                             (default: 80)
--lr                   initial learning rate                                 (default: 1e-4)
--weight_decay         weight decay coefficient                              (default: 1e-5)
--use_gpu              Whether to use GPU for training                       (default: True)
--calculate_prediction whether to calculate prediction                       (default: True)
--seed                 Random seed for reproducibility                       (default: 78438379)
```

#### Example - 样例
使用SMILES格式数据集来训练Transformer模型：
```bash
cd Spectrum2Structure/
chmod +x run_train.sh
./run_train.sh transformer-smiles
```

自定义设置：
```bash
cd Spectrum2Structure/
python train.py --model Transformer --mode smiles \
    --hidden_dim 768 --dropout 0.1 --layers 6 --heads 6 \
    --batch_size 256 --max_epochs 95 --lr 1e-4 --weight_decay 1e-5 \
    --use_gpu True --calculate_prediction True
```

### Model Evaluation - 模型评估

不同数据集可用的任务

SELFIES格式数据集: ```eval```, ```generation```, ```topk```

混合分子格式数据集: ```eval```, ```generation```

SMILES格式数据集: ```eval```

```bash
cd Spectrum2Structure/
chmod +x run_test.sh
./run_test.sh <model> <checkpoint> <mode> [output_file]
```

或者，你也可以使用自定义设置来测试模型。
```bash
cd Spectrum2Structure/
python test.py --model <model> --mode <mode> --task <task> \
    --checkpoints <checkpoints> --batch_size <batch_size> \
    --use_gpu <use_gpu> --output_file <output_file>
```

可选参数:
```bash
--mode            Choose encoding mode: selfies, smiles, or mixture          (default: selfies)
--task            Task type: evaluation, generation, or top‑k prediction     (default: eval)
--model           Model architecture: LSTM, GRU, GPT, Transformer            (default: Transformer)
--checkpoints     Path to model checkpoint (required)                                   
--batch_size      Batch size for testing                                     (default: 256)
--use_gpu         Whether to use GPU for inference                           (default: True)
--seed            Random seed for reproducibility                            (default: 78438379)
--output_file     Custom output file name (optional)                         (default: None)
```

#### Example - 样例
使用Transformer模型来得到Topk结果：
```bash
cd Spectrum2Structure/
chmod +x run_test.sh
./run_test.sh transformer checkpoints/Transformer-epoch=96-step=30070.ckpt topk
```

自定义设置：
```bash
cd Spectrum2Structure/
python test.py --model Transformer --mode selfies --task topk \
    --checkpoints checkpoints/Transformer-epoch=96-step=30070.ckpt --batch_size 256 --use_gpu True
```

---

## Questions, Comments, or Concerns? - 任何问题，意见或疑虑？

For anything code related, please open an issue. Otherwise, please open a discussion tab.
如涉及代码相关的问题，请提交 Issue；其他情况请在讨论区发帖。

---

## Citation - 引用

If you find this model and code are useful in your work, please cite:
如果您在工作中发现本模型和代码有用，请引用以下文献：
```bibtex
@article{french2025revolutionizing,
  title={Revolutionizing Spectroscopic Analysis Using Sequence-to-Sequence Models I: From Infrared Spectra to Molecular Structures},
  author={French, Ethan and Deng, Xianqi and Chen, Siqi and Ju, Cheng-Wei and Cheng, Xi and Zhang, Lijun and Liu, Xiao and Guan, Hui and Lin, Zhou},
  journal={ChemRxiv preprint doi:10.26434/chemrxiv-2025-n4q84},
  year={2025}
}
```

---

## Acknowledgements - 致谢

H.G. and Z.L. acknowledge the financial support from from the University of Massachusetts
Amherst under their Start-Up Funds and the ADVANCE Collaborative Grant. All au-
thors acknowledge UMass/URI Unity Cluster and MIT SuperCloud for providing high-
performance computing (HPC) resources.

关卉和林舟感谢马萨诸塞大学阿默斯特分校提供的启动经费和 ADVANCE 协作资助所给予的财政支持。所有作者感谢 UMass/URI Unity 集群和 MIT SuperCloud 提供的高性能计算（HPC）资源支持。
