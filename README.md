# HyPSAM: Hybrid Prompt-driven Segment Anything Model for RGB-Thermal Salient Object Detection (IEEE Transactions on Circuits and Systems for Video Technology (TCSVT25))
This are the results of the TCSVT2025 paper "HyPSAM: Hybrid Prompt-driven Segment Anything Model for RGB-Thermal Salient Object Detection". [URL](https://ieeexplore.ieee.org/document/11177578)

https://github.com/user-attachments/assets/ef79dc19-9ca1-4f57-96a1-2575e95c1e9d

RGB-thermal salient object detection (RGB-T SOD)
aims to identify prominent objects by integrating complementary
information from RGB and thermal modalities. However,
learning the precise boundaries and complete objects remains
challenging due to the intrinsic insufficient feature fusion
and the extrinsic limitations of data scarcity. In this paper,
we propose a novel hybrid prompt-driven segment anything
model (HyPSAM), which leverages the zero-shot generalization
capabilities of the segment anything model (SAM) for RGB-T
SOD. Specifically, we first propose a dynamic fusion network
(DFNet) that generates high-quality initial saliency maps as
visual prompts. DFNet employs dynamic convolution and
multi-branch decoding to facilitate adaptive cross-modality
interaction, overcoming the limitations of fixed-parameter kernels
and enhancing multi-modal feature representation. Moreover, we
propose a plug-and-play refinement network (P2RNet) which
serves as a general optimization strategy to guide SAM in
refining saliency maps by using hybrid prompts. The text
prompt ensures reliable modality input, while the mask and
box prompts enable precise salient object localization. Extensive
experiments on three public datasets demonstrate that our
method achieves state-of-the-art performance. Notably, HyPSAM
has remarkable versatility, seamlessly integrating with different
RGB-T SOD methods to achieve significant performance gains,
thereby highlighting the potential of prompt engineering in this
field

<img width="1061" height="526" alt="图片" src="https://github.com/user-attachments/assets/350d8cdd-1851-47d6-80de-9fee6a330a55" />

## Comparison with the state-of-the-art methods
<img width="1116" height="507" alt="图片" src="https://github.com/user-attachments/assets/904e1844-f9ba-495f-b86c-2ee605130d55" />

## DFNet results
[DFNet](https://github.com/milotic233/HyPSAM/blob/main/dfnet.zip)

## HyPSAM results
[HyPSAM](https://github.com/milotic233/HyPSAM/blob/main/hypsam.zip)

## Citation
Please cite this paper in your publications if it helps your research:

```
@ARTICLE{hypsam,
  author={Hou, Ruichao and Li, Xingyuan and Ren, Tongwei and Zhou, Dongming and Wu, Gangshan and Cao, Jinde},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={HyPSAM: Hybrid Prompt-driven Segment Anything Model for RGB-Thermal Salient Object Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Decoding;Object detection;Image segmentation;Convolution;Kernel;Feature extraction;Training;Computer architecture;Semantics;Adaptation models;RGB-thermal;salient object detection;dynamic convolution;hybrid prompts;segment anything model},
  doi={10.1109/TCSVT.2025.3613770}}
```
