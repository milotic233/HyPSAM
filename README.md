# HyPSAM: Hybrid Prompt-driven Segment Anything Model for RGB-Thermal Salient Object Detection
This are the results of "HyPSAM: Hybrid Prompt-driven Segment Anything Model for RGB-Thermal Salient Object Detection".

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
