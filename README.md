# EfficientCellSeg
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19abichcVaeAlbbojcaLUh5c_rCu8pYIL?usp=sharing)

Efficient encoder-decoder model for cell segmentation in 3D microscopy images. 
3D microscopy images are analyzed slice-wise as stacks of 2D slices.
Context from adjacent 2D slices is encoded using our pseudocoloring algorithm (see below). 
  
## Results in the [Cell Segmentation Benchmark](http://celltrackingchallenge.net/latest-csb-results):
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Team / Method</th>
      <th>Model(s)</th>
      <th>#params &dArr;</th>
      <th>SEG Score &uArr;</th>
      <th>Ranking*</th>
    </tr>
  </thead>
  <tr>
    <td>Fluo-C3DH-A549-SIM</td>
    <td>DKFZ-GE [<a href=https://github.com/MIC-DKFZ/nnUNet>Repo<a/>]</td> 
    <td>4x 3D nnU-Net</td>
    <td>176.0M</td>
    <td>0.955</td>
    <td>1/12</td>
  </tr>
  <tr>
    <td></td>
    <td>Ours</td> 
    <td>EfficientCellSeg</td>
    <td>6.7M</td>
    <td>0.951</td>
    <td>2/12</td>
  </tr>
  <tr>
    <td>Fluo-N3DL-TRIC</td>
    <td>KIT-Sch-GE [<a href=https://git.scc.kit.edu/KIT-Sch-GE/2021_segmentation>Repo<a/>]</td> 
    <td>Dual U-Net</td>
    <td>46.4M</td>
    <td>0.821</td>
    <td>1/8</td>
  </tr>
  <tr>
    <td></td>
    <td>Ours</td> 
    <td>EfficientCellSeg</td>
    <td>6.7M</td>
    <td>0.782</td>
    <td>3/8</td>
  </tr>
  <tr>
    <td>Fluo-C3DL-MDA231</td>
    <td>KIT-Sch-GE [<a href=https://git.scc.kit.edu/KIT-Sch-GE/2021_segmentation>Repo<a/>]</td> 
    <td>Dual U-Net</td>
    <td>46.4M</td>
    <td>0.710</td>
    <td>1/19</td>
  </tr>
  <tr>
    <td></td>
    <td>Ours</td> 
    <td>2x EfficientCellSeg</td>
    <td>13.4M</td>
    <td>0.646</td>
    <td>2/19</td>
  </tr>
</table>
*Rankings as of 16.04.2022
<br></br>

## Example results of our segmentation method for 2D slices of 3D microscopy images:
![Example results](assets/example_2Dslices.png?raw=true "Example Results")

## (Spatial-) Context Aware Pseudocoloring:
![Context Aware Pseudocoloring](assets/context_aware_pcolor.png?raw=true "Context Aware Pseudocoloring")

## Conference Paper
> [EfficientCellSeg: Efficient Volumetric Cell Segmentation Using Context Aware Pseudocoloring](https://openreview.net/forum?id=KnJsGdhx1kH),
> Wagner, Royden and Rohr, Karl,
> *MIDL 2022*; *arXiv ([arXiv:2204.03014](https://arxiv.org/abs/2204.03014))*

## Citation
```bibtex
@inproceedings{wagner2022efficientcellseg,
  title={EfficientCellSeg: Efficient Volumetric Cell Segmentation Using Context Aware Pseudocoloring},
  author={Royden Wagner and Karl Rohr},
  booktitle={Medical Imaging with Deep Learning},
  year={2022}
}
```
