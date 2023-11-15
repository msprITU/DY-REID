# YouReID model zoo 

## Introduction

This file documents collection of models trained with you-reid framework. All numbers were obtained with 2 NVIDIA V100 GPU. The software in use were PyTorch 1.6, CUDA 10.1.

We compare some state of the art models in the following pictures, using the performance in those paper.
![image](zoo/market1501_performance.png) ![image](zoo/duke_performance.png)

"*" stands for using extra information, "^" stands for using larger backbone

## Models
<table>
    <tr>
        <th></th><th>Market1501<br>mAP/rank-1</th><th>DukeMTMC<br>mAP/rank-1</th><th>MSMT17<br>mAP/rank-1</th><th>config</th><th>log</th>
    </tr>
    <tr>
        <td>baseline</td><td>87.65/94.80</td><td>77.21/88.33</td><td>54.61/78.80</td>
        <td><a href="../example/baseline/baseline_dist_bn.yaml">config</a></td><td><a href="">log</a></td>
    </tr>
    <tr>
        <td>PCB</td><td>84.94/94.33</td><td>77.59/87.84</td><td>55.45/78.51</td>
        <td><a href="../example/pcb/pcb.yaml">config</a></td><td><a href="">log</a></td>
    </tr>
    <tr>
        <td>MGN</td><td>89.28/95.84</td><td>80.57/89.14</td><td>61.67/83.31</td>
        <td><a href="../example/mgn/mgn.yaml">config</a></td><td><a href="">log</a></td>
    </tr>
    <tr>
        <td>CACENET</td><td>90.00/95.67</td><td>81.13/89.90</td><td>61.9/83.77</td>
        <td><a href="../example/cacenet/cacenet.yaml">config</a></td><td><a href="">log</a></td>
    </tr>
</table>




#### multi sources
We contribute some reid samples to opencv community, you can use these model in [opencv](https://github.com/opencv/opencv/pull/19108), and you also can visit them at [ReID_extra_testdata](https://github.com/ReID-Team/ReID_extra_testdata).
The following table shows the performance of these model


<table>
    <tr>
        <th></th><th>Market1501<br>mAP/rank-1</th><th>DukeMTMC<br>mAP/rank-1</th><th>MSMT17<br>mAP/rank-1</th><th>config</th><th>download</th>
    </tr>
    <tr>
        <td>youtu_reid_baseline_lite</td>
        <td>87.86/95.01</td><td>79.75/89.05</td><td>58.82/80.81</td>
        <td><a href="../example/baseline/baseline_lite_multidataset.yaml">config</a></td>
        <td>
            <a href="https://drive.google.com/file/d/1l-8Lj9OPs4D6qKGAljbJgZuxGvENkDjl/view?usp=sharing">weight</a> 
            <a href="https://drive.google.com/file/d/1CinUtnkO_r9120qEuL1c8EbMuYzql2Cu/view?usp=sharing">onnx_weight</a> 
            <a href="https://drive.google.com/file/d/1QmmbU3c2Nw6UR5AGvU5B-v2kRiPKIJQ3/view?usp=sharing">log</a>
        </td>
    </tr>
    <tr>
        <td>youtu_reid_baseline_medium</td>
        <td>90.75/96.32</td><td>83.38/91.56</td><td>65.30/85.08</td>
        <td><a href="../example/baseline/baseline_medium_multidataset.yaml">config</a></td>
        <td>
            <a href="https://drive.google.com/file/d/1bhOMl4masd63alS3HMZ060miyM7R31uR/view?usp=sharing">weight</a> 
            <a href="https://drive.google.com/file/d/1CinUtnkO_r9120qEuL1c8EbMuYzql2Cu/view?usp=sharing">onnx_weight</a>
            <a href="https://drive.google.com/file/d/1QmmbU3c2Nw6UR5AGvU5B-v2kRiPKIJQ3/view?usp=sharing">log</a>
         </td>
    </tr>
    <tr>
        <td>youtu_reid_baseline_large</td>
        <td>91.85/96.73</td><td>84.40/91.88</td><td>68.68/87.04</td>
        <td><a href="../example/baseline/baseline_large_multidataset.yaml">config</a></td>
        <td>
            <a href="https://drive.google.com/file/d/1zoM1o_6o7otV0VfUxtCwj7O_AbBplWnu/view?usp=sharing">weight</a> 
            <a href="https://drive.google.com/file/d/1yU609diFdkre2j7f4rHxZWKO-vkS-04c/view?usp=sharing">onnx_weight</a>
            <a href="https://drive.google.com/file/d/1iP6sSSxofoeeb-6Q4BDjrD7ncWvT5TEP/view?usp=sharing">log</a>
         </td>
    </tr>
</table>

