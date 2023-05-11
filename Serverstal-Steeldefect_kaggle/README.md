# Severstal: Steel Defect Detection - Kaggle
### [Kaggle Contest](https://www.kaggle.com/c/severstal-steel-defect-detection/overview)

Task: Localize and classify surface defects on a steel sheets (Semantic Segmentation)

## Overview
The production process of flat sheet steel is especially delicate. From heating and rolling, to drying and cutting, several machines touch flat steel by the time it’s ready to ship. Today, Severstal uses images from high frequency cameras to power a defect detection algorithm. <br>
Competition involved developing models for localizing and classifying surface defects on a steel sheet.

**Evaluation:**
The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by: <br>

<img src="https://render.githubusercontent.com/render/math?math=\LARGE{\frac{2\times |X \cap Y|}{|X| %2B |Y|}}">

where X is the predicted set of pixels and Y is the ground truth. The Dice coefficient is defined to be 1 when both X and Y are empty.

---

## Compete Stats
Accuracy: 89.5% | rank1's 90.8% <br>
Rank: 345 / 2427 <br>
Team: Magma Blues - [GokulNC](https://github.com/GokulNC), [PremK](https://github.com/Prem-kumar27), [JGeoB](https://github.com/JosephGeoBenjamin)


## Repo Usage
:warning: Codes were salvaged for reference, might not be in working condition entirely

1. Set python path to the realpath of folder *Kaggle_Serverstal_SteelDefectDetection* <br>
`export PYTHONPATH=/path_to_repo/SteelDefectDetection_Serverstal_kaggle/:$PYTHONPATH`

2. Install requirements.txt using `pip`

3. Use the scripts in the tasks folder to train models and run inference