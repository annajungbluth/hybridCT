[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17785725.svg)](https://doi.org/10.5281/zenodo.17785725)

# HybridCT
HybridCT is a computationally efficient method for practical cloud tomography that combines deep learning and linear algebra to reconstruct cloud extinction volumes from multi-angle imaging observations.
It combines a retrieval emulator with a reconstruction stage: (1) the retrieval is emulated using a deep learning U-Net architecture which is trained to map multi-angle radiance measurements into optical thickness values. (2) the reconstruction stage takes these multi-angle optical thickness values as input and uses SART (Simultaneous Algebraic Reconstruction Technique) to computationally reconstruct the cloud extinction volume.

This repository contains code and resources to demonstrate the method using synthetic cloud fields from Large Eddy Simulations and synthetic multi-angle radiance fields which were rendered using the 3D Monte Carlo radiative tranfer model MYSTIC and the libRadtran library. The synthetic multi-angle images represent the viewing angle geometry of the Multi-angle Imaging Spectroradiometer (MISR) onboard the Terra spacecraft at its red channel (670 nm).

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .
```
## Usage
The repository contains two main directories:
- `emulator` contains the code to train the DL cloud optical thickness retrieval emulator and predict multi-angle optical thickness from radiance input.
- `reconstruction` contains the code to perform a tomographic reconstruction using a custom SART solver which takes multi-angle optical thickness maps as input and returns the 3D cloud extinction volume.

### Download Data
Download cloud data and model weights from [Zenodo](https://zenodo.org/uploads/15171312) and save to `hybridCT/data` as indicated by `TODO` files.
```
.
├── data
│   ├── ground_truth
│   ├── predicted_data
│   └── training_data
```

### Train Model
In `hybridCT/emulator` run
```bash
python3 fit_model.py --yaml config/CONFIG
```
This will generate new model parameters in `hybridCT/emulator/artifacts`.

### Predict Optical Thickness
Generate predicted multi-angle optical thickness maps by running
```bash
python3 evalute_and_merge.py
```
in `hybridCT/emulator`. This will generate new datasets in `hybridCT/data/predicted_data`.


### Run Tomographic Reconstruction
In `hybridCT/reconstruction` execute
```bash
python3 analyze_mystic_scene.py [--mode predicted]
python3 analyze_mystic_scene.py --mode truth
python3 analyze_mystic_scene.py --mode isolated
```
to reproduce the results in the paper using the predicted optical thickness (default), the ground-truth optical thickness for the cloud field, and the isolated cloud.
The necessary datasets for predicted and ground-truth cloud properties are included in Zenodo.

### Validation
Both directories `emulator` and `reconstruction` contain a `validation` subdirectory.
For the `emulator` the ML predicted optical thickness is compared to the ground truth as well as the 1D LUT-based results. The LUT is stored in `hybridCT/emulator/validation/data/lut_disort.npz`.
For the `reconstruction` the results are compared with AT3D. To execute scripts calling AT3D, please download and install [version 3.0](https://github.com/CloudTomography/AT3D). This step is optional since the AT3D results are inlcuded in `hybridCT/reconstruction/validation/results`.
