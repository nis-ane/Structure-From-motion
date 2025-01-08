
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href=https://github.com/nis-ane/Structure-From-Motion">
    <img src="figures/banner.png" alt="Logo" width="800" height="120">
  </a>

  <h3 align="center">Structure from Motion</h3>

  <p align="center">
    <a href="https://docs.google.com/presentation/d/16j4EOtrAW0P2WsVtkxl3FyqDBXDBwdE-QyWcZeNN2XI/edit">Presentation</a>
    .
    <a href="/3dcv_formulas_term_project.pdf">Report</a>
  </p>
</div>


<!--
This project is done as a part of Course:3D computer Vision. 
-->
This project implements of Structure from motion pipeline from scratch. It takes set of images and then generates the pose of frames and the 3D Point Cloud.

---

<!-- TABLE OF CONTENTS -->
## **Table of Contents**
1. [Project Structure](#project-structure)
2. [Pipeline](#pipeline)
3. [Installation](#installation)
4. [Running the Code](#running-the-code)
5. [Output](#output)
6. [Team Members](#team-members)

---

## Project Structure
```
├── data
|   ├── stage1
│   └── stage2
├── figures
├── notebook
├── src
|   ├── bundle_adjustment.py
|   ├── correspondence.py
|   ├── essential_mat.py
|   ├── frame.py
|   ├── jacobian.py
|   ├── map.py
|   ├── match.py
|   ├── optimize.py
|   ├── pipeline.py
|   ├── pose_estimation.py
|   ├── sparse_ba.py
|   ├── triangulation.py
|   ├── utils.py
|   └── visualize.py
├── test
|   ├── test_essential_mat.py
|   ├── test_index.py
|   ├── test_linear_pnp.py
|   ├── test_matches.py
|   └── test_triangulation.py
├── 3dcv_formulas_term_project.pdf
├── README.md
├── __init__.py
├── group_id.txt
└── requirements.txt
```

## Pipeline
### 1. Estimation of 2D-2D correspondences
#### Stage 1:
For Stage 1, correspondences were provided hence we directly read the correspondecne from the file.

#### Stage 2:
- In Stage 2, we first carry out feature detection using SIFT or ORB.
- Then we generate 2D-2D correspondences by matching the points using descriptors.
- We carry out various filtering Techniques
    - Ratio Test 1
    - Ratio Test 2
    - Forward Backward Consistency check
    - Ransac Filtering

Hence from Step 1 we generate 2D-2D correspondences.

---

### 2. Estimation of Essential Matrix
Next step is to estimate essential matrix using the generated 2D-2D correspondences.

### 3. Decomposition of Essentail matrix to R and T
Once essential matrix is estimated we decompose it into R and T of the second frame with respect to the first frame.

### 4: Triangulation
As the Rotation and tranlationof both matrix are computed we can then get the projection matrix for both frames. Using the 2D-2D correspondecnes and the Projection Matrix we then traingulate the points to get 3D coordinates.

### 5: Linear PnP(Pose from 3D-2D correspondence)
- Now for next frame we compute 2D-2D correspondence first
- For a portion of estimated matches we have corresponding 3D points already traingualted
- We registed the image points of new frame to those 3D coordinates in space
- Once we have 3D-2D correspondence we then estimate the pose of new frame using Linear PnP.

### 6: Bundle Adjustment
As there is error in estimate of correspondence , estimate of pose and as well as the traingulated point the error propagation occurs as we are using the same triangulated points to estimate pose of new frame. So we use bundle adjustment to reduce reconstruction error
- First we compute jacobian matrix
- Jacobian matrix consist of derivative of error function with respect to R,C and X
- We then use Gauss Newton non linear optimization to solve this. However the solution is not straing forward as the hessian matrix is very large but also sparse as same time.
- So we use Forward backward substitution for estiamtion of camera parameter update and points update separately.

---

## Installation

Clone the repository:
```bash
git clone https://github.com/nis-ane/Structure-From-motion.git
```
```bash
cd Structure-From-motion
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Code:
To run the pipeline, run the command below in the project directory:
```bash
python -m src.pipeline -d "boot" -s 1 -t 0
```

### Command-Line Arguments
- `-h, --help`            : show this help message and exit
- `-d, --dataset DATASET`:
                        name of dataset to generate the parameters. For Stage 1, possible values are `box` and `boot`
-  `-s, --stage STAGE`:
                        stage of Project. The resources precomputed assumption is based on the stage of project. For example if Stage is 1, we dont carry out feature detection and matching
-  `-t, --gt GT` :         whether `gt` is available or not. This is used for specifying the camera parameter file

---

## Output

The generated outputs can be found under the respective dataset folder inside the [`data`](data/) directory.

---

## Team Members
1. [Nischal Maharjan](https://www.nischalmaharjan.info.np/)
2. [Hevra Petekkaya](linkedin.com/in/hevra-petekkaya-b451881bb)
3. [Hewan Shrestha](https://hewanshrestha.github.io/)
