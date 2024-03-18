# 3DCV_Project

This project basically is the implementation of Structure from motion pipeline. It takes set of images and then generates the pose of frames and the 3d point cloud.

# Steps carried out in the Pipeline:
## 1. Estimation of 2d-2d correspondences
### Stage 1:
For stage 1 correspondence were provided hence we directly read the correspondecne from the file

### Stage 2:
- In stage 2 we first carry out feature detection using SIFT or ORB.
- Then we generate 2d-2d correspondence by matching the points using descriptors
- We carry out various filtering Techniques
    - Ratio Test 1
    - Ratio Test 2
    - Forward Backward Consistency check
    - Ransac Filtering

Hence frome step 1 we generate 2d- 2d correspondences

## 2. Estimation of Essential Matrix
Next step is to estimate essential matrix using the generated 2d-2d correspondences

## 3. Decomposition of Essentail matrix to R and T
Once essential matrix is estimated we decompose it into R and T of the second frame with respect to the first frame.

## 4: Triangulation
As the Rotation and tranlationof both matrix are computed we can then get the projection matrix for both frames. Using the 2d-2d correspondecnes and the Projection Matrix we then traingulate the points to get 3d coordinates

## 5: Linear Pnp(Pose from 3D-2D correspondence)
- Now for next frame we compute 2d-2d correspondence first
- For a portion of estimated matches we have corresponding 3d points already traingualted
- We registed the image points of new frame to those 3d coordinates in space
- Once we have 3D-2D correspondence we then estimate the pose of new frame using Linear PnP.

## 6: Bundle Adjustment
As there is error in estimate of correspondence , estimate of pose and as well as the traingulated point the error propagation occurs as we are using the same triangulated points to estimate pose of new frame. So we use bundle adjustment to reduce reconstruction error
- First we compute jacobian matrix
- Jacobian matrix consist of derivative of error function with respect to R,C and X
- We then use Gauss Newton non linear optimization to solve this. However the solution is not straing forward as the hessian matrix is very large but also sparse as same time.
- So we use Forward backward substitution for estiamtion of camera parameter update and points update separately.


# Commands
To run the pipeline run the command below in the project directory
```
python -m src.pipeline -d "boot" -s 1 -t 0
```

## Arguments
- -h, --help            :show this help message and exit
- -d, --dataset DATASET:
                        Name of dataset to generate the parameters. For stage 1 possible values are 'box' and 'boot'
-  -s, --stage STAGE:
                        Stage of Project. The resources precomputed assumption is based on the stage of project. For example if stage is 1 we dont carry out feature detection and matching
-  -t, --gt GT :         Whether gt is available or not. This is used for specifying the camera parameter file.
