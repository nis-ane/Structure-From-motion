"""
This is the main script where all integration of whole pipeline is to be done
"""

import argparse
import os

def run_pipeline():
    # Iterate over the images
        # Compute the Rotation and translation between two images
        # Compute 3d points using triangulation
        # Carry out bundle adjustment
    # Add extra image
    # Compute rotation and translation with image with great correspondence
    # Compute 3d point of new points 
    # Carry out bundle adjustment.
    pass

if __name__ == "__main__":
    print("Estimating camera extrinsic parameters")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default='box',
        help="Name of dataset to generate the parameters. For stage 1 possible values are 'box' and 'boot'",
    )
    parser.add_argument(
        "-s",
        "--stage",
        type=int,
        default=1,
        help="Stage of Project. The resources precomputed assumption is based on the stage of project",
    )
    args = parser.parse_args()

    root_folder = f"./Stage_{args.stage}/submission"
    dataset_folder = os.path.join(root_folder, args.dataset) 
    assert os.path.exists(dataset_folder), "Dataset does not exist. Check the folder of dataset is inside the folder submission"

    image_folder = os.path.join(dataset_folder, "images")
    assert os.path.exists(image_folder), "Image Folder missing inside the dataset folder"

    correspondence_folder = os.path.join(dataset_folder, "correspondences")
    run_pipeline()