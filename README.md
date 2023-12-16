# Gazetracking - Onboarding 

## Overview

### Steps

#### Step 1: Install Conda

#### Step 2: Install dependencies

Run `conda env create -f environment.yml` at root folder.

If update existing env, use `conda env update --file environment.yml --prune`.

If you want to change the environment name, change the `name` attribute (1st line) in `environment.yml`.

#### Step 3: Activate conda env and test run

In VsCode terminal, run `conda env list` to show the list of conda environment you have. It should show the * at the top, base environment.

Then run `conda activate BCI-Gazetrack` (or different name if you changed it). Run `conda env list` again to confirm, the * should be at the same line as BCI-Gazetrack.

![Step 3 example img](example/step3.png "Step 3")

Open the `step3.py` file to see if there are warnings about "cv2 can't resolve". You can open VsCode Command Pallete (Ctrl + Shift + P on Windows) and choose **Python: Select Interpreter**, then choose BCI-Gazetrack one.

Run `python step3.py`. If your computer has a camera connected (hopefully), it will show you *a breathtaking sight*!

**Press Q to exit**. You can read a bit in the code to understand the minimal code to get OpenCV running.

#### Step 4: Mediapipe library

[MediaPipe](https://github.com/google/mediapipe) is an open-source framework from Google for building pipelines to perform computer vision inference over arbitrary sensory data such as video or audio.

We will use this library to map the facial landmarks of our face, therefore get the location of important features such as the eyes and irises.

Run `python step4.py`, and it should also open your camera with extra stuff draw on your face. We mostly only care about the FACEMESH_TESSELATION (which is the web-like, grey thin lines draw on your face).

![Step 4 example img](example/step4.png "Step 4")

 Each or the vertices for this mesh called a landmark and has an id, which you can reference with `facemesh_landmark_ref` in `\res` folder. Zoom in real close and you will see the number.