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

Open the `step3.py` file to see if there are warnings about "cv2 can't resolve". You can open VsCode Command Pallete (Ctrl + Shift + P on Windows) and choose **Python: Select Interpreter**, then choose BCI-Gazetrack one.

Run `python step3.py`. If your computer has a camera connected (hopefully), it will show you *a breathtaking sight*!

You can read a bit in the code to understand the minimal code to get OpenCV running.