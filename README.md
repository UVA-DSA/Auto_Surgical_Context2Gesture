# Towards Surgical Context Inference and Translation to Gestures

This repo for the paper titled 'Towards Surgical Context Inference and Translation to Gestures'.

<img src="https://github.com/UVA-DSA/Auto_Surgical_Context2Gesture/blob/main/Example_Clips/projectSuturing_S03_T04_slow.gif" width="500px">
Suturing

<img src="https://github.com/UVA-DSA/Auto_Surgical_Context2Gesture/blob/main/Example_Clips/NP_S04_T01_PRED.gif" width="500px">
Needle Passing

<img src="https://github.com/UVA-DSA/Auto_Surgical_Context2Gesture/blob/main/Example_Clips/KT_S02_T02_PRED.gif" width="500px">
Knot Tying


## Getting Started
run the following commands with python 3.9 or above
* `python -m venv context-env`
* `.\context-env\Scripts\activate`
* `pip install -r requirements.txt`
* `python .\run_pipeline.py Knot_Tying ALL 2023_DL` - to run the script generates context labels based on the deeplab instrument masks without kinematics


### Naming Conventions 
TASKS can be Needle_Passing, Knot_Tying, Suturing

Masks belong to MASK SETS such as 2023_ICRA, COGITO_GT, 2023_DL, ...

Each task subject trial combination represents a unique TRIAL ```<Task>_S<Subject number>_T<Trial number>``` 

## Folder Structure
* context_inference
    * context_inference.py  -- Contains context inference logic
    * contour_extraction.py
    * contour_template.json
    * metrics.py            -- IOU between predicted context and consensus context
    * utils.py
* data
    * context_labels
        * consensus
        * `<Labeler>`
    * contours
    * masks
        * 2023_DL
            * ring
            * thread
            * leftgrasper
            * needle
            * rightgrasper
                * ```<Task>_<Subject>_<Trial>.png```
                    * frame_0001.png
    * images
        * ```<Task>_<Subject>_<Trial>.png```
            * frame_0001.png
* eval 
    * contour_images
    * labeled_images
        * ```<Task>_<Subject>_<Trial>.png```
            * frame_0001.png
    * pred_context_labels
        * 2023_DL           
            * ```<Task>_<Subject>_<Trial>.txt```
* seg
    * Image segmentation scripts
* run_pipeline.py -- runs context prediction pipeline