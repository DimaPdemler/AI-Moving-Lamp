# AI-Moving-Lamp

## Introduction
Github of the AI Moving lamp project that we completed for our physics 124 class

A video summary of the project can be seen in this [youtube video](https://youtu.be/8-hrm-s7x8I).


## Folder Structure

- The folder **Final_report_CAD** contains all the CAD files for the project that we used to 3D print.
- The folder **Training_and_Preprocessing** contains two files, one that preprocesses folders of image data (that contains notebooks and dont) into a dataset folder that can be used by the training file. The second file is this training jupyter notebook that takes the preprocessed folder and trains the network and then after completion saves it as a tensorflow lite model to run on the raspberry pi.
- **Final Report.pdf** is the full report on how to we made this project happen and different technical specifications.
- **Lamp_code.py** is the file that the rasberry pi runs to control the whole moving lamp. This takes the tensorflow lite model and takes steps of movement with the dc motors and processes the image through the model to see if the current image it sees is a notebook or not.
- **wholeassembly.stl** is the whole 3D CAD model of the lamp that you can view on here.

