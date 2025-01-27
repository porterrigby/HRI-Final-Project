# HRI Final Project

GitHub Repository: https://github.com/porterrigby/HRI-Final-Project

All code used for the project is contained within this GitHub repository. 
Retico runner files used for each experiment live within the runners/ directory, and
should run as-is assuming all dependencies are present on the local system. The data/
directory holds all of the timing data that was collected for experiment 1, as well as
the Jupyter Notebook where data analysis was performed. The src/ directory contains
all of the module files required for the runners to function. Below are directions for
using each runner file.

grounding_runner.py
- requires that mic_runner.py is also running
- grounding_runner.py and mic_runner.py must both use the
same ip address for ZMQ
- `python grounding_runner.py <rtdetr | yolov8 | yolov11>`

mic_runner.py
- `python mic_runner.py`

timing_runner.py
- `python timing_runner.py <rtdetr | yolov8 | yolov11> <model size to use> <confidence threshold> <number of samples to take> <output file>`
- a script is available in the root directory of the repository called `timings.sh`, 
and was used for automating data collection for experiment 1.
