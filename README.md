# EECS 442: Computer Vision - Video Stabilization
Final Project for EECS 442: Computer Vision Fall 2017. Project by Albert Lo, Eric Yang, Ethan Hong, Sam Kim

## Getting Started
Dependencies:
...
python 2.7 <br/>
opencv 2.4.11 <br/>
sk-video <br/>
numpy <br/>
matlab 2016b <br/>
cvx for matlab <br/>
<br/>
(Optional: for solve_path.py) 
Install PuLP : https://scaron.info/blog/linear-programming-in-python-with-pulp.html <br/>
...
## Usage
	Run `python driver.py` within `project/`

## Project Schema
    1. Estimating the original path
    2. Estimating a new smooth camera path
    3. Synthesize the stable video using the estimated smooth path

