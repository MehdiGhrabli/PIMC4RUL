# PIMC4RUL 


<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#About-the-project">About The Project</a>
    </li>
    <li>
      <a href="#Requirements">Requirements</a>
    <li><a href="#User guide">User guide</a></li>
    <ul>
        <li><a href="#Examples from the paper">Examples from the paper</a></li>
      </ul>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository hosts the code as well as the datasets used for the paper "Physics-informed Markov chains for remaining useful life prediction of wire bonds in power electronic modules". 
Below is a graphical abstract providing a brief description. The link for the paper will soon be made available, pending the acceptence of the paper in the Microelectronics Reliability journal.

![Portrait University Research Poster in Blue Pink Playful and Illustrative Style](https://github.com/user-attachments/assets/0c5596ff-81d3-4654-b04c-6e78dac4a9d8)



<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Requirements

Under construction

## User guide

The pipeline predicts iteratively the evolution of the health indicator $V_{ce}$. The prediction loop consists of 4 parts : 
* Inferring damage $l_c$ from indicator $V_{ce}$ using Gaussian Process Regression, coded in file GP_code.py  
* Estimate mechanical properties from temperature swing $\Delta T$ and current crack $l_c$ in Generate_Features.py
* Estimate the damage evolution using Kernel Density Estimation, visualized in KDE_vis.ipynb and implemented in full pipeline code main.py
* Evaluate the indicator value using the Gaussian Process Regression's mean.
In order to utilize this pipeline, the user can download this project and execute a command in a command prompt in the same directory as the downloaded project.

Example : 
```
python run_pipeline.py --bwp 0.01 --confidence 90 --number_of_montecarlo_iterations 100
```
In this example, bwp, confidence and number_of_montecarlo_iterations are arguments to be adjusted by the user, depending on the use case. 
The list of arguments is the following : 
* simulated-data-path : The path of the simulated data, set to data\Simulated_data.csv by default
* initial-contact-length : The initial contact length (in mm) between the wire and the metallization, 0.3615 by default
* eps : An arbitrarly small value to avoid numerical errors, taken to be equal to $10^{-8}\times 0.3615$.
* t-min : The minimum temperature of the cycle, equal to $55°C$ in this work
* svr-kernel : The kernel used in the support vector regression, rbf by default
* svr-c :
* svr-gamma :
* destructive-data-path : The path of the destructive data, linking $V_{ce}$ to $l_c$, data\Cross section analysis.xlsx in this repository
* train-data-path : A file containing the paths of runs to failure to be used in training, train_data_files_standard_70.txt for example
* test-data-path : Path of the test run, data\Runs_to_failure_70\Rth Module 33L.xlsx for example
* history-size : Size of historic data, equal to 2 by default
* prediction-size : Size of predictions, equal to 1 by default
* data-size : Number of synthetic runs to failure to be used
* path-of-generated-data : 
* used-features-path :
* bwp :
* gp-std : Gaussian process regression's standard deviation parameter
* kde-kernel : Kernel density estimation kernel, taken as gaussian
* start-cycle-index : 
* confidence : Level of the confidence interval to be adjusted according to the user's preference, 0.9 by default
* number-of-montecarlo-iterations : Number of Monte Carlo iterations, set to 100 
* regular-data-path :

### Examples from the paper

```
python run.py Maze-Simple-v0 maze_baseline --horizon 75 --seed 0 --epochs 1500
```

<!-- ROADMAP -->
## Roadmap

- [x] Submit datasets
- [x] Submit commented code
- [ ] Link the research paper
- [x] Provide a user guide 
- [ ] Incorporate data generation as an indenpendant function to be used in run.py
- [ ] Incorporate SVR functions as an indenpendant function to be used in run.py

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact
For any inquires or suggestions : 

Mehdi Ghrabli  - mehdi.ghrabli1@ens-paris-saclay.fr

Project Link: [https://github.com/MehdiGhrabli/PIMC4RUL](https://github.com/MehdiGhrabli/PIMC4RUL)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citation
Pending the acceptence of the paper in the Microelectronics Reliability journal.
