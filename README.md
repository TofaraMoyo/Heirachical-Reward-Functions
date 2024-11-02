 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporally-layered-architecture-for-efficient/openai-gym-on-pendulum-v1)](https://paperswithcode.com/sota/openai-gym-on-pendulum-v1?p=temporally-layered-architecture-for-efficient) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporally-layered-architecture-for-efficient/openai-gym-on-mountaincarcontinuous-v0)](https://paperswithcode.com/sota/openai-gym-on-mountaincarcontinuous-v0?p=temporally-layered-architecture-for-efficient)


# Temporally Layered Architecture

This repository is the official implementation of [Optimizing Attention and Cognitive  Control Costs Using Temporally-Layered Architectures](https://doi.org/10.1162/neco_a_01718). 
<p align="center">
    <img src="Images/TLA Architecture.jpg" alt="Temporally Layered Architecture" width="500"/>
</p>
<!-- 📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->




## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python main.py --env_name <environment> --seed <seed>
```
## Evaluation

To evaluate TLA on {evironment} over seeds [0-9], run:


[//]: # ()
```eval

python eval.py --env_name <environment>

```
The script will evaluate TLA model and output all the metrics reported in the paper. Example:

```
--------------Pendulum-v1---------------
Mean Rewards: -154.91541574934072
STD Rewards: 31.966949159992797
mean Repetitions: 0.70325
Mean actions: 200.0
Mean decisions: 62.31
Mean jerk: 0.622791588306427
Repetitions%: 70.325%
Mean slow actions: 34.0
Mean fast actions: 34.06
```
## Pre-trained Models

You can download pretrained models on HuggingFace:

- [Pendulum-v1](https://huggingface.co/devdharpatel/tla-Pendulum-v1)


## Results

Results from the paper. Models compared:

TD3: [Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html)  
TempoRL: [TempoRL: Learning When to Act](https://proceedings.mlr.press/v139/biedenkapp21a.html)  
TLA: Ours. Temporally Layered Architecture.

### [Continuous Control (Papers with code link)](https://paperswithcode.com/paper/temporally-layered-architecture-for-efficient)

| Environment     | Normalized AUC (TD3) | Normalized AUC (TempoRL) | Normalized AUC (TLA) | Average Reward (TD3) | Average Reward (TempoRL)      | Average Reward (TLA)         |
|-----------------|----------------------|--------------------------|----------------------|----------------------|-------------------------------|------------------------------|
| **Pendulum**        | 0.85                 | 0.85                     | 0.87                 | -147.38 (29.68)      | -149.38 (44.64)               | -154.92 (31.97)              |
| **MountainCar**     | 0.19                 | 0.64                     | 0.82                 | 0 (0)                | 84.56 (28.27)                 | 93.88 (0.75)                 |
| **Inv-Pendulum**    | 0.97                 | 0.77                     | 0.96                 | 1000 (0)             | 984.21 (47.37)                | 1000 (0)                     |
| **Inv-DPendulum**   | 0.96                 | 0.94                     | 0.92                 | 9359.82 (0.07)       | 9352.61 (2.20)                | 9356.67 (1.23)               |
| **Hopper**          | 0.66                 | 0.43                     | 0.75                 | 3439.12 (120.98)     | 2607.86 (342.23)              | 3458.22 (117.92)             |
| **Walker2d**        | 0.56                 | 0.52                     | 0.53                 | 4223.47 (543.6)      | 4581.69 (561.95)              | 3878.41 (493.97)             |
| **Ant**             | 0.60                 | 0.33                     | 0.52                 | 5131.90 (687.00)     | 3507.85 (579.95)              | 5163.54 (573.19)             |
| **HalfCheetah**     | 0.79                 | 0.50                     | 0.58                 | 10352.58 (947.69)    | 6627.74 (2500.78)             | 9571.99 (1816.02)            |


| Environment     | Action Repetition (TD3) | Action Repetition (TempoRL) | Action Repetition (TLA) | Average Jerk/time step (TD3) | Average Jerk/time step (TempoRL) | Average Jerk/time step (TLA) |
|-----------------|-------------------------|-----------------------------|-------------------------|-------------------------------|---------------------------------|-----------------------------|
| **Pendulum**        | 7.44%                   | 34.94%                      | 70.32%                  | 1.02                          | 0.94                            | 0.62                        |
| **MountainCar**     | 9.08%                   | 75.99%                      | 91.4%                   | 0.1                           | 1.12                            | 1.11                        | 
| **Inv-Pendulum**    | 1.12%                   | 45.97%                      | 88.82%                  | 1.11                          | 1.62                            | 0.12                        |
| **Inv-DPendulum**   | 0.95%                   | 14.9%                       | 75.22%                  | 0.1                           | 0.61                            | 0.14                        |
| **Hopper**          | 2.51%                   | 64.99%                      | 57.22%                  | 0.46                          | 0.4                             | 0.25                        |
| **Walker2d**        | 2.14%                   | 69.47%                      | 47.45%                  | 0.27                          | 0.2                             | 0.21                        |
| **Ant**             | 0.82%                   | 22.01%                      | 12.68%                  | 0.43                          | 0.39                            | 0.38                        |
| **HalfCheetah**     | 5.64%                   | 14.07%                      | 18.05%                  | 0.8                           | 0.65                            | 0.67                        | 



| Environment     | Average Decisions (TD3) | Average Decisions (TempoRL) | Average Decisions (TLA) | Average MMACs (TD3) | Average MMACs (TempoRL) | Average MMACs (TLA) | 
|-----------------|-------------------------|-----------------------------|-------------------------|---------------------|-------------------------|---------------------|
| **Pendulum**        | 200                     | 139.39                       | 62.31                   | 24.30               | 34.14                   | 12.42               | 
| **MountainCar**     | 999                     | 116.47                       | 10.6                    | 120.98              | 28.60                   | 2.54                | 
| **Inv-Pendulum**    | 1000                    | 532.57                       | 111.79                  | 121.90              | 131.01                  | 26.05               | 
| **Inv-DPendulum**   | 1000                    | 850.95                       | 247.76                  | 124.70              | 213.59                  | 57.46               | 
| **Hopper**          | 998.99                  | 269.85                       | 423.91                  | 125.17              | 68.43                   | 72.02               |
| **Walker2d**        | 988.17                  | 297.4                        | 513.12                  | 127.08              | 77.29                   | 92.07               | 
| **Ant**             | 960.57                  | 741.22                       | 860.21                  | 160.22              | 248.53                  | 243.22              | 
| **HalfCheetah**     | 1000                    | 889.57                       | 831.42                  | 128.60              | 230.13                  | 182.35              | 






## Credits
The Temporally Layered Architecture was developed by [Devdhar Patel](https://www.devdharpatel.com/) under the supervision of Prof. [Terrence Sejnowski](https://www.salk.edu/scientist/terrence-sejnowski/) and Prof. [Hava Siegelmann](https://www.cics.umass.edu/about/directory/hava-siegelmann) at UMass Amherst. 

## Citation

The paper can be cited with the following bibtex entry:

```
@article{10.1162/neco_a_01718,
    author = {Patel, Devdhar and Sejnowski, Terrence and Siegelmann, Hava},
    title = "{Optimizing Attention and Cognitive Control Costs Using Temporally Layered Architectures}",
    journal = {Neural Computation},
    pages = {1-30},
    year = {2024},
    month = {10},
    issn = {0899-7667},
    doi = {10.1162/neco_a_01718},
    url = {https://doi.org/10.1162/neco\_a\_01718},
    eprint = {https://direct.mit.edu/neco/article-pdf/doi/10.1162/neco\_a\_01718/2474695/neco\_a\_01718.pdf},
}
```


[//]: # (## Contributing)

[//]: # ()
[//]: # (>📋  Pick a licence and describe how to contribute to your code repository. )
