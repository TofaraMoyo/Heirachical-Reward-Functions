[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporally-layered-architecture-for-efficient/continuous-control-on-pendulum-v1)](https://paperswithcode.com/sota/continuous-control-on-pendulum-v1?p=temporally-layered-architecture-for-efficient)

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

[//]: # (## Evaluation)

[//]: # ()
[//]: # (To evaluate my model on ImageNet, run:)

[//]: # ()
[//]: # (```eval)

[//]: # (python eval.py --model-file mymodel.pth --benchmark imagenet)

[//]: # (```)

[//]: # ()
[//]: # (>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results &#40;section below&#41;.)

[//]: # (## Pre-trained Models)

[//]: # ()
[//]: # (You can download pretrained models here:)

[//]: # ()
[//]: # (- [My awesome model]&#40;https://drive.google.com/mymodel.pth&#41; trained on ImageNet using parameters x,y,z. )

[//]: # ()
[//]: # (>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained &#40;if applicable&#41;.  Alternatively you can have an additional column in your results table with a link to the models.)

[//]: # ()
[//]: # (## Results)

[//]: # ()
[//]: # (Our model achieves the following performance on :)

[//]: # ()
[//]: # (### [Image Classification on ImageNet]&#40;https://paperswithcode.com/sota/image-classification-on-imagenet&#41;)

[//]: # ()
[//]: # (| Model name         | Top 1 Accuracy  | Top 5 Accuracy |)

[//]: # (| ------------------ |---------------- | -------------- |)

[//]: # (| My awesome model   |     85%         |      95%       |)

[//]: # ()
[//]: # (>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. )

[//]: # ()
[//]: # ()
[//]: # (## Contributing)

[//]: # ()
[//]: # (>📋  Pick a licence and describe how to contribute to your code repository. )
