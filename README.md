# On the discussion of sparse measurements interpolation with dense grid covariate(s)
-------------------


0 Motivation
-------------------
The attempt to recover a complete continuous random field based upon only a few point measurements/observations is fascinating! Humans are taking such attempt to almost any scale in the pursuit of science and art. Using few boreholes for a geographer to [_Krige_](https://en.wikipedia.org/wiki/Kriging) the distribution of gold is no different from a sculptor using a point machine to sample out the shape of Bernini's masterpiece [_The Rape of Proserpina_](https://en.wikipedia.org/wiki/The_Rape_of_Proserpina). For both science and art, accuracy is the prime. While accuracy, during such type of processes, is conveyed implicitly through the vividness of art masterpieces, scientists are by no means to be less hypercritical and obsessing about the nature of the accuracy under different conditions. 

1 Background
-------------------
This snippet of scientific experiment is fully devoted to examine the accuracy in the problem of using point observations to recunstruct continuous field. You can imagine accuracy as a function of several properties of observation at hand: quality, density, distribution, variation, etc. Then there are several scenarios of field reconstruction accuracy depending on the combination of observation properties, which are well worth understanding. Exhaustively exploration of the scenarios is fundamental to understand the potentials and limitations of observations in reconstructing the underlying continuous field(s), or at least have an impression of reliability of using limited observations at hand, in real life cases. However, enumerate every scenario can be tedious. 

Here in this snippet, I would start to touch on this topic for both scientific and educational purpose. I try to significantly narrow down the scenarios in order to have a clean and easy start. First, working as geospatial data scientist, I would like to frame the experiment within the field of geography. But this is still overwhelming as there are already many versions of _Kriging_ or _Gaussian Process Regression_ taking care of using point observations in various situations. So, second, I will start with a special scenario where inference of continuous geospatial field through point observation is assisted by regularly gridded data, and envision that growing availability of airborne data would underpin this scenario as a major research theme very soon. This snippet would appear to be bizarre at the very beginning since it tries to be general on a very special case. But I believe as we build more upon the theme, the value would grow in either research or education. In a nutshell, we will:
 - explore the reliablity of using point observations along with covariate(s) in feild reconstruction in various scenarios;
 - explore the performance of commonly used algorithms in field construction with point observations and their grid covariate;
 - how different scenarios are manifested in situations with real life data.

Thus the experiment is deployed with dummy datasets and then the real ones.

2 Concepts
-------------------
Using Earth Observation (EO) data as a complement or even replacement of ground-based sparse measurements is prevalent in many mapping activities, relevant ones can be soil, humidity, pollutants, particles, energy, etc. Such application of EO data in understanding and interpreting geographic processes becomes universal along the increasing availability of satellite imagery data, which renders the interpolating point measurements assisted by gridded imagery data as a scientific problem. In reflection of the title, I coin the problem as **Sparse point interpolation with dense grid covariate(s)**. To investigate such problem, there are, again, already many potential scenarios: the grid covariate (satellite imagery data) can either weakly or strongly correlated to the point observation; the point observations can be dense or sparse, the point observations can be regularly or irregularly distributed; one or both of the grid covariate and point observations can be noise perturbed; the noises can be of different properties (noise colors) and intensities...But I will try to not reduce the scenarios any more and inspect them incrementally.

The concept is intuitive: I will start to explore how continuous field reconstruction is impacted by scenarios of observations at hand, this can simply be inspected by watching the accuracy varies along with different scenarios. Then I will use real datasets for comparison and see if such variation is manifested. 

### 2.1 Dummy dataset
To realize the concept, a dummy dataset is created representing any field that can be modelled by _f_ (Fig.1). In geostatistics, we also would like to know what the most conventional parameters are so that reconstructed field through geostatistical methods can be compared and validated. Thus I also try to approximate the dummy datasets using the most frequently used geostatistical method: the _Kriging_, which is more widely known as the _Gaussian Process_. Here I will continue to use _Gaussian Process_ for the notion that [it is more general in the field statistics and has a longer history](http://www.gaussianprocess.org/). Holding this dummy field with ground truth geostatistical parameters, the point observations and their grid covariate are created out of this ground truth field to simulate scenarios of situations we may encounter. For instance, a simple scenario could be achieved by linear transformation of the ground truth field to obtain the grid covariate, while the point observations can be [semi-random](https://blog.demofox.org/2017/05/29/when-random-numbers-are-too-random-low-discrepancy-sequences/) samples from the ground truth field (Fig.1). More general scenarios can be obtained by simulating and adding possible [noises](https://en.wikipedia.org/wiki/Colors_of_noise) on top of the simple scenario with only linearly transformation, whereas points can also be perturbed and more sparse.

<img src="/images/(20200130)Framework.png"> 

_Fig.1 Conceptual design for examining field reconstruction through dummy point and grid covariate datasets._

The reconstruction can be achieved through either conventional geostatistical methods, or more advanced machine learning based techniques, but in either case the problem remains in the domain of regression, so any regression technique is subject to be used for comparison or setting the benchmark (to be decided in progress).

### 2.2 Real datasets
The rationale of selecting real life cases is that continuous field is preferred. In the application of interpolating point observations, I would like to first avoid tricky situations of encountering sharp edges and patchness on the surface of fields. Thus I attempt to avoid any form of recognizable edges or boundaries in terms of the so called [_fiat_ and _bona fide_ boundaries](http://www.columbia.edu/~av72/papers/Ppr_2000.pdf). In this sense, airborne geographic phenonmenon which disperse over space and time would be handy choices. Here, I use near surface air temperature and pollutant NO<sub>2</sub> (Fig.2). Examination of these phenomena is practically significance as relevant to our environment, climate, and health. Below is a sample map of surface temperature and NO<sub>2</sub> at local scale around the city of Utrecht, the Netherlands.

<img src="/images/(20200130)realData.png" width="600" height="350">

_Fig.2 Real datasets: air pollution (left) and surface temperature (right)._

Both of the maps provide spatial distribution of temperature and NO<sub>2</sub> with resolution higher than 30m. The disperstion patterns on the two maps are distinctively different but intuitive and informative: high temperature is observed around densely built areas where building outlines are visible, whereas high concentration of NO<sub>2</sub> is around transportation arterials and road networks are highlighted. If one needs to see this kind of maps as frequent as daily or even hourly, no existing sensor would meet the demand. In fact, creating such maps daily or hourly largely relies on _in-situ_ observations, which can be dense or sparse depends on the place. In recent decades, the development of EO infrasctructure provides a significant complement to the _in-situ_ observations. However, the power of the EO in complementing the ground based _in-situ_ has not been thoroughly inspected. For instance, whether different spatial and temporal resolutions of the EO can capture the variations of the target phenomenon. More importantly, what EO actually "see" can be quite different from the target phenomenon measured near the ground, such as temperature--[the gap between surface and near surface air temperature is insufficiently understood](https://www.sciencedirect.com/science/article/pii/S0034425703000798). This potential weak correlation is well aligned with the scenario of using less desirable grid covariate for point observation interpolation. In short, framing these real life cases into the problem of **Sparse point interpolation with dense grid covariate(s)** is suitable.

3 Techniques
-------------------
### 3.1 Incremental actions
A general framework of playing with dummy datasets for scenario analysis needs to encompass (again can be referred back to Fig.1):
 - ground truth continuous random field generated by any arbitrary function _f_;
 - ground truth geostatistical parameter approximated by [_Gaussian Process_](https://github.com/SheffieldML/notebook/tree/deploy/GPy);
 - scenario of sparse or dense point observations _Z_ generated by sampling from the ground truth field _f_;
 - scenario of grid covariate _g_ generated by transforming the ground truth field _f_;
 - reconstruct the ground truth _f_ as a simulated field _f'_ by using _Z_ and _g_;
 - inspect the results in terms of accuracy/uncertainties (the inspection is conducted by applying various algorithms varying from conventional geostatistical inference to advanced machine learning techniques).
 
Critical actions can easily be identified at each point of data generation: ground truth field, point observations, and grid covariate. Selection of a particular algorithm for reconstructing the field can also be considered as another action, which I will put aside for now. Let's focus on data generation. In order to examine the accuracy as a function of the properties of the generated data, any **action** generating the data with certain property needs to be parameterized. For instance, the accuracy can be inspected by one of the property of the generated data--the noise intensity or _variance_ of the added noise to the point samples. Concretely, the accuracy is governed as:

![e1]

[e1]: http://chart.apis.google.com/chart?cht=tx&chl=acc=f(actions)
 ,where **actions** are parameterized according to the nature of the actions themselves. For instance, the **action** of generating point samples can be parameterized by the distribution of the sample locations. In this case, by saying "distribution", I need quantities such as _randomness_ and _adjacency_ (sparsity or density to separate samples). I also need sampling schemes such as [_blind_ or _adaptive_](https://ieeexplore.ieee.org/abstract/document/6112220) to create samples following parameterized distribution. For another instance, to make the point samples less ideal and more general, the **action** of adding noise to point samples can be parameterized according to how intense the noise can be. In this case, by saying "how intense" or "intensity", I may further constrain the idea as two quantities: _variance_(noise value is large or small) and _lengthscale_(how quickly the noise varies across space). However, apart from these particular cases, how many actions are expected and what are the parameters? It seems the **action(s)** can be arranged incrementally to render dummy datasets from ideal to general as in table below:

_Table 1 Actions for parameterizing scenarios that can be conducted incrementally._

<img src="/images/(20200210)actions.png"> 

Here, I identify three major **actions**, which can be conducted incrementally to generalize scenarios, such as **perturbation** can be added on top of point sampling to approximate a realistic condition. The **actions** boil down hierarchically into quantities to be parameterized, which successively forms scenarios. According to the amount of quantities to be parameterized in Table 1, it seems, again, many scenarios can be explored. But now the steps are clear, the starting point can be the most basic ones with blind sampling of point observations and covariate obtaind through simple linear transformation of the ground truth continuous field.

### 3.2 Action parameterization
One option for the basic scenarios can be **blind sampling** of point obsevations and **linear transformation covariate** without any **perturbation**. This basic scenario can be a showcase of action parameterization. In this case, I also fixed the linearly transformed covariate simply as -1.0 scaled original ground truth field. Thus only the point sampling scheme can be parameterized. By referring to **blind sampling**, I created the artificial point observation of the original field by applying the [_Poisson-Disc Sampling_ ](https://www.jasondavies.com/poisson-disc/) to ensure there will be no clusters or gaps among the sampled points caused by the commonly used pure random sampling. The _Poisson-Disc Sampling_ leads to a more natural point patterns distributed over an area. With this specific **blind sampling** scheme, there is only one parameter needed to control this sampling action: density or sparsity. Briefly, the basic scenario would be a reduced version of Fig.1 as shown in Fig.3 below. 

<img src="/images/04_noiseFreeWorkflow.png"> 

_Fig.3 Basic scenario with sparsity of point observation as parameterized action._

Frankly, noise-free situation is too ideal to leverage the strength of any type of Gaussian Process modeling and be used as a standard scenario to benchmark different inference technique. By hypothesizing the points and covariate as correlated Gaussian Processes, the _Coregionalized Gaussian Process modeling_ implemented in python as [_GPy_](https://github.com/SheffieldML/notebook/tree/deploy/GPy) can easily find out that the correlation between the point observations of the field and covariate is -1.0 as the values in the points vary rigorously to the opposite to the corresponding values in the covariate. Sparsifying the point observations does not impact the result thus the accuracy stays with RMSE=0. More specifically, as I created the ground truth field with a variation lengthscale of 3.0, the _Gaussian Process modeling_ also found this lengthscale of the ground truth field quite accurately along the changing sparsity of the artificial point observation (Fig.4).

<img src="/images/04_noiseFreeLenInf.png" width="500" height="250"> 

_Fig.3 Inference of the lengthscale of the ground truth field along sparsified point observations._

It is safe to conclude that making inference about the underlying field with only a few point observations can be quite reliable under the noise-free scenario. Of course, the covariate is also created with a very simple and restricted manner.





