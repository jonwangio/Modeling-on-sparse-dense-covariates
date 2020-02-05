# On the discussion of sparse measurements interpolation with dense grid covariate(s)
-------------------


0 Motivation
-------------------
The attempt to recover a complete continuous random field based upon only a few point measurements/observations is fascinating! Humans are taking such attempt to almost any scale in the pursuit of science and art. Using few boreholes for a geographer to [_Krige_](https://en.wikipedia.org/wiki/Kriging) the distribution of gold is no different from a sculptor using a point machine to sample out the shape of Bernini's masterpiece [_The Rape of Proserpina_](https://en.wikipedia.org/wiki/The_Rape_of_Proserpina). For both science and art, accuracy is the prime. While accuracy, during such type of processes, is conveyed implicitly through the vividness of art masterpieces, scientists are by no means to be less hypercritical and obsessing about the nature of the accuracy under different conditions. 

1 Background
-------------------
This snippet of scientific experiment is fully devoted to examine the accuracy in the problem of using point observations to recunstruct continuous field. You can imagine accuracy as a function of several properties of observation at hand: quality, density, distribution, variation, etc. Then there are several scenarios of field reconstruction accuracy depending on the combination of observation properties, which are well worth understanding. Exhaustively exploration of the scenarios is fundamental to understand the potentials and limitations of observations in reconstructing the underlying continuous field(s), or at least have an impression of reliability of using limited observations at hand, in real life cases. However, enumerate every scenario can be tedious. 

Here in this snippet, I would start to touch on this topic for both scientific and educational purpose. I try to significantly narrow down the scenarios in order to have a clean and easy start. First, working as geospatial data scientist, I would like to frame the experiment within the field of geography. But this is still overwhelming as there are already many versions of _Kriging_ or _Gaussian Process Regression_ taking care of using point observations in various situations. So, second, I will start with a special scenario where inference of continuous geospatial field through point observation is assisted by regularly gridded data, and envision that growing availability of airborne data would underpin this scenario as a major research theme very soon. This snippet would appear to be bizarre at the very beginning since it tries to be general on a very special case. But I believe as we build more upon the theme, the value would grow in either research or education.

2 Concepts and techniques
-------------------
Using Earth Observation (EO) data as a complement or even replacement of ground-based sparse measurements is prevalent in many mapping activities, relevant ones can be soil, humidity, pollutants, particles, energy, etc. Such application of EO data in understanding and interpreting geographic processes becomes universal along the increasing availability of satellite imagery data, which renders the interpolating point measurements assisted by gridded imagery data as a scientific problem. In reflection of the title, I coin the problem as **Sparse point interpolation with dense grid covariate(s)**. To investigate such problem, there are, again, already many potential scenarios: the grid covariate (satellite imagery data) can either weakly or strongly correlated to the point observation; the point observations can be dense or sparse, the point observations can be regularly or irregularly distributed; one or both of the grid covariate and point observations can be noise perturbed; the noises can be of different properties (noise colors) and intensities...But I will try to not reduce the scenarios any more and inspect them incrementally.

### Concepts
The concept is intuitive. 






<img src="/images/(20200130)Framework.png">





<img src="/images/00_origFunc.png" width="350" height="300">
