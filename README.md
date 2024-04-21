# Gamma-Telescope-ML-project

This ML model simulates registration of high energy gamma particles in an atmospheric Cherenkov telescope.

### Dataset Information:
The data are MC generated to simulate registration of high energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope using the imaging technique.
Cherenkov gamma telescope observes high energy gamma rays,
taking advantage of the radiation emitted by charged particles produced inside the electromagnetic showers initiated by the gammas, and developing in the atmosphere.
This Cherenkov radiation (of visible to UV wavelengths) leaks through the atmosphere and gets recorded in the detector, allowing reconstruction of the shower parameters. 
The available information consists of pulses left by the incoming Cherenkov photons on the photomultiplier tubes, arranged in a plane, the camera.
Depending on the energy of the primary gamma, a total of few hundreds to some 10000 Cherenkov photons get collected,
in patterns (called the shower image), allowing to discriminate statistically those caused by primary gammas (signal) from the images of hadronic showers initiated by cosmic rays in the upper atmosphere (background).

Typically, the image of a shower after some pre-processing is an elongated cluster.
Its long axis is oriented towards the camera center if the shower axis is parallel to the telescope's optical axis, i.e. if the telescope axis is directed towards a point source.
A principal component analysis is performed in the camera plane, which results in a correlation axis and defines an ellipse.
If the depositions were distributed as a bivariate Gaussian, this would be an equidensity ellipse.
The characteristic parameters of this ellipse (often called Hillas parameters) are among the image parameters that can be used for discrimination.
The energy depositions are typically asymmetric along the major axis, and this asymmetry can also be used in discrimination.
There are, in addition, further discriminating characteristics, like the extent of the cluster in the image plane, or the total sum of depositions.



### Variable Information:
1.  fLength:  continuous  # major axis of ellipse [mm]
    2.  fWidth:   continuous  # minor axis of ellipse [mm] 
    3.  fSize:    continuous  # 10-log of sum of content of all pixels [in #phot]
    4.  fConc:    continuous  # ratio of sum of two highest pixels over fSize  [ratio]
    5.  fConc1:   continuous  # ratio of highest pixel over fSize  [ratio]
    6.  fAsym:    continuous  # distance from highest pixel to center, projected onto major axis [mm]
    7.  fM3Long:  continuous  # 3rd root of third moment along major axis  [mm] 
    8.  fM3Trans: continuous  # 3rd root of third moment along minor axis  [mm]
    9.  fAlpha:   continuous  # angle of major axis with vector to origin [deg]
   10.  fDist:    continuous  # distance from origin to center of ellipse [mm]
   11.  class:    g,h         # gamma (signal), hadron (background)

   g = gamma (signal):     12332
   h = hadron (background): 6688

   For technical reasons, the number of h events is underestimated. In the real data, the h class represents the majority of the events.

   The simple classification accuracy is not meaningful for this data,
   since classifying a background event as signal is worse than classifying a signal event as background. 
   For comparison of different classifiers an ROC curve has to be used. 
   The relevant points on this curve are those, 
   where the probability of accepting a background event as signal is below one of the following thresholds: 0.01, 0.02, 0.05, 0.1, 0.2
   depending on the required quality of the sample of the accepted events for different experiments.

### Dataset link:
```
http://rexa.info/paper?id=ac6e674e9af20979b23d3ed4521f1570765e8d68
```
