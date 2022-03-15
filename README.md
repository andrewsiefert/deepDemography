# deepDemography
This repository contains code and data to fit a deep learning model of tree growth rates using high-dimensional trait and environmental data. The model first creates low-dimensional representations (embeddings) of species' trait values and site environmental values, and then learns a mapping from these trait/environmental spaces to growth. These maps (performance landscapes) can be visualized for each point in environmental space, illustrating the relationships between multidimensional phenotypes and growth. 

The models are implemented using Keras and Tensorflow in Python. The code in <code>traitscape.py</code> defines the TraitScape model class, which includes methods to set up, fit, and evaluate models. 

A demonstration of the models using tree growth data from the eastern US (US Forest Service Inventory & Analysis), gap-filled trait data from TRY, and climate data from WorldClim, can be seen by running <code>code/fit_models.py</code>. The models are stochastic, so output will vary each time you run it, but here are examples of performance landscapes for different points in a two-dimensional climate space: 

![](/results/growth_landscapes.jpg)
  
