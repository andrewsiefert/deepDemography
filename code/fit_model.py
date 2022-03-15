import pandas as pd
import numpy as np

from traitscapes import TraitScape


# load data
X = pd.read_csv('data/X.csv')
y = pd.read_csv("data/y.csv")

# remove trees with negative and very large growth rates
drop = (y.ann_dia_growth > 0.5) | (y.ann_dia_growth < 0)
X = X.loc[-drop, :]
y = y.loc[-drop, :]

# set up and fit model
ts = TraitScape(X, y)
ts.split_data()
ts.create_transformer()
ts.create_model(e_hidden=3,
                e_units=32,
                g_hidden=2, g_units=32,
                dropout=0,
                lr=0.0005,
                epochs=50)

# evaluate model
ts.null_mse()   # baseline test set MSE
ts.evaluate()   # model test set MSE

# get climate, soil, and trait embeddings for test set
clim_emb, soil_emb, trait_emb = ts.get_embeddings(ts.X_test)

# look at trait "loadings" on trait embedding axes
traits = ts.X_test.iloc[: , ts.trait_ind]
traits.apply(lambda x: x.corr(pd.Series(trait_emb[:, 0]))).sort_values()
traits.apply(lambda x: x.corr(pd.Series(trait_emb[:, 1]))).sort_values()

# look at climate/topographic variable "loadings" on climate embedding axes
clim = ts.X_test.iloc[: , ts.clim_ind]
clim.apply(lambda x: x.corr(pd.Series(clim_emb[:, 0]))).sort_values()
clim.apply(lambda x: x.corr(pd.Series(clim_emb[:, 1]))).sort_values()

# create grid of climate and trait embedding values
clim1 = np.linspace(np.min(clim_emb[:, 0]), np.max(clim_emb[:, 0]), 5).tolist()
clim2 = np.linspace(np.min(clim_emb[:, 1]), np.max(clim_emb[:, 1]), 5).tolist()
trait1 = np.linspace(np.min(trait_emb[:, 0]), np.max(trait_emb[:, 0]), 20).tolist()
trait2 = np.linspace(np.min(trait_emb[:, 1]), np.max(trait_emb[:, 1]), 20).tolist()
soil1 = np.mean(soil_emb[:, 0])
soil2 = np.mean(soil_emb[:, 1])
dia1 = [8]

points = np.array(np.meshgrid(clim1, clim2, soil1, soil2, trait1, trait2, dia1)).T.reshape(-1, 7)

clim_grid = points[:, 0:2]
soil_grid = points[:, 2:4]
trait_grid = points[:, 4:6]
dia_grid = points[:, 6]

# get predictions for grid values
growth_grid = ts.growth_model.predict([clim_grid, soil_grid, trait_grid, dia_grid])

# create data frame of grid points and predictions
df = pd.DataFrame(np.concatenate((clim_grid, trait_grid, growth_grid), axis=1),
                  columns=['clim1', 'clim2', 'trait1', 'trait2', 'growth'])

# export values
df.to_csv("results/growth_grid.csv", index=False)