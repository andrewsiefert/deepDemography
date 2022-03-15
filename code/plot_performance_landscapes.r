library(tidyverse)


# read in growth predictions
d <- read_csv("results/growth_grid.csv") %>%
  mutate_at(vars(clim1, clim2), ~round(., 1)) %>%
  mutate(clim1 = paste("clim1 =", clim1),
         clim2 = paste("clim2 =", clim2))


# plot growth performance landscapes for different points in climate space
ggplot(d, aes(x = trait1, y = trait2, fill = growth)) +
  geom_tile() +
  geom_contour(aes(z = growth), bins = 30, color = "white", alpha = 0.5) +
  facet_grid(clim1 ~ clim2) + 
  scale_color_viridis_c() +
  scale_fill_viridis_c()

ggsave("results/growth_landscapes.pdf")
