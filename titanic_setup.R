# Load package(s) ----
library(tidymodels)
library(tidyverse)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

## load data ----
titanic <- read_csv("data/titanic.csv") %>%
  janitor::clean_names() %>%
  mutate(
    survived = factor(survived, levels = c("yes", "no")),
    pclass = factor(pclass)
  )

