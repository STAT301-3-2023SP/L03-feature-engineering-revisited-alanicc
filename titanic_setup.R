# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(naniar)
library(knitr)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

## load data ----
titanic <- read_csv("data/titanic.csv") %>%
  janitor::clean_names() %>%
  mutate(
    survived = factor(survived, levels = c("Yes", "No")),
    pclass = factor(pclass)
  )


titanic_split <- initial_split(data = titanic, prop = 0.80, strata = survived)

titanic_test <- testing(titanic_split)

titanic_train <- training(titanic_split)

#titanic folds
titanic_fold <- vfold_cv(titanic_train, v = 5, repeats = 3, 
                         strata = survived)

save(titanic_fold, titanic_train, titanic_test, file = "results/titanic_setup.rda")


#skim 
skimr::skim_without_charts(titanic_train)

#kable of missing data
titanic_train %>% 
  miss_var_summary() %>% 
  kable()

#visualize
titanic %>% 
  vis_miss()
