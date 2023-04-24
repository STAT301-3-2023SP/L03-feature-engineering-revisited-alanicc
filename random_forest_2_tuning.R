# {random forest #2} tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# load required objects ----
load(file = "results/titanic_setup.rda")

# recipe 2
titanic_rec_2 <- recipe(survived ~., data = titanic_train) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_rm(name) %>% 
  step_rm(passenger_id) %>% 
  step_rm(cabin) %>% 
  step_impute_median(age) %>% 
  step_impute_knn(embarked) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

# Define model ----
rf_mod_2 <- rand_forest(mode = "classification",
                      min_n = tune(),
                      mtry = tune()) %>% 
  set_engine("ranger")

# set-up tuning grid ----
rf_params_2 <- parameters(rf_mod) %>% 
  # N := maximum number of random predictor columns we want to try 
  # should be less than the number of available columns
  update(mtry = mtry(c(1, 10))) 

# define tuning grid
rf_grid_2 <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_workflow_2 <- workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(titanic_rec_2)

# Tuning/fitting ----
tic("random forest 1")

rf_tune_2 <- tune_grid(
  rf_workflow_2,
  resamples = titanic_fold,
  grid = rf_grid_2,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"))

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

rf_tictoc_2 <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# metrics
autoplot(rf_tune_2, metric = "roc_auc")

rf_metric_2 <- rf_tune_2 %>% 
  show_best(metric = "roc_auc")


# Write out results & workflow
save(rf_tune_2, rf_tictoc_2, rf_metric_2,
     file = "results/tuning_rf_2.rda")
