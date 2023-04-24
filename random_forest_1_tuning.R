# {random forest #1} tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# load required objects ----
load(file = "results/titanic_setup.rda")

# recipe 1
titanic_rec_1 <- recipe(survived ~ ., data = titanic_train) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_rm(name) %>% 
  step_rm(passenger_id) %>% 
  step_rm(cabin) %>% 
  step_impute_mean(age) %>% 
  step_impute_mode(embarked) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) 

# Define model ----
rf_mod <- rand_forest(mode = "classification",
                      min_n = tune(),
                      mtry = tune()) %>% 
  set_engine("ranger")

# set-up tuning grid ----
rf_params <- parameters(rf_mod) %>% 
  # N := maximum number of random predictor columns we want to try 
  # should be less than the number of available columns
  update(mtry = mtry(c(1, 10))) 

# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(titanic_rec_1)

# Tuning/fitting ----
tic("random forest 1")

rf_tune <- tune_grid(
  rf_workflow,
  resamples = titanic_fold,
  grid = rf_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"))

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

rf_tictoc <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# metrics
autoplot(rf_tune, metric = "roc_auc")

rf_metric_1 <- rf_tune %>% 
  show_best(metric = "roc_auc")


# Write out results & workflow
save(rf_tune, rf_tictoc, rf_metric_1,
     file = "results/tuning_rf_1.rda")
