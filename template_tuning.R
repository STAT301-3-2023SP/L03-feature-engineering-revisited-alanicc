# {MODEL TYPE} tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# load required objects ----


# Define model ----


# set-up tuning grid ----


# define tuning grid


# workflow ----


# Tuning/fitting ----
tic("MODEL TYPE")
# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

YOUR_NAME_CHOICE <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# Write out results & workflow

