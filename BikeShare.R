library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)
library(patchwork)

# Goal: Predict the hourly count from the 20th to the end of the month

# Read in the data
training_data <- vroom("train.csv")
testing_data <- vroom("test.csv")

# EDA ------------------------------

plot1 <- ggplot(data = training_data, aes(x = temp, y = count)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  xlab("Temperature") +
  ylab("Rental Count")

plot1

plot2 <- ggplot(data = training_data, aes(x = factor(weather))) +
  geom_bar(aes(fill = factor(weather))) +
  xlab("Weather Type") +
  ylab("Hours")

plot2

hourly_count <- training_data |>
  group_by(weather) |>
  summarise(n = n(), count = sum(count) / n)

plot3 <- ggplot(data = hourly_count, aes(x = factor(weather), y = count)) +
  geom_col(aes(fill = factor(weather))) +
  xlab("Weather Type") +
  ylab("Average Rental Count per Hour")

plot3 <- plot3 + scale_fill_brewer(palette = "Set1")

plot3

plot4 <- ggplot(data = training_data, aes(x = windspeed)) +
  geom_histogram(fill='firebrick') +
  xlab("Windspeed") +
  ylab("Rental Count")
  
plot4

hourly_data <- training_data |>
  mutate(date = as.Date(datetime), hour_of_day = hour(datetime))

hourly_data2 <- hourly_data |>
  group_by(hour_of_day) |>
  summarise(avg_count = mean(count))

plot5 <- ggplot(data = hourly_data2, aes(x = hour_of_day, y = avg_count)) +
  geom_col(aes(fill = factor(hour_of_day))) +
  xlab("Hour of Day") +
  ylab("Average Rental Count")
  
plot5

final_plot <- (plot2 + plot3) / (plot4 + plot5)

final_plot

# Linear Regression w/ Personal Feature Engineering 

training_data <- training_data %>%
  mutate(across(c("weather", "season", "holiday", "workingday"), as.factor))

testing_data <- testing_data %>%
  mutate(across(c("weather", "season", "holiday", "workingday"), as.factor))

levels(training_data$weather)[4] <- levels(training_data$weather)[3]
levels(testing_data$weather)[4] <- levels(testing_data$weather)[3]

training_data <- training_data |>
  mutate(date = as.Date(datetime), hour_of_day = hour(datetime))

training_data <- training_data |>
  mutate(day=weekdays(date), month=month(date))

training_data$time_of_day <- as.factor(ifelse(training_data$hour_of_day<7, "night",
                                       ifelse(training_data$hour_of_day<15, "morning",
                                       ifelse(training_data$hour_of_day<23, "evening",
                                       ifelse(training_data$hour_of_day<24, "night")))))

testing_data <- testing_data |>
  mutate(date = as.Date(datetime), hour_of_day = hour(datetime))

testing_data <- testing_data |>
  mutate(day=weekdays(date), month=month(date))

testing_data$time_of_day <- as.factor(ifelse(testing_data$hour_of_day<7, "night",
                                      ifelse(testing_data$hour_of_day<15, "morning",
                                      ifelse(testing_data$hour_of_day<23, "evening",
                                      ifelse(testing_data$hour_of_day<24, "night")))))

training_data <- training_data %>%
  mutate(across(c("month", "day", "time_of_day"), as.factor))

testing_data <- testing_data %>%
  mutate(across(c("month", "day", "time_of_day"), as.factor))

my_og_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula=log(count) ~ weather + atemp + temp + humidity + windspeed + month + day + time_of_day, data=training_data)
      
predictions <- predict(my_og_linear_model, new_data = testing_data)

predictions <- exp(predictions)

kaggle_submission <- predictions %>%
  bind_cols(., testing_data) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=pmax(0, count)) |>
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./OGLinearPreds.csv", delim=",")

# Penalized Regression using Recipe ----------------------------

install.packages('glmnet')

training_data <- vroom("train.csv")
testing_data <- vroom("test.csv")

training_data <- training_data %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

my_recipe <- recipe(count ~ ., data = training_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_time(datetime, features=c("hour")) %>%
  step_date(datetime, features=c("month")) %>%
  step_cut(datetime_hour, breaks=c(7, 15, 24)) %>%
  step_rm(datetime, temp, season, holiday, workingday) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

preg_model <- linear_reg(penalty = 0.01, mixture = 1) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data = training_data)

preg_predictions <- predict(preg_wf, new_data = testing_data)

preg_predictions <- exp(preg_predictions)

kaggle_submission <- preg_predictions %>%
  bind_cols(., testing_data) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=pmax(0, count)) |>
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./PenalizedRegPreds.csv", delim=",")


# Simple Linear Regression using Recipe

my_linear_model <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression") 

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_linear_model) %>%
  fit(data = training_data)

bike_predictions <- predict(bike_workflow, new_data = testing_data)

bike_predictions <- exp(bike_predictions)

bike_predictions

kaggle_submission <- bike_predictions %>%
  bind_cols(., testing_data) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=pmax(0, count)) |>
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")


# Poisson Regression -----------------------------

library(poissonreg)

my_pois_model <- poisson_reg() %>%
  set_engine("glm") %>%
  set_mode("regression") %>%
  fit(formula=count ~ weather + temp + atemp + humidity + windspeed + month + day + time_of_day, data=training_data)

# Generate predictions
pois_bike_predictions <- predict(my_pois_model, new_data=testing_data)

pois_bike_predictions

# Format for submission
pois_kaggle_submission <- pois_bike_predictions %>%
  bind_cols(., testing_data) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=pmax(0, count)) |>
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=pois_kaggle_submission, file="./PoisPreds.csv", delim=",")
