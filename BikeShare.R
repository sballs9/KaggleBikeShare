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

# Ensure 'weather' is a factor
training_data$weather <- as.factor(training_data$weather)
testing_data$weather <- as.factor(testing_data$weather)
training_data$season <- as.factor(training_data$season)
testing_data$season <- as.factor(testing_data$season)
training_data$holiday <- as.factor(training_data$holiday)
testing_data$holiday <- as.factor(testing_data$holiday)
training_data$workingday <- as.factor(training_data$workingday)
testing_data$workingday <- as.factor(testing_data$workingday)

# Replace 4th level with 3rd level
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

training_data$month <- as.factor(training_data$month)
testing_data$month <- as.factor(testing_data$month)
training_data$day <- as.factor(training_data$day)
testing_data$holiday <- as.factor(testing_data$holiday)
training_data$time_of_day <- as.factor(training_data$time_of_day)
testing_data$time_of_day <- as.factor(testing_data$time_of_day)

# Linear Regression ----------------------------

# Set up and fit linear model
my_linear_model <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression") |>
  fit(formula=log(count) ~ weather + atemp + temp + humidity + windspeed + month + day + time_of_day, data=training_data)

alias(my_linear_model$fit)
# Generate predictions
bike_predictions <- predict(my_linear_model, new_data=testing_data)

bike_predictions <- exp(bike_predictions)

bike_predictions

# Format the predictions for submission to Kaggle
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
