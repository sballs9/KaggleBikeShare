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

# Linear Regression ----------------------------

# Set up and Fit Linear Model
my_linear_model <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression") |>
  fit(formula=log(count) ~ season + holiday + workingday + weather + temp + atemp + humidity + windspeed, data=training_data)

# Generate Predictions
bike_predictions <- predict(my_linear_model, new_data=testing_data)

# Format the Predictions for submission to Kaggle
kaggle_submission <- bike_predictions |>
  bind_cols(., testing_data) |>
  select(datetime, .pred) |>
  rename(count=.pred) |>
  mutate(count=pmax(0, count)) |>
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")
