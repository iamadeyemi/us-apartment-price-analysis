# Install necessary packages if they haven't been installed yet
install.packages(c("stringr", "viridis", "knitr", "tidyverse", "ggplot2", 
                   "descstat", "psych", "reshape2", "tibble", "forecast"))

# Load libraries
library(stringr)
library(viridis)
library(knitr)
library(tidyverse)
library(ggplot2)
library(descstat)
library(psych)
library(reshape2)
library(tibble)

# Read CSV file (assuming it exists at this path and is correctly formatted)
df <- read_csv("C:/Users/Sir_Peluxx/Desktop/Project Work/Assignment/Md Hossain/apartments.csv")

# Display first few rows of original dataframe for verification
head(df)

# Remove unnecessary columns
df_cleaned <- df %>%
  select(-c(id, category, title, body, amenities, currency, fee, has_photo, pets_allowed, 
            price_display, price_type, address, source, time, cityname))

# Suppress structure output after cleaning
invisible(capture.output(str(df_cleaned)))

cat("\n✅ Data loaded and preprocessed successfully.\n")

head(df_cleaned)


# Function to generate data description
describe_data <- function(df) {
  
  # Get variable names and data types
  var_info <- data.frame(
    Variable = names(df),
    Data_Type = sapply(df, class),
    Description = rep(NA, length(names(df))) 
  )
  
  # Identify qualitative variables
  qualitative_vars <- var_info %>%
    filter(Data_Type %in% c("factor", "character")) %>%
    select(Variable)
  
  # Get number of observations
  sample_size <- nrow(df)
  
  # Print results
  cat("📌 **Data Description**\n\n")
  print(var_info)
  
  cat("\n📌 **Qualitative Variables:**\n")
  print(qualitative_vars)
  
  cat("\n📌 **Number of Observations (Sample Size):**", sample_size, "\n")
}

describe_data(df)

# Summary statistics
describe(df_cleaned)

library(ggplot2)
library(plotly)

# Function to create an interactive histogram
plot_distribution <- function(data, column) {
  p <- ggplot(data, aes(x = .data[[column]])) +
    geom_histogram(fill = "blue", color = "black", bins = 30, alpha = 0.7) +
    geom_density(aes(y = after_stat(density) * max(after_stat(count))), 
                 color = "red", linewidth = 1) +
    labs(title = paste("Distribution of", column), x = column, y = "Count") +
    theme_minimal()
  
  ggplotly(p) %>% layout(title = list(text = paste("Distribution of Dataset"))) # Corrected title format
}

# Generate distribution plots
p1 <- plot_distribution(df_cleaned, "price")
p2 <- plot_distribution(df_cleaned, "bathrooms")
p3 <- plot_distribution(df_cleaned, "bedrooms")
p4 <- plot_distribution(df_cleaned, "square_feet")

# Display plots in a grid
subplot(p1, p2, p3, p4, nrows = 2, margin = 0.05, titleX = TRUE) 


# Price vs Square Feet
suppressWarnings({
  p1 <- ggplot(df_cleaned, aes(x = square_feet, y = price)) +
    geom_point(alpha = 0.5, color = "blue") +
    geom_smooth(method = "lm", color = "red", linewidth = 1) +  
    labs(title = "Price vs Square Feet", x = "Square Feet", y = "Price") +
    theme_minimal()
  
  ggplotly(p1)  
})

# Price vs Number of Bedrooms
suppressWarnings({
  p2 <- ggplot(df_cleaned, aes(x = bedrooms, y = price)) +
    geom_boxplot(fill = "lightblue", color = "black") +
    labs(title = "Price vs Bedrooms", x = "Bedrooms", y = "Price") +
    theme_minimal()
  
  ggplotly(p2)
})

# Select only numeric variables
df_numeric <- df_cleaned %>%
  select(price, bathrooms, bedrooms, square_feet)

# Compute correlation matrix
cor_matrix <- cor(df_numeric, use = "complete.obs")

# Convert matrix for interactive plot
cor_data <- as.data.frame(as.table(cor_matrix))
names(cor_data) <- c("Variable1", "Variable2", "Correlation")

# Create interactive correlation heatmap
p_corr <- plot_ly(
  data = cor_data, 
  x = ~Variable1, 
  y = ~Variable2, 
  z = ~Correlation, 
  type = "heatmap",
  colorscale = "RdBu",  
  colorbar = list(title = "Correlation")
) %>%
  layout(
    title = "📊 Correlation Matrix",
    xaxis = list(title = "", tickangle = 45),
    yaxis = list(title = ""),
    margin = list(l = 100, r = 100, t = 50, b = 50)
  )

p_corr <- p_corr %>%
  add_annotations(
    x = cor_data$Variable1,
    y = cor_data$Variable2,
    text = round(cor_data$Correlation, 2),
    showarrow = FALSE,
    font = list(color = "black", size = 14)
  )

p_corr

# Count listings by state
state_counts <- df_cleaned %>%
  group_by(state) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head(10)

# Convert state to a factor
state_counts$state <- factor(state_counts$state, levels = rev(state_counts$state))

# Barplot of top 10 states
p3 <- ggplot(state_counts, aes(x = state, y = count)) +
  geom_bar(stat = "identity", fill = viridis(10, option = "C"), color = "black") +
  geom_text(aes(label = count), hjust = -0.2, size = 5, fontface = "bold") +
  labs(title = "🏡 Top 10 States with the Most Apartment Listings", 
       subtitle = "Based on available rental listings in the dataset", 
       x = "State", 
       y = "Number of Listings") +
  theme_minimal() +
  coord_flip()
ggplotly(p3)

# Boxplot for price
p4 <- ggplot(df_cleaned, aes(y = price)) +
  geom_boxplot(fill = "orange", color = "black") +
  labs(title = "Boxplot of Apartment Prices", y = "Price") +
  theme_minimal()
ggplotly(p4)

# Aggregate listings by state
state_counts <- df_cleaned %>%
  group_by(state) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

p_treemap <- plot_ly(
  state_counts,
  type = "treemap",
  labels = ~state,
  parents = NA,
  values = ~count,
  textinfo = "label+value",
  marker = list(
    colorscale = "Viridis",  
    line = list(color = "black", width = 1)  
  )
) %>%
  layout(
    title = "📊 Interactive Treemap of Listings by State",
    paper_bgcolor = "white", 
    plot_bgcolor = "white" 
  )

# Display treemap
p_treemap

# Fit the initial multiple linear regression model
mlr_model <- lm(price ~ bathrooms + bedrooms + square_feet, data = df_cleaned)

# Function to create interactive residual plots
plot_residuals <- function(x, y, title, xlab, ylab) {
  p <- ggplot(data.frame(x, y), aes(x = x, y = y)) +
    geom_point(color = "blue", alpha = 0.7) +
    geom_smooth(method = "loess", color = "red", linewidth = 1) +
    labs(title = title, x = xlab, y = ylab) +
    theme_minimal(base_size = 14)
  
  ggplotly(p) %>% layout(title = title)
}

# Compute residual diagnostics
fitted_values <- fitted(mlr_model)
residuals <- resid(mlr_model)
std_residuals <- rstandard(mlr_model)
leverage <- hatvalues(mlr_model)

# Generate diagnostic plots
p1 <- plot_residuals(fitted_values, residuals, 
                     "Residuals vs Fitted", "Fitted Values", "Residuals")

p2 <- plot_residuals(qnorm(ppoints(length(std_residuals))), sort(std_residuals), 
                     "Normal Q-Q Plot", "Theoretical Quantiles", "Standardized Residuals")

p3 <- plot_residuals(fitted_values, abs(std_residuals), 
                     "Scale-Location Plot", "Fitted Values", "√|Standardized Residuals|")

p4 <- plot_residuals(leverage, std_residuals, 
                     "Residuals vs Leverage", "Leverage", "Standardized Residuals")

# Arrange all plots in a 2x2 layout with better spacing
subplot(p1, p2, p3, p4, nrows = 2, margin = 0.07, titleX = TRUE) %>%
  layout(title = "Residual Diagnostic Plots")


# Step 2: Checking Model Fit Metrics
print("Step 2: Checking model fit metrics...")

# Get model summary
model_summary <- summary(mlr_model)

# Extract key metrics
r_squared <- model_summary$r.squared
adj_r_squared <- model_summary$adj.r.squared
f_statistic <- model_summary$fstatistic[1]
p_value <- pf(f_statistic, model_summary$fstatistic[2], model_summary$fstatistic[3], lower.tail = FALSE)

# Create a data frame for display
metrics_df <- data.frame(
  Metric = c("R-squared", "Adjusted R-squared", "F-statistic", "P-value"),
  Value = c(round(r_squared, 4), round(adj_r_squared, 4), round(f_statistic, 4), format(p_value, scientific = TRUE))
)

fig <- plot_ly(
  type = 'table',
  header = list(
    values = c("<b>Metric</b>", "<b>Value</b>"),
    fill = list(color = "lightgray"),
    align = "center"
  ),
  cells = list(
    values = rbind(metrics_df$Metric, metrics_df$Value),
    fill = list(color = "white"),
    align = "center"
  )
)

# Display table
fig


# ==========================
# Influential Points (Cook’s Distance)
# ==========================
print("Step 3: Checking for influential observations using Cook’s Distance...")

# Calculate Cook’s Distance
cook_d <- cooks.distance(mlr_model)

# Set threshold for high influence
threshold <- 4 / length(cook_d)

# Find observations with high influence
high_influence_points <- which(cook_d > threshold)

if (length(high_influence_points) > 0) {
  print(paste("Identified", length(high_influence_points), "potentially influential observations."))
  print("Indices of high-influence points:")
  print(high_influence_points)
} else {
  print("No highly influential observations detected.")
}

install.packages("car")  
library(car)            

# ==========================
# 4. Handling Problematic Observations
# ==========================
print("Step 4: Checking and Removing Influential Observations...")

# Calculate Cook’s Distance
cook_d <- cooks.distance(mlr_model)

# Define threshold (common rule: 4/N)
threshold <- 4 / nrow(df_cleaned)

# Find problematic observations
problematic_points <- which(cook_d > threshold)

if (length(problematic_points) > 0) {
  print(paste("Identified", length(problematic_points), "influential observations."))
  
  # Display problematic points before removal
  print("Problematic Observations (Indices):")
  print(problematic_points)
  
  # Dropping the influential observations
  print("Dropping identified observations and refitting the model...")
  df_cleaned <- df_cleaned[-problematic_points, ]
  
  # Refit the model after removal
  mlr_model <- lm(price ~ ., data = df_cleaned)  # Refit using all predictors
  print("Model refitted after removing problematic observations.")
  
} else {
  print("No highly influential observations detected. Proceeding with the current dataset.")
}


# Load necessary packages
library(car)       # For vif()
library(dplyr)     # For mutate() and %>%
library(plotly)    # For interactive tables

print("Step 2: Checking for multicollinearity using Variance Inflation Factor (VIF)...")

# Compute VIF values
vif_values <- vif(mlr_model)

# Convert VIF values to a data frame
vif_df <- data.frame(
  Predictor = names(vif_values),  # Extract variable names
  VIF = vif_values  # Use VIF values directly
)

# Set VIF threshold
vif_threshold <- 5

# Add a flag for high multicollinearity
vif_df <- vif_df %>%
  mutate(Flag = ifelse(VIF > vif_threshold, "High Multicollinearity", "Acceptable"))

# Print high VIF predictors
if (any(vif_df$VIF > vif_threshold)) {
  print("The following predictors have high multicollinearity (VIF > 5):")
  print(vif_df$Predictor[vif_df$VIF > vif_threshold])
}

# Create an interactive table with Plotly
vif_table <- plot_ly(
  type = 'table',
  header = list(
    values = c("<b>Predictor</b>", "<b>VIF</b>", "<b>Status</b>"),
    fill = list(color = "lightgray"),
    align = "center"
  ),
  cells = list(
    values = rbind(vif_df$Predictor, round(vif_df$VIF, 2), vif_df$Flag),  # Round VIF values for readability
    fill = list(color = "white"),
    align = "center"
  )
)

vif_table  # Display the table

# Create interactive VIF Bar Chart
vif_plot <- plot_ly(
  vif_df,
  x = ~Predictor,
  y = ~VIF,
  type = 'bar',
  marker = list(
    color = ifelse(vif_df$VIF > vif_threshold, "red", "blue")
  )
) %>%
  layout(
    title = "Variance Inflation Factor (VIF) Analysis",
    xaxis = list(title = "Predictor Variables"),
    yaxis = list(title = "VIF Value"),
    shapes = list(
      list(
        type = "line",
        x0 = -0.5, x1 = length(vif_df$Predictor) - 0.5,
        y0 = vif_threshold, y1 = vif_threshold,
        line = list(color = "red", dash = "dash")
      )
    )
  )

subplot(vif_table, vif_plot, nrows = 2, margin = 0.07, titleX = TRUE)


# ==========================
# 6. Removing High-VIF Predictors and Refitting Model
# ==========================
print("Step 6: Removing highly collinear predictors (VIF > 5) and refitting the model...")

# Drop high-VIF predictors
df_cleaned <- df_cleaned[, !names(df_cleaned) %in% c("state", "latitude", "longitude")]

# Refit the multiple linear regression model
mlr_model <- lm(price ~ ., data = df_cleaned)

# Print new model summary
print("Model refitted after removing high-VIF predictors. Here's the updated summary:")
summary(mlr_model)

# Recheck VIF after removing problematic variables
print("Recalculating VIF to ensure multicollinearity is addressed...")
vif_values <- vif(mlr_model)
print("Updated VIF values:")
print(vif_values)


# Load required libraries
library(car)
library(plotly)
library(dplyr)
library(broom)

print("Step 6: Removing highly collinear predictors (VIF > 5) and refitting the model...")

# Compute original VIF values before removing variables
original_vif <- vif(mlr_model)

# Print VIF results
print("Initial VIF values calculated. Checking for high multicollinearity...")

# Drop high-VIF predictors
high_vif_predictors <- c("state", "latitude", "longitude")
df_cleaned <- df_cleaned[, !names(df_cleaned) %in% high_vif_predictors]

# Refit the multiple linear regression model
mlr_model <- lm(price ~ ., data = df_cleaned)

print("Model refitted after removing high-VIF predictors.")

# Recheck VIF after removing problematic variables
updated_vif <- vif(mlr_model)
print("Updated VIF values calculated. Proceeding with model summary...")

# Extract model summary statistics
model_summary <- glance(mlr_model)
coefficients_summary <- tidy(mlr_model)

# Convert summary statistics to a data frame for Plotly
summary_df <- data.frame(
  Metric = c("R-squared", "Adjusted R-squared", "Residual Standard Error", "F-statistic", "p-value"),
  Value = c(
    round(model_summary$r.squared, 4),
    round(model_summary$adj.r.squared, 4),
    round(model_summary$sigma, 4),
    round(model_summary$statistic, 4),
    format(model_summary$p.value, scientific = TRUE)
  )
)

# Convert coefficients to a data frame for Plotly
coefficients_df <- data.frame(
  Predictor = coefficients_summary$term,
  Estimate = round(coefficients_summary$estimate, 4),
  Std_Error = round(coefficients_summary$std.error, 4),
  t_value = round(coefficients_summary$statistic, 4),
  p_value = format(coefficients_summary$p.value, scientific = TRUE)
)

summary_table <- plot_ly(
  type = 'table',
  header = list(
    values = c("<b>Metric</b>", "<b>Value</b>"),
    fill = list(color = "lightgray"),
    align = "center"
  ),
  cells = list(
    values = rbind(summary_df$Metric, summary_df$Value),
    fill = list(color = "white"),
    align = "center"
  )
)

coefficients_table <- plot_ly(
  type = 'table',
  header = list(
    values = c("<b>Predictor</b>", "<b>Estimate</b>", "<b>Std. Error</b>", "<b>t-value</b>", "<b>p-value</b>"),
    fill = list(color = "lightgray"),
    align = "center"
  ),
  cells = list(
    values = rbind(
      coefficients_df$Predictor,
      coefficients_df$Estimate,
      coefficients_df$Std_Error,
      coefficients_df$t_value,
      coefficients_df$p_value
    ),
    fill = list(color = "white"),
    align = "center"
  )
)

subplot(summary_table, coefficients_table, nrows = 2, margin = 0.1, titleX = TRUE)


print("Step 6: Removing highly collinear predictors (VIF > 5) and refitting the model...")

# Compute original VIF values before removing variables
original_vif <- vif(mlr_model)

# Print VIF results
print("Initial VIF values calculated. Checking for high multicollinearity...")

# Drop high-VIF predictors
high_vif_predictors <- c("state", "latitude", "longitude")
df_cleaned <- df_cleaned[, !names(df_cleaned) %in% high_vif_predictors]

# Refit the multiple linear regression model
mlr_model <- lm(price ~ ., data = df_cleaned)

print("Model refitted after removing high-VIF predictors.")

# Recheck VIF after removing problematic variables
updated_vif <- vif(mlr_model)
print("Updated VIF values calculated. Proceeding with model summary...")

# Extract model summary statistics using `broom`
model_summary <- glance(mlr_model)

# Convert summary statistics to a data frame for Plotly
summary_df <- data.frame(
  Metric = c("R-squared", "Adjusted R-squared", "F-statistic", "P-value"),
  Value = c(
    round(model_summary$r.squared, 4),
    round(model_summary$adj.r.squared, 4),
    round(model_summary$statistic, 4),
    format(model_summary$p.value, scientific = TRUE)
  )
)

summary_table <- plot_ly(
  type = 'table',
  header = list(
    values = c("<b>Metric</b>", "<b>Value</b>"),
    fill = list(color = "lightgray"),
    align = "center"
  ),
  cells = list(
    values = rbind(summary_df$Metric, summary_df$Value),
    fill = list(color = "white"),
    align = "center"
  )
)

summary_table


# Binomial Logistic Regression

# Load necessary packages
library(dplyr)     # For data manipulation
library(car)       # For VIF calculation
library(ggplot2)   # For visualizations
library(plotly)    # For interactive tables
library(caret)     # For confusionMatrix()

# Step 1: Convert price into a binary variable
df_cleaned$binary_price <- ifelse(df_cleaned$price > mean(df_cleaned$price, na.rm = TRUE), 1, 0)

print("Binary variable 'binary_price' created based on the mean price.")

# Step 2: Fit a binomial logistic regression model
logit_model <- glm(binary_price ~ bathrooms + bedrooms + square_feet, 
                   data = df_cleaned, 
                   family = binomial)

print("Binomial logistic regression model fitted successfully.")

# Extract model summary statistic
logit_summary <- broom::tidy(logit_model)

# Convert summary statistics to a data frame for Plotly
summary_df <- data.frame(
  Predictor = logit_summary$term,
  Estimate = round(logit_summary$estimate, 4),
  Std_Error = round(logit_summary$std.error, 4),
  Z_Value = round(logit_summary$statistic, 4),
  P_Value = format(logit_summary$p.value, scientific = TRUE)
)

# Create an interactive Plotly table
logit_table <- plot_ly(
  type = 'table',
  header = list(
    values = c("<b>Predictor</b>", "<b>Estimate</b>", "<b>Std. Error</b>", "<b>Z-Value</b>", "<b>P-Value</b>"),
    fill = list(color = "lightgray"),
    align = "center"
  ),
  cells = list(
    values = rbind(summary_df$Predictor, summary_df$Estimate, summary_df$Std_Error, summary_df$Z_Value, summary_df$P_Value),
    fill = list(color = "white"),
    align = "center"
  )
)

# Display the logistic regression summary table
logit_table

# Step 3: Generate probability predictions
df_cleaned$predicted_prob <- predict(logit_model, type = "response")

# Convert probabilities to binary classification using 0.5 as the threshold
df_cleaned$predicted_class <- ifelse(df_cleaned$predicted_prob > 0.5, 1, 0)

print("Predictions generated successfully. Now evaluating model performance.")

# Step 4: Create the confusion matrix
conf_matrix <- table(Predicted = df_cleaned$predicted_class, Actual = df_cleaned$binary_price)
cm_results <- confusionMatrix(conf_matrix)  # ✅ Fix applied here

print("Confusion matrix generated.")

# Convert matrix to dataframe for heatmap
cm_df <- as.data.frame(as.table(conf_matrix))

# Extract unique labels
actual_classes <- unique(cm_df$Actual)
predicted_classes <- unique(cm_df$Predicted)

# Create a Plotly heatmap with annotations
conf_matrix_heatmap <- plot_ly(
  x = actual_classes, 
  y = predicted_classes, 
  z = matrix(cm_df$Freq, nrow = length(predicted_classes), byrow = TRUE),
  type = "heatmap",
  colorscale = "Blues",
  text = cm_df$Freq,  # Add text labels
  texttemplate = "%{text}",  # Format text display
  showscale = TRUE
) %>%
  layout(
    title = "Confusion Matrix Heatmap",
    xaxis = list(title = "Actual Class"),
    yaxis = list(title = "Predicted Class"),
    annotations = lapply(1:nrow(cm_df), function(i) {
      list(
        x = cm_df$Actual[i],
        y = cm_df$Predicted[i],
        text = cm_df$Freq[i],
        showarrow = FALSE,
        font = list(color = "black", size = 14)
      )
    })
  )

# Print confirmation message
print("Confusion matrix heatmap with numbers generated.")

# Display the heatmap
conf_matrix_heatmap

# Extract classification report details
classification_report <- data.frame(
  Metric = c("Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1 Score"),
  Value = c(
    round(cm_results$overall["Accuracy"], 4),
    round(cm_results$byClass["Sensitivity"], 4),
    round(cm_results$byClass["Specificity"], 4),
    round(cm_results$byClass["Precision"], 4),
    round(cm_results$byClass["F1"], 4)
  )
)

# Create a Plotly table for classification report
classification_table <- plot_ly(
  type = 'table',
  header = list(
    values = c("<b>Metric</b>", "<b>Value</b>"),
    fill = list(color = "lightgray"),
    align = "center"
  ),
  cells = list(
    values = rbind(classification_report$Metric, classification_report$Value),
    fill = list(color = "white"),
    align = "center"
  )
)

# Print confirmation message
print("Classification report table generated.")

# Display the table
classification_table
