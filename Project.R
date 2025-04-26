##########################################Package Loading#############################################
# Load necessary libraries for text analysis and machine learning
library(quanteda)  # Natural language processing and document-feature matrix (DFM) handling
library(ggrepel)  # Prevent overlapping text labels in plots
library(textclean)  # Text cleaning functions
library(tidyverse)  # Data manipulation and visualization
library(glmnet)  # Regularized regression models (Lasso, Ridge, ElasticNet)
library(sentimentr)  # Sentiment analysis
library(stm)  # Structural Topic Models for topic modeling
library(readxl)  # Read Excel files into R
library(dplyr)  # Data manipulation using the tidyverse
library(spacyr)  # NLP processing such as tokenization, named entity recognition (NER), and parsing
library(politeness)  # Politeness detection in text
library(semgram)  # Semantic motif analysis
library(lexicon)  # Lexicon-based sentiment analysis tools
library(textdata) # Load necessary libraries for financial sentiment analysis
library(ggplot2) # Load ggplot2 for accuracy visualization
library(doc2concrete)
#install.packages('text2vec')
library(text2vec)
library('readxl')



# Load custom functions if available
source("vectorFunctions.R") 
source("kendall_acc.R")  # Custom function for computing Kendall's accuracy metric
vecSmall<-readRDS("vecSmall.RDS")
load("wfFile.RData")



######################################################Financial Data Lasso and Transfer Learning####################################################
# Coefficient Plot
# Word frequency file - to reweight common words
TMEF_dfm<-function(text,
                   ngrams=1,
                   stop.words=TRUE,
                   min.prop=.01){
  # First, we check our input is correct
  if(!is.character(text)){  
    stop("Must input character vector")
  }
  drop_list=""
  #uses stop.words arugment to adjust what is dropped
  if(stop.words) drop_list=stopwords("en") 
  # quanteda pipeline
  text_data<-text %>%
    replace_contraction() %>%
    tokens(remove_numbers=TRUE,
           remove_punct = TRUE) %>%
    tokens_wordstem() %>%
    tokens_select(pattern = drop_list, 
                  selection = "remove") %>%
    tokens_ngrams(ngrams) %>%
    dfm() %>%
    dfm_trim(min_docfreq = min.prop,docfreq_type="prop")
  return(text_data)
}
set.seed(2025)

fin <-read_excel("finance.xlsx")

##Lasso Model
train_split=sample(1:nrow(fin),25000)

fin_train<-fin[train_split,]
fin_test<-fin[-train_split,]

fin_dfm_train<-TMEF_dfm(fin_train$Review, ngrams = 1:3)
fin_dfm_test<-TMEF_dfm(fin_test$Review,ngrams = 1:3, min.prop = 0) %>%
  dfm_match(colnames(fin_dfm_train))

# Train a vector classifier
lasso_fin <- glmnet::cv.glmnet(
  x = fin_dfm_train,
  y = fin_train$Rating,
  alpha = 1,  # Lasso
  lambda = exp(seq(log(0.0005), log(0.05), length = 150)),  # Try adjusting the lambda range to avoid over-penalizing
  nfolds = 10
)

plot(lasso_fin)
test_dfm_predict<-predict(lasso_fin,
                          newx = fin_dfm_test, s="lambda.min")

acc_fin <- kendall_acc(test_dfm_predict,fin_test$Rating)

acc_fin


coef_fin <- coef(lasso_fin, s = "lambda.min") %>%
  as.matrix() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score = "s1") %>%  
  filter(score != 0 & ngram != "(Intercept)")


plotDat <- coef_fin %>%
  left_join(data.frame(ngram = colnames(fin_dfm_train),
                       freq = colMeans(as.matrix(fin_dfm_train))),
            by = "ngram") %>%
  mutate_at(vars(score, freq), ~round(., 3))


ggplot(plotDat, aes(x = score, y = freq, label = ngram, color = score)) +
  scale_color_gradient2(low = "navyblue", 
                        mid = "grey", 
                        high = "forestgreen", 
                        midpoint = 0) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_point() +
  geom_label_repel(max.overlaps = 15) +  
  scale_x_continuous(limits = c(-.2, .2), 
                     breaks = seq(-.2, .2, .05)) +
  scale_y_continuous(trans = "log2",  
                     breaks = c(.01, .05, .1, .2, .5, 1, 2, 5)) +
  theme_bw() +
  labs(x = "LASSO Coefficient", y = "Feature Frequency (log scale)") +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16))




#################################Financial Lasso and Transfer Learning###############################################
TMEF_dfm<-function(text,
                    ngrams=c(1,2),
                    stop.words=FALSE,
                    min.prop= 0.001){
  # First, we check our input is correct
  if(!is.character(text)){  
    stop("Must input character vector")
  }
  drop_list=""
  #uses stop.words arugment to adjust what is dropped
  if(stop.words) drop_list=stopwords("en") 
  # quanteda pipeline
  text_data<-text %>%
    replace_contraction() %>%
    tokens(remove_numbers=TRUE,
           remove_punct = TRUE) %>%
    tokens_wordstem() %>%
    tokens_select(pattern = drop_list, 
                  selection = "remove") %>%
    tokens_ngrams(ngrams) %>%
    dfm() %>%
    dfm_trim(min_docfreq = min.prop,docfreq_type="prop")
  return(text_data)
}



set.seed(2025)

fin <-read_excel("finance.xlsx")

##Lasso Model
train_split=sample(1:nrow(fin),25000)

fin_train<-fin[train_split,]
fin_test<-fin[-train_split,]

fin_dfm_train<-TMEF_dfm(fin_train$Review)
fin_dfm_test<-TMEF_dfm(fin_test$Review,min.prop = 0) %>%
  dfm_match(colnames(fin_dfm_train))

# Train a vector classifier
lasso_fin <- glmnet::cv.glmnet(
  x = fin_dfm_train,
  y = fin_train$Rating,
  alpha = 1,  # Lasso
  lambda = exp(seq(log(0.0005), log(0.05), length = 150)),  # Try adjusting the lambda range to avoid over-penalizing
  nfolds = 10
)

plot(lasso_fin)
test_dfm_predict<-predict(lasso_fin,
                          newx = fin_dfm_test, s="lambda.min")

acc_fin <- kendall_acc(test_dfm_predict,fin_test$Rating)

acc_fin

##Other Models
# Vector
vdat_train<-vecCheck(fin_train$Review,
                     vecSmall,
                     wfFile,
                     PCAtrim=1)

vdat_test<-vecCheck(fin_test$Review,
                    vecSmall,
                    wfFile,
                    PCAtrim=1)



lasso_vec<-glmnet::cv.glmnet(x=vdat_train,
                             y=fin_train$Rating)

test_vec_predict<-predict(lasso_vec,newx = vdat_test,
                          s="lambda.min")
acc_vec <- kendall_acc(test_vec_predict,fin_test$Rating)
acc_vec

#Dictionary
loughran_words <- textdata::lexicon_loughran()

positive_dict <- loughran_words %>%
  filter(sentiment == "positive") %>%
  pull(word) %>%
  paste(collapse = " ")

ddr_sims_train <- vecSimCalc(x = fin_train$Review,
                             y = positive_dict,
                             vecfile = vecSmall,
                             wffile = wfFile,
                             PCAtrim = 1)

ddr_sims_test <- vecSimCalc(x = fin_test$Review,
                            y = positive_dict,
                            vecfile = vecSmall,
                            wffile = wfFile,
                            PCAtrim = 1)

acc_ddr <- kendall_acc(ddr_sims_test, fin_test$Rating)
print(acc_ddr)

lm_positive_dict <- dictionary(list(
  positive = loughran_words %>% filter(sentiment == "positive") %>% pull(word)
))

traditional_dict_train <- fin_train$Review %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(lm_positive_dict) %>%
  convert(to = "data.frame")

traditional_dict_test <- fin_test$Review %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(lm_positive_dict) %>%
  convert(to = "data.frame")

traditional_dict_test$positive_norm <- traditional_dict_test$positive / 
  ntoken(tokens(fin_test$Review))

acc_traditional <- kendall_acc(traditional_dict_test$positive_norm, 
                               fin_test$Rating)
print(acc_traditional)


library(politeness)
library(textdata)
library(quanteda)
library(quanteda.textmodels)
install.packages("spacyr")
spacyr::spacy_install()
library(spacyr)
spacyr::spacy_initialize()


# Step 1: Extract politeness features
politeness_train <- politeness::politeness(fin_train$Review, parser = "spacy")
politeness_test <- politeness::politeness(fin_test$Review, parser = "spacy")

# Step 2: Train a Lasso model using politeness features
lasso_politeness <- glmnet::cv.glmnet(x = as.matrix(politeness_train),
                                      y = fin_train$Rating)

# Step 3: Predict on test set
test_politeness_predict <- predict(lasso_politeness, newx = as.matrix(politeness_test), s = "lambda.min")

# Step 4: Compute Kendall's Tau accuracy for politeness model
acc_politeness <- kendall_acc(test_politeness_predict, fin_test$Rating)
print(acc_politeness)

# Step 5: Store all model accuracy results
model_accuracy_results <- list(
  Lasso = as.numeric(acc_fin),
  Vector = as.numeric(acc_vec),
  DDR_Dictionary = as.numeric(acc_ddr),
  Traditional_Dictionary = as.numeric(acc_traditional),
  Politeness_Model = as.numeric(acc_politeness)  # Add politeness model accuracy
)

# Convert accuracy results into a tidy data frame
model_accuracy_df <- bind_rows(lapply(model_accuracy_results, function(x) data.frame(Accuracy = x)), .id = "Model")

# Define a better color palette
model_colors <- c("Lasso" = "#E74C3C", 
                  "Vector" = "#3498DB", 
                  "DDR_Dictionary" = "#2ECC71", 
                  "Traditional_Dictionary" = "#9B59B6",
                  "Politeness_Model" = "#F1C40F")  # New color for politeness model

# Step 6: Plot updated accuracy comparison including politeness model
ggplot(model_accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7, width = 0.6, color = "black") +  # Wider boxes, black outline
  geom_jitter(width = 0.15, alpha = 0.6, color = "black", size = 2) +  # More visible individual points
  scale_fill_manual(values = model_colors) +  # Apply custom colors
  labs(title = "Model Performance on Finance Dataset",
       x = "Model Type",
       y = "Kendall Accuracy") +
  theme_minimal(base_size = 15) +  # Larger base font size for readability
  theme(
    legend.position = "none",  # Hide legend
    text = element_text(family = "Arial", face = "bold"),  # Use a clean font
    axis.text.x = element_text(angle = 25, vjust = 1, hjust = 1, size = 14),  # Rotate labels slightly
    axis.text.y = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18),  # Center title, bold
    panel.grid.major = element_line(color = "grey80", linetype = "dashed"),  # Soft gridlines
    panel.grid.minor = element_blank()  # Remove minor gridlines for a clean look
  )


##App Level transfer learning
app_names <- unique(fin$AppName)
accuracy_results <- data.frame(AppName = character(), Accuracy = numeric())
print(app_names)
app_train = fin %>% filter(AppName != "The Wall Street Journal.")
app_test = fin %>% filter(AppName == "The Wall Street Journal.")
length(app_test)


app_dfm_train<-TMEF_dfm(app_train$Review)
app_dfm_test<-TMEF_dfm(app_test$Review,min.prop = 0) %>%
  dfm_match(colnames(app_dfm_train))
app_dfm_test <- dfm_match(app_dfm_test, features = colnames(app_dfm_train))

train_matrix <- convert(app_dfm_train, to = "matrix")
test_matrix <- convert(app_dfm_test, to = "matrix")

cat("Training matrix dimensions:", dim(train_matrix), "\n")
cat("Test matrix dimensions:", dim(test_matrix), "\n")

# Train a vector classifier
lasso_app <- glmnet::cv.glmnet(
  x = app_dfm_train,
  y = app_train$Rating,
  alpha = 1,  # Lasso
  lambda = exp(seq(log(0.0001), log(0.05), length = 150)),  # Try adjusting the lambda range to avoid over-penalizing
  nfolds = 10
)

plot(lasso_app)


test_dfm_predict_app<-predict(lasso_app,
                              newx = app_dfm_test, s="lambda.min")

acc_app <- kendall_acc(test_dfm_predict_app,app_test$Rating)

acc_app

accuracy_results <- rbind(accuracy_results, data.frame(AppName = 'Journal', Accuracy = acc_app))


library(ggplot2)

# Given accuracy_results dataframe
accuracy_results <- data.frame(
  AppName = c("Starling", "PayPal", "Monzo", "Yahoo", "Journal"),
  Accuracy.acc = c(76.04, 71.64, 80.04, 74.18, 77.65),
  Accuracy.lower = c(75.10, 70.44, 79.19, 73.18, 76.72),
  Accuracy.upper = c(76.99, 72.85, 80.88, 75.18, 78.58)
)

# Create a box plot
ggplot(accuracy_results, aes(x = AppName, y = Accuracy.acc)) +
  geom_boxplot(aes(lower = Accuracy.lower, upper = Accuracy.upper, middle = Accuracy.acc, 
                   ymin = Accuracy.lower, ymax = Accuracy.upper),
               stat = "identity", width = 0.5, fill = "lightblue", color = "black") +
  geom_point(aes(y = Accuracy.acc), color = "red", size = 3) +  # Actual accuracy points
  theme_minimal() +
  labs(title = "Accuracy Distribution Across Apps",
       x = "App Name",
       y = "Accuracy (%)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels





##Categorial level TL
sho <- read_excel("shopping.xlsx")
dat <- read_excel("Dating.xlsx")
fod <- read_excel("fooddelivery.xlsx")
gam <- read_excel("Games.xlsx")
pro <- read_excel("Productivity Applications.xlsx")
soc <- read_excel("Social Media.xlsx")
tra <- read_excel("travel.xlsx")
ent = read_excel('Entertainment.xlsx')
mus = read_excel('Music.xlsx')


# List of other datasets
datasets <- list(music = mus, shopping = sho, dating = dat, food = fod, gaming = gam, productivity = pro, social = soc, travaling = tra, entertainment = ent)

# Function to run transfer learning on a single dataset
run_transfer_learning <- function(dataset, dataset_name) {
  cat("\nProcessing dataset:", dataset_name, "\n")
  
  # Ensure there are enough rows to sample
  train_size <- min(28000, nrow(dataset))
  
  # Randomly sample train-test split
  set.seed(2025)  # Ensure reproducibility
  sampled_indices <- sample(1:nrow(dataset), train_size)
  new_train <- dataset[sampled_indices, ]
  new_test <- dataset[-sampled_indices, ]
  
  # Convert Rating to numeric (Fixes the error)
  new_test$Rating <- as.numeric(new_test$Rating)
  
  # Convert text data to document-feature matrix (DFM)
  new_dfm_train <- TMEF_dfm(new_train$Review)
  new_dfm_test <- TMEF_dfm(new_test$Review, min.prop = 0) %>%
    dfm_match(colnames(fin_dfm_train))  # Match feature space
  
  # Predict using the pre-trained Lasso model
  test_dfm_predict <- predict(lasso_fin, newx = new_dfm_test, s = "lambda.min")
  
  # Evaluate accuracy
  acc <- kendall_acc(test_dfm_predict, new_test$Rating)
  
  # Ensure accuracy is numeric before printing
  cat("Accuracy for", dataset_name, ":", as.numeric(acc), "\n")
  
  # Return accuracy
  return(acc)
}


# Run transfer learning separately for each dataset and store results
accuracy_results <- list()

for (name in names(datasets)) {
  accuracy_results[[name]] <- run_transfer_learning(datasets[[name]], name)
}

# Print final accuracy results
cat("\nFinal Transfer Learning Accuracy Results:\n")
print(accuracy_results)

accuracy_results[["finance"]] <- acc_fin 
# Convert accuracy results into a tidy data frame
accuracy_df <- bind_rows(lapply(accuracy_results, function(x) data.frame(Accuracy = as.numeric(x))), .id = "Dataset")

# Define a more colorful palette
color_palette <- c("finance" = "red", "shopping" = "blue", "dating" = "green", "food" = "purple", 
                   "gaming" = "orange", "productivity" = "cyan", "social" = "pink", "travaling" = "yellow", "entertainment" = "brown",'music' = 'grey' )

# Plot boxplot with CI and highlight "fin" dataset
ggplot(accuracy_df, aes(x = Dataset, y = Accuracy, fill = Dataset)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6) +  # Box plot for CI
  geom_jitter(width = 0.2, alpha = 0.4, color = "black") +  # Show individual data points
  scale_fill_manual(values = color_palette) +  # Assign unique colors to each dataset
  labs(title = "Transfer Learning Accuracy Across Datasets",
       x = "Dataset",
       y = "Kendall Accuracy") +
  theme_minimal() +
  theme(
    legend.position = "right",  # Show legend for colors
    text = element_text(size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate labels
    plot.title = element_text(hjust = 0.5, face = "bold")
  )





##Game Analysis

# convert gaming dataset into a DFM
gam_dfm_test <- TMEF_dfm(gam$Review, min.prop = 0) %>%
  dfm_match(colnames(fin_dfm_train))  # Match feature space

# Convert to sparse matrix
gam_dfm_test_matrix <- as(gam_dfm_test, "dgCMatrix")

# Predict ratings using the trained Lasso model
gam$predicted_rating <- predict(lasso_fin, newx = gam_dfm_test_matrix, s = "lambda.min")

# Ensure Rating and predicted_rating are numeric
gam$Rating <- as.numeric(gam$Rating)
gam$predicted_rating <- as.numeric(gam$predicted_rating)

# Compute absolute prediction error
gam <- gam %>%
  mutate(error = abs(predicted_rating - Rating))

# Extract top 10 correctly predicted high ratings (Rating >= 4)
correct_high <- gam %>%
  filter(round(predicted_rating) == Rating & Rating >= 4) %>%
  arrange(error) %>%
  slice(1:100)

# Extract top 10 correctly predicted low ratings (Rating <= 2)
correct_low <- gam %>%
  filter(round(predicted_rating) == Rating & Rating <= 2) %>%
  arrange(error) %>%
  slice(1:100)

# Extract top 10 incorrectly predicted high ratings (Predicted >= 4, Actual <= 2)
wrong_high <- gam %>%
  filter(round(predicted_rating) >= 4 & Rating <= 2 & round(predicted_rating) != Rating) %>%
  arrange(desc(error)) %>%
  slice(1:100)

# extract top 10 incorrectly predicted low ratings (Predicted <= 2, Actual >= 4)
wrong_low <- gam %>%
  filter(round(predicted_rating) <= 2 & Rating >= 4 & round(predicted_rating) != Rating) %>%
  arrange(desc(error)) %>%
  slice(1:100)

# Print selected reviews
print("Top 10 Correctly Predicted High Ratings:")
print(correct_high$Review)

print("Top 10 Correctly Predicted Low Ratings:")
print(correct_low$Review)

print("Top 10 Incorrectly Predicted High Ratings:")
print(wrong_high$Review)

print("Top 10 Incorrectly Predicted Low Ratings:")
print(wrong_low$Review)

# Merge incorrect predictions with the training dataset
fin_train_updated <- bind_rows(fin_train, sample_n(gam, 100))


# Recreate DFMs for both train and test to ensure they align
fin_dfm_train_updated <- TMEF_dfm(fin_train_updated$Review)
fin_dfm_test_updated <- TMEF_dfm(gam$Review)

# Create a consistent feature space
full_feature_space <- colnames(fin_dfm_train_updated)

# Apply dfm_match() to enforce same features in both datasets
fin_dfm_train_updated <- dfm_match(fin_dfm_train_updated, full_feature_space)
fin_dfm_test_updated <- dfm_match(fin_dfm_test_updated, full_feature_space)

#  Convert to sparse matrices
fin_dfm_train_updated_matrix <- as(fin_dfm_train_updated, "dgCMatrix")
fin_dfm_test_updated_matrix <- as(fin_dfm_test_updated, "dgCMatrix")

# Retrain the Lasso model
lasso_fin_improved <- glmnet::cv.glmnet(
  x = fin_dfm_train_updated_matrix,
  y = fin_train_updated$Rating,
  alpha = 1,  
  lambda = exp(seq(log(0.00005), log(0.05), length = 150)),  
  nfolds = 10
)

plot(lasso_fin_improved)

# Predict using the corrected test matrix
gam$predicted_rating_improved <- predict(lasso_fin_improved, 
                                         newx = fin_dfm_test_updated_matrix, 
                                         s = "lambda.min")

# Compute Kendall accuracy for the improved model
acc_gam_improved <- kendall_acc(gam$predicted_rating_improved, gam$Rating)
print("Improved Model Accuracy on Gaming Dataset:")
print(acc_gam_improved)

library(ggplot2)
library(dplyr)
library(quanteda)
library(glmnet)

#  Store old accuracy results
old_accuracy_results <- accuracy_results  

#  New accuracy results list
new_accuracy_results <- list()

#  Function to run transfer learning using the new model
run_transfer_learning_improved <- function(dataset, dataset_name) {
  cat("\nProcessing dataset with improved model:", dataset_name, "\n")
  
  # Ensure there are enough rows to sample
  train_size <- min(28000, nrow(dataset))
  
  # Randomly sample train-test split
  set.seed(0938)  # Ensure reproducibility
  sampled_indices <- sample(1:nrow(dataset), train_size)
  new_train <- dataset[sampled_indices, ]
  new_test <- dataset[-sampled_indices, ]
  
  # Convert Rating to numeric
  new_test$Rating <- as.numeric(new_test$Rating)
  
  # Convert text data to DFM
  new_dfm_test <- TMEF_dfm(new_test$Review, min.prop = 0) %>%
    dfm_match(colnames(fin_dfm_train_updated))  # Match updated feature space
  
  # Convert to sparse matrix
  new_dfm_test_matrix <- as(new_dfm_test, "dgCMatrix")
  
  # Predict using the improved Lasso model
  test_dfm_predict <- predict(lasso_fin_improved, newx = new_dfm_test_matrix, s = "lambda.min")
  
  # Evaluate accuracy
  acc <- kendall_acc(test_dfm_predict, new_test$Rating)
  
  # Ensure accuracy is numeric before printing
  cat("Improved Accuracy for", dataset_name, ":", as.numeric(acc), "\n")
  
  # Return accuracy
  return(acc)
}

# Run transfer learning using the new improved model
for (name in names(datasets)) {
  new_accuracy_results[[name]] <- run_transfer_learning_improved(datasets[[name]], name)
}

#  Store finance accuracy using new model
new_accuracy_results[["finance"]] <- acc_gam_improved  

#  Convert old accuracy results into a tidy data frame
old_accuracy_df <- bind_rows(lapply(old_accuracy_results, function(x) data.frame(Accuracy = as.numeric(x), Model = "Old")), .id = "Dataset")

#  Convert new accuracy results into a tidy data frame
new_accuracy_df <- bind_rows(lapply(new_accuracy_results, function(x) data.frame(Accuracy = as.numeric(x), Model = "New")), .id = "Dataset")

#  Combine old and new results
accuracy_comparison_df <- bind_rows(old_accuracy_df, new_accuracy_df)

#  Define colors: Blue for Old, Red for New
color_palette <- c("Old" = "blue", "New" = "red")

#  Plot both old and new accuracies
ggplot(accuracy_comparison_df, aes(x = Dataset, y = Accuracy, fill = Model)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6) +  # Box plot for CI
  geom_jitter(width = 0.2, alpha = 0.4, color = "black") +  # Show individual data points
  scale_fill_manual(values = color_palette) +  # Assign colors to Old and New
  labs(title = "Comparison of Old vs. New Model Accuracy Across Datasets",
       x = "Dataset",
       y = "Kendall Accuracy") +
  theme_minimal() +
  theme(
    legend.position = "right",  # Show legend for model versions
    text = element_text(size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate labels
    plot.title = element_text(hjust = 0.5, face = "bold")
  )




##Add Game Features 
# Convert Rating to numeric to avoid errors
gam$Rating <- as.numeric(gam$Rating)
fin_train$Rating <- as.numeric(fin_train$Rating)

# Define different sizes of gaming data to add
game_review_sizes <- c(50, 100, 500, 1000)
accuracy_results <- data.frame(Gaming_Reviews = integer(), Accuracy = numeric())

# Loop through different sizes of gaming reviews to add
for (size in game_review_sizes) {
  
  cat("\nTraining with", size, "gaming reviews added...\n")
  
  # Merge sampled incorrect gaming reviews into training dataset
  fin_train_updated <- bind_rows(fin_train, sample_n(gam, size))
  
  #  Recreate DFMs for both train and test to ensure alignment
  fin_dfm_train_updated <- TMEF_dfm(fin_train_updated$Review)
  fin_dfm_test_updated <- TMEF_dfm(gam$Review)
  
  #  Create a consistent feature space
  full_feature_space <- colnames(fin_dfm_train_updated)
  
  # Apply dfm_match() to enforce same features in both datasets
  fin_dfm_train_updated <- dfm_match(fin_dfm_train_updated, full_feature_space)
  fin_dfm_test_updated <- dfm_match(fin_dfm_test_updated, full_feature_space)
  
  # Convert to sparse matrices
  fin_dfm_train_updated_matrix <- as(fin_dfm_train_updated, "dgCMatrix")
  fin_dfm_test_updated_matrix <- as(fin_dfm_test_updated, "dgCMatrix")
  
  #  Retrain the Lasso model
  lasso_fin_improved <- glmnet::cv.glmnet(
    x = fin_dfm_train_updated_matrix,
    y = fin_train_updated$Rating,
    alpha = 1,  
    lambda = exp(seq(log(0.00005), log(0.05), length = 150)),  
    nfolds = 10
  )
  
  #  Predict using the corrected test matrix
  gam$predicted_rating_improved <- predict(lasso_fin_improved, 
                                           newx = fin_dfm_test_updated_matrix, 
                                           s = "lambda.min")
  
  # Compute Kendall accuracy for the improved model
  acc_gam_improved <- kendall_acc(gam$predicted_rating_improved, gam$Rating)
  
  #  Store results
  accuracy_results <- rbind(accuracy_results, data.frame(Gaming_Reviews = size, Accuracy = as.numeric(acc_gam_improved)))
  
  cat("Accuracy for", size, "gaming reviews added:", as.numeric(acc_gam_improved), "\n")
}

#  Plot accuracy improvements
accuracy_first_values <- accuracy_results %>%
  group_by(Gaming_Reviews) %>%
  slice(1) %>%
  ungroup()

# Plot line chart using only the first accuracy value per review size
ggplot(accuracy_first_values, aes(x = Gaming_Reviews, y = Accuracy)) +
  geom_line(color = "blue", size = 1.2) +  # Smooth Line
  geom_point(color = "red", size = 3) +  # Points on Line
  scale_x_continuous(breaks = unique(accuracy_first_values$Gaming_Reviews), limits = c(50, max(accuracy_first_values$Gaming_Reviews))) +  
  labs(title = "Accuracy Improvement with Additional Gaming Reviews",
       x = "Number of Gaming Reviews Added",
       y = "Kendall Accuracy") +
  theme_minimal(base_size = 14) + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12)
  )






######################################################Multinomial Model Prediction####################################################



######################Data Pre-Process and Cleaning#############################
TMEF_dfm<-function(text,
                   ngrams=1,
                   stop.words=TRUE,
                   min.prop=.001){
  # First, we check our input is correct
  if(!is.character(text)){  
    stop("Must input character vector")
  }
  drop_list=""
  #uses stop.words arugment to adjust what is dropped
  if(stop.words) drop_list=stopwords("en") 
  # quanteda pipeline
  text_data<-text %>%
    replace_contraction() %>%
    tokens(remove_numbers=TRUE,
           remove_punct = TRUE) %>%
    tokens_wordstem() %>%
    tokens_select(pattern = drop_list, 
                  selection = "remove") %>%
    tokens_ngrams(ngrams) %>%
    dfm() %>%
    dfm_trim(min_docfreq = min.prop,docfreq_type="prop")
  return(text_data)
}

# List of Excel files to be read
# Only select the Excel file to be read
file_list <- c("Games.xlsx", "Music.xlsx", "shopping.xlsx", "finance.xlsx", 'travel.xlsx','fooddelivery.xlsx', 'Dating.xlsx', 'Social Media.xlsx','Entertainment.xlsx','Productivity Applications.xlsx')

# Set the category name corresponding to the file
category_names <- c(
  "Games.xlsx" = "Games",
  "Music.xlsx" = "Music",
  "shopping.xlsx" = "Shopping",
  "finance.xlsx" = "Finance",
  "fooddelivery.xlsx" = "FoodDelivery",
  "travel.xlsx" = "Travel",
  "Dating.xlsx" = "Dating",
  "Social Media.xlsx" = "Social Media",
  'Entertainment.xlsx' = 'Entertainment',
  'Productivity Applications.xlsx' = 'Productivity' 
)

# Read and merge data to ensure data type consistency
app_reviews <- bind_rows(lapply(file_list, function(file) {
  data <- read_excel(file, col_types = "text")  # Force all columns to be read as character type
  data$Category <- category_names[file]  # Add a Category Column
  return(data)
}))

# Select relevant columns
app_reviews <- app_reviews %>%
  dplyr::select(AppName, Category, Rating, Review) 

set.seed(2025)  # Ensure reproducibility

# Compute the sample size (10% of total dataset)
sample_size <- round(0.1 * nrow(app_reviews))

# Randomly select 10% of the data
app_reviews_sample <- app_reviews[sample(1:nrow(app_reviews), sample_size), ]

# Function to detect negations in text and append "_NEG" suffix
# requires spacyr and sentimentr
negation_scoper<-function(text,lemmas=T){
  text_p<-spacy_parse(text,lemma =lemmas,entity = F,pos=F)
  text_p<-text_p %>%
    mutate(negation=1*(token%in%c(lexicon::hash_valence_shifters %>%
                                    filter(y==1) %>%
                                    pull(x))),
           clause_end=1*grepl("^[,.:;!?]$",token)) %>%
    group_by(sentence_id) %>%
    mutate(clause_count=cumsum(clause_end)) %>%
    group_by(doc_id,sentence_id,clause_count) %>%
    mutate(negated=cumsum(negation)-negation) %>%
    ungroup()
  if(lemmas){
    text<-text_p %>%
      mutate(lemma_neg=ifelse(negated==0,lemma,
                              paste0(lemma,"_NEG"))) %>%
      mutate(doc_id=as.numeric(gsub("text","",doc_id))) %>% 
      arrange(doc_id) %>%
      group_by(doc_id) %>%
      summarize(text=paste(lemma_neg,collapse=" ")) %>%
      pull(text)
  } else{
    text<-text_p  %>%
      mutate(token_neg=ifelse(negated==0,token,
                              paste0(token,"_NEG"))) %>%
      mutate(doc_id=as.numeric(gsub("text","",doc_id))) %>% 
      arrange(doc_id) %>%
      mutate(token_neg=ifelse(grepl("'",token),token_neg,paste0(" ",token_neg))) %>%
      group_by(doc_id) %>%
      summarize(text=paste(token_neg,collapse="")) %>%
      pull(text)
  }
  return(text)
}

# Apply negation detection function to the review column
app_reviews_sample <- app_reviews_sample %>%
  mutate(negated_review = negation_scoper(Review, lemmas = TRUE)  # Ensure correct input type
  )

# Compute politeness score using the spacy NLP parser
app_reviews_politeness <- politeness(app_reviews_sample$Review, parser = "spacy")

# Compute additional text-based features
app_reviews_sample <- app_reviews_sample%>%
  mutate(
    negated_review_sentences = get_sentences(negated_review),  # Convert to sentences
    sentiment = sentiment_by(negated_review_sentences) %>% pull(ave_sentiment),  # Use preprocessed text
    politeness_score = apply(app_reviews_politeness, 1, mean, na.rm = TRUE),
    speech_wdct = str_count(negated_review, "[[:alpha:]]+")
  )




#################Build Train and Test data and n-grmas Matrix###################
set.seed(2025)
total_samples <- nrow(app_reviews_sample)
train_indices <- sample(1:total_samples, 0.8 * total_samples)  # 80% Training
remaining_indices <- setdiff(1:total_samples, train_indices)  # remaining 20%
test_indices <- sample(remaining_indices, 0.2* total_samples)  # 20% testing

app_reviews_train <-app_reviews_sample[train_indices,]
app_reviews_test <- app_reviews_sample[test_indices,]

# Create Document-Feature Matrix (DFM) for text-based features
# Generate n-grams (1 to 3) from negated reviews
# This converts text into numerical representations for machine learning
dfm_train <- TMEF_dfm(app_reviews_train$negated_review, ngrams = 1:3)
dfm_test <- TMEF_dfm(app_reviews_test$negated_review, ngrams = 1:3) %>%
  dfm_match(colnames(dfm_train))  # Ensure the same feature space in both datasets

# Convert DFM objects into standard matrix format
# This makes them compatible with machine learning models
dfm_train_matrix <- convert(dfm_train, to = "matrix")
dfm_test_matrix <- convert(dfm_test, to = "matrix")

# Extract additional text-based features from the training dataset
additional_features_train <- app_reviews_train %>%
  dplyr::select(sentiment, politeness_score, speech_wdct) %>%
  mutate(across(everything(), as.numeric)) %>%
  as.matrix()

# Extract additional text-based features from the testing dataset
additional_features_test <- app_reviews_test %>%
  dplyr::select(sentiment, politeness_score, speech_wdct) %>%
  mutate(across(everything(), as.numeric)) %>%
  as.matrix()


# Load the Loughran-McDonald Financial Sentiment Dictionary
loughran_words <- textdata::lexicon_loughran()

# Creating a positive and negative dictionary
lm_dict <- dictionary(list(
  positive = loughran_words %>% filter(sentiment == "positive") %>% pull(word),
  negative = loughran_words %>% filter(sentiment == "negative") %>% pull(word)
))

print(lm_dict)

# Compute dictionary matching features for the training set
dict_train <- dfm_lookup(dfm_train, dictionary = lm_dict) %>%
  convert(to = "data.frame") %>%
  dplyr::select(-doc_id)

# Compute dictionary matching features for the test set
dict_test <- dfm_lookup(dfm_test, dictionary = lm_dict) %>%
  convert(to = "data.frame") %>%
  dplyr::select(-doc_id)

# Normalization: dictionary word frequency / total number of words in the text
dict_train <- dict_train / ntoken(dfm_train)
dict_test <- dict_test / ntoken(dfm_test)

# Make sure the dictionary features are numeric and convert them to matrices
dict_train <- as.matrix(as.data.frame(lapply(dict_train, as.numeric)))
dict_test <- as.matrix(as.data.frame(lapply(dict_test, as.numeric)))

# Handle NA values to prevent them from affecting glmnet calculations
dict_train[is.na(dict_train)] <- 0
dict_test[is.na(dict_test)] <- 0


# Re-merge features
final_train_matrix_1 <- cbind(dfm_train_matrix, additional_features_train, dict_train)
final_test_matrix_1 <- cbind(dfm_test_matrix, additional_features_test, dict_test)
final_test_matrix_1 <- final_test_matrix_1[, colnames(final_train_matrix_1)]

# Make sure all columns are numeric
final_train_matrix_1 <- apply(final_train_matrix_1, 2, as.numeric)
final_test_matrix_1 <- apply(final_test_matrix_1, 2, as.numeric)


# Merge DFM + Additional Features
# Combine the Document-Feature Matrix (DFM) with additional extracted features
final_train_matrix <- cbind(dfm_train_matrix, additional_features_train)
final_test_matrix <- cbind(dfm_test_matrix, additional_features_test)

# Ensure that the test matrix aligns with the training matrix by keeping only matching columns
final_test_matrix <- final_test_matrix[, colnames(final_train_matrix)]

# Convert all elements in the matrices to numeric format to ensure compatibility with ML models
final_test_matrix <- apply(final_test_matrix, 2, as.numeric)
final_train_matrix <- apply(final_train_matrix, 2, as.numeric)



#################################Build Multinomial Model############################################
# Train multinomial classification models using LASSO (glmnet)
# Multinomial classification is computationally more expensive
app_cats <- glmnet::cv.glmnet(
  x = final_train_matrix,
  y = app_reviews_train$Category,
  family = "multinomial"
)

# Train another multinomial classification model using the raw DFM features
app_cats_1 <- glmnet::cv.glmnet(
  x = dfm_train,
  y = app_reviews_train$Category,
  family = "multinomial"
)

# Train multinomial classification model using an extended feature set (DFM + additional features)
app_cats_2 <- glmnet::cv.glmnet(
  x = final_train_matrix_1,
  y = as.factor(app_reviews_train$Category),
  family = "multinomial"
)

# Plot Model Performance
# Visualize cross-validation performance for all models
plot(app_cats)   # Plot model trained on combined feature set
plot(app_cats_1) # Plot model trained on raw DFM features
plot(app_cats_2) # Plot model trained on extended feature set

# Predict on Test Data
# Generate predictions for each model
app_predict_label <- predict(app_cats, newx = final_test_matrix, type = "class")[,1]
app_predict_label_1 <- predict(app_cats_1, newx = dfm_test_matrix, type = "class")[,1]
app_predict_label_2 <- predict(app_cats_2, newx = final_test_matrix_1, type = "class")[,1]

# Evaluate Model Accuracy
# Compute accuracy scores for each model
accuracy <- mean(app_predict_label == app_reviews_test$Category)
accuracy_1 <- mean(app_predict_label_1 == app_reviews_test$Category)
accuracy_2 <- mean(app_predict_label_2 == app_reviews_test$Category)

# Print model accuracy results between original models and features-added dfm
print(paste("Model Accuracy:", accuracy))
print(paste("Model Accuracy:", accuracy_1))
print(paste("Model Accuracy:", accuracy_2))

# Convert predicted labels and actual category labels to numeric format for consistency
app_predict_label_1 <- as.numeric(as.factor(app_predict_label_1))
app_reviews_test$Category <- as.numeric(as.factor(app_reviews_test$Category))



################################Benchmark Models Comparisons#####################

kendall_acc <- function(x, y, percentage=TRUE) {
  if (length(x) != length(y)) stop("Lengths of x and y must be equal.")
  kt <- cor(x, y, method="kendall")  # 计算Kendall秩相关系数
  kt.acc <- 0.5 + kt / 2
  kt.se <- sqrt((kt.acc * (1 - kt.acc)) / length(x))
  report <- data.frame(
    acc = kt.acc,
    lower = kt.acc - 1.96 * kt.se,
    upper = kt.acc + 1.96 * kt.se
  )
  report <- round(report, 4)
  if (percentage) report <- report * 100  # 以百分比表示
  return(report)
}

kendall_acc(app_predict_label_1,app_reviews_test$Category)

# Load necessary library for Naive Bayes classification
library(e1071)  # Naive Bayes

# Convert DFM to Matrix and Combine with Additional Features
# This ensures compatibility with the Naive Bayes classifier
dfm_train_matrix <- convert(dfm_train, to = "matrix")
dfm_test_matrix <- convert(dfm_test, to = "matrix")

# Train a Naive Bayes model using the training matrix
nb_model <- naiveBayes(x = dfm_train_matrix, y = app_reviews_train$Category)
nb_predictions <- predict(nb_model, newdata = dfm_test_matrix)
nb_accuracy <- mean(nb_predictions == app_reviews_test$Category)
print(paste("Naive Bayes Model Accuracy:", nb_accuracy))
nb_predictions <- as.numeric(as.factor(nb_predictions))
kendall_acc(nb_predictions,app_reviews_test$Category)


# Load necessary library for Random Forest classification
library(randomForest)

# Train a Random Forest model using training data
rf_model <- randomForest(x = dfm_train_matrix, y = as.factor(app_reviews_train$Category), ntree = 100)
rf_predictions <- predict(rf_model, newdata = dfm_test_matrix)
rf_accuracy <- mean(rf_predictions == app_reviews_test$Category)
rf_predictions <- as.numeric(as.factor(rf_predictions))
kendall_acc(rf_predictions,app_reviews_test$Category)
print(paste("Random Forest Model Accuracy:", rf_accuracy))


# Load necessary library for XGBoost classification
library(xgboost)

# Convert data formats
dtrain <- xgb.DMatrix(data = dfm_train_matrix, label = as.numeric(as.factor(app_reviews_train$Category)) - 1)
dtest <- xgb.DMatrix(data = dfm_test_matrix)

# Training the XGBoost model
xgb_model <- xgboost(
  data = dtrain,
  objective = "multi:softmax",
  num_class = length(unique(app_reviews_train$Category)),
  nrounds = 100
)

# predict
xgb_predictions <- predict(xgb_model, newdata = dtest)
xgb_accuracy <- mean(xgb_predictions == as.numeric(as.factor(app_reviews_test$Category)) - 1)
print(paste("XGBoost Model Accuracy:", xgb_accuracy))

xgb_predictions <- as.numeric(as.factor(xgb_predictions))
kendall_acc(xgb_predictions,app_reviews_test$Category)


accuracy_results <- data.frame(
  Model = c("Multinomial", "Naive Bayes", "Random Forest", "XGBoost"),
  Accuracy = c(
    kendall_acc(app_predict_label_1, app_reviews_test$Category)$acc,
    kendall_acc(nb_predictions, app_reviews_test$Category)$acc,
    kendall_acc(rf_predictions, app_reviews_test$Category)$acc,
    kendall_acc(xgb_predictions, app_reviews_test$Category)$acc
  ),
  Lower = c(
    kendall_acc(app_predict_label_1, app_reviews_test$Category)$lower,
    kendall_acc(nb_predictions, app_reviews_test$Category)$lower,
    kendall_acc(rf_predictions, app_reviews_test$Category)$lower,
    kendall_acc(xgb_predictions, app_reviews_test$Category)$lower
  ),
  Upper = c(
    kendall_acc(app_predict_label_1, app_reviews_test$Category)$upper,
    kendall_acc(nb_predictions, app_reviews_test$Category)$upper,
    kendall_acc(rf_predictions, app_reviews_test$Category)$upper,
    kendall_acc(xgb_predictions, app_reviews_test$Category)$upper
  )
)

# Load ggplot2 for visualization
library(ggplot2)

library(ggplot2)

ggplot(accuracy_results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.7, width = 0.6, color = "black") + 
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2, color = "black") +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison", y = "Accuracy (%)", x = "Model") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))


ggplot(app_reviews_train, aes(x = Category, y = speech_wdct, fill = Category)) +
  geom_boxplot() + theme_minimal()

ggplot(app_reviews_train, aes(x = Category, y = sentiment, fill = Category)) +
  geom_boxplot() + theme_minimal()

ggplot(app_reviews_train, aes(x = Category, y = politeness_score, fill = Category)) +
  geom_boxplot() + theme_minimal()



################################Confusion Matrix################################
# Confusion matrix - useful for evaluating multinomial classification performance
# Displays the predicted vs actual category distribution

# Generate confusion matrix comparing predicted labels to actual categories
table(app_predict_label_1, app_reviews_test$Category)

# Create a more readable version of the confusion matrix by truncating category names
table(app_predict_label_1, substr(app_reviews_test$Category, 0, 10))

# Export the confusion matrix to a CSV file for further analysis
table(app_predict_label_1, app_reviews_test$Category) %>%
  write.csv("app_table.csv")


# Predict probabilities for each class using the trained multinomial model
# Setting type="response" returns probabilities for each document across all classes
apps_predict <- predict(app_cats_1,
                        newx = dfm_test,
                        type = "response")[,,1] %>%
  round(4)  # Round probabilities to 4 decimal places

# Display the first few rows of the predicted probability matrix
head(apps_predict)

# Display the dimensions of the probability matrix
# Rows correspond to documents, and columns represent the probability for each category
dim(apps_predict)


# Predict on test data using the multinomial model
test_issue_predict <- predict(app_cats_1, newx = dfm_test_matrix)

# Select the column index with the highest predicted value for each row
test_pred_labels <- apply(test_issue_predict, 1, which.max)

# Print the predicted labels (numeric indices)
print(test_pred_labels)

# Convert numeric predictions to factor labels
test_pred_labels <- factor(test_pred_labels, levels = 1:10,
                           labels = levels(app_reviews_test$Category))

# Install and load the caret package (if not already installed)
if (!requireNamespace("caret", quietly = TRUE)) {
  install.packages("caret", dependencies = TRUE)
}
library(caret)

# Generate confusion matrix
conf_matrix <- confusionMatrix(test_pred_labels, app_reviews_test$Category)

# Print confusion matrix
print(conf_matrix)


# Convert confusion matrix to a data frame
conf_matrix_df <- as.data.frame(conf_matrix$table)

# Rename columns for clarity
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Freq")


# Convert Predicted and Actual to factors with category names
conf_matrix_df$Predicted <- factor(conf_matrix_df$Predicted, levels = 1:10, labels = category_names)
conf_matrix_df$Actual <- factor(conf_matrix_df$Actual, levels = 1:10, labels = category_names)

# Plot the confusion matrix with category names
conf_plot <- ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +  # Add frequency labels
  scale_fill_gradient(low = "blue", high = "red") +  # Color gradient
  labs(title = "Confusion Matrix", 
       x = "Actual Category", 
       y = "Predicted Category", 
       fill = "Frequency") +  # Add labels
  theme_minimal() +  # Minimal theme
  theme(axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate labels for readability
        panel.background = element_rect(fill = "white", color = NA),  # White panel background
        plot.background = element_rect(fill = "white", color = NA)
  )

ggsave("Confusion Matrix.png", dpi=200,width=15,height=10)







########################################################Topic Analysis (Focus on Games)#######################################################


################################Load and Clean the Data################################
# Load Excel file containing game reviews
Games <- read_excel('Games.xlsx')

# Load necessary libraries for data processing, sentiment analysis, and string manipulation
library(dplyr)  # Data manipulation
library(sentimentr)  # Sentiment analysis
library(stringr)  # String operations

# Function to detect negations and append "_NEG" suffix to negated words
# Requires spacyr and sentimentr for NLP processing
negation_scoper <- function(text, lemmas = TRUE) {
  text_p <- spacy_parse(text, lemma = lemmas, entity = FALSE, pos = FALSE)  # Perform NLP parsing
  text_p <- text_p %>%
    mutate(negation = 1 * (token %in% c(lexicon::hash_valence_shifters %>% filter(y == 1) %>% pull(x))),
           clause_end = 1 * grepl("^[,.:;!?]$", token)) %>%
    group_by(sentence_id) %>%
    mutate(clause_count = cumsum(clause_end)) %>%
    group_by(doc_id, sentence_id, clause_count) %>%
    mutate(negated = cumsum(negation) - negation) %>%
    ungroup()
  
  if (lemmas) {
    text <- text_p %>%
      mutate(lemma_neg = ifelse(negated == 0, lemma, paste0(lemma, "_NEG"))) %>%
      mutate(doc_id = as.numeric(gsub("text", "", doc_id))) %>%
      arrange(doc_id) %>%
      group_by(doc_id) %>%
      summarize(text = paste(lemma_neg, collapse = " ")) %>%
      pull(text)
  } else {
    text <- text_p %>%
      mutate(token_neg = ifelse(negated == 0, token, paste0(token, "_NEG"))) %>%
      mutate(doc_id = as.numeric(gsub("text", "", doc_id))) %>%
      arrange(doc_id) %>%
      mutate(token_neg = ifelse(grepl("'", token), token_neg, paste0(" ", token_neg))) %>%
      group_by(doc_id) %>%
      summarize(text = paste(token_neg, collapse = "")) %>%
      pull(text)
  }
  return(text)
}

# Process game reviews with sentiment analysis and negation detection
Games <- Games %>%
  mutate(
    desc_wdct = str_count(Review, "[[:alpha:]]+"),  # Count number of words in each review
    Review_negated = negation_scoper(Review),  # Apply negation detection to reviews
    sentiment = Review_negated %>%
      get_sentences() %>%  # Split text into sentences for better sentiment analysis
      sentiment_by() %>%
      pull(ave_sentiment)  # Compute sentiment score per review
  ) %>%
  dplyr::select(where(~ !all(is.na(.))))  # Remove columns where all values are NA


################################Build Train and Test Models and ngrams dfm################################

# Set seed for reproducibility
set.seed(2025)

# Split the dataset into training and test sets
train_split <- sample(1:nrow(Games), 8000)  # Select 8000 samples for training
Games_reviews_test <- Games[train_split, ]  # Assign selected samples to test set
Games_reviews_train <- Games[-train_split, ]  # Assign remaining samples to training set

# Function to create a Document-Feature Matrix (DFM) with n-grams and stopword removal
TMEF_dfm <- function(text,
                     ngrams = 1:3,
                     stop.words = TRUE,
                     min.prop = .001) {
  # Validate input type
  if (!is.character(text)) {
    stop("Must input character vector")
  }
  drop_list = ""
  # Remove stopwords if specified
  if (stop.words) drop_list = stopwords("en") 
  
  # Text preprocessing pipeline
  text_data <- text %>%
    replace_contraction() %>%  # Expand contractions (e.g., "can't" -> "cannot")
    tokens(remove_numbers = TRUE,
           remove_punct = TRUE) %>%
    tokens_wordstem() %>%  # Perform stemming (reduce words to their base form)
    tokens_select(pattern = drop_list, selection = "remove") %>%  # Remove stopwords
    tokens_ngrams(ngrams) %>%  # Generate n-grams (e.g., bigrams, trigrams)
    dfm() %>%
    dfm_trim(min_docfreq = min.prop, docfreq_type = "prop")  # Trim low-frequency features
  return(text_data)
}

# Create Document-Feature Matrices (DFM) for training and test sets
Games_dfm_train <- TMEF_dfm(Games_reviews_train$Review, ngrams = 1:3)
Games_dfm_test <- TMEF_dfm(Games_reviews_test$Review, ngrams = 1:3) %>%
  dfm_match(colnames(Games_dfm_train))  # Ensure test matrix uses same features as training matrix

# Identify valid indices where documents contain tokens
valid_train_indices <- which(ntoken(Games_dfm_train) > 0)
valid_test_indices <- which(ntoken(Games_dfm_test) > 0)

# Filter out empty documents from the DFM
Games_dfm_train <- dfm_subset(Games_dfm_train, ntoken(Games_dfm_train) > 0)
Games_dfm_test <- dfm_subset(Games_dfm_test, ntoken(Games_dfm_test) > 0)

# Ensure the review dataset matches the filtered DFM
Games_reviews_train <- Games_reviews_train[valid_train_indices, ]
Games_reviews_test <- Games_reviews_test[valid_test_indices, ]

# Convert rating column to numeric format
Games_reviews_train$Rating <- as.numeric(Games_reviews_train$Rating)
Games_reviews_test$Rating <- as.numeric(Games_reviews_test$Rating)

# Convert Star Rating into a Binary Classification
# Define ratings 4 and above as "high" (1), and others as "low" (0)
Games_reviews_train$Rating_binary <- ifelse(Games_reviews_train$Rating >= 4, 1, 0)
Games_reviews_test$Rating_binary <- ifelse(Games_reviews_test$Rating >= 4, 1, 0)




################################Conduct Topic Modelling################################


set.seed(2025)
# Train a Structural Topic Model (STM) with 20 topics
topic_mod_20 <- stm(Games_dfm_train, K = 20)
saveRDS(topic_mod_20, file = "topic_mod_20.RDS")  # Save model to disk
topic_mod_20 <- readRDS("topic_mod_20.RDS")  # Load model

# Extract the number of topics from the model settings
topicNum <- topic_mod_20$settings$dim$K

# Define topic names for better interpretability
topic_names <- paste0("Topic", 1:topicNum)

# Plot the topic distribution summary
plot(topic_mod_20, type = "summary", n = 7, xlim = c(0, .3), labeltype = "frex",
     topic.names = topic_names)

# Define meaningful names for the 20 extracted topics
topic_names <- c(
  "Gaming Experiences",
  "Positive Recommendations",
  "Among Us",
  "In-game Currency & Spending",
  "Social Networking",
  "Minecraft",
  "Candy Crush",
  "Friend Making",
  "Advertisement",
  "Genshin Impact",
  "Ranking & Competition",
  "Negative Sentiment",
  "Bugs & Game Fixes",
  "Login & Account Issues",
  "Emoji",
  "Pokemon Go",
  "Replay Value",
  "Star Rating",
  "Positive Feedback",
  "Pay Win Eco"
)

# Print topic names for verification
print(topic_names)

# Generate word cloud for a specific topic
library(wordcloud)
cloud(topic_mod_20, topic = 8)

# Load igraph package for network visualization
if (!requireNamespace("igraph", quietly = TRUE)) {
  install.packages("igraph")
}
library(igraph)

# Generate topic correlation network
g <- topicCorr(topic_mod_20)

# Plot the topic correlation graph
plot(g,
     vertex.label = topic_names,   # Set node labels
     vertex.size = 10,             # Adjust node size
     vertex.label.cex = 1.5,       # Set label font size
     vertex.label.color = "black",  # Label color
     edge.width = 1,               # Edge width
     main = "Topic Correlation Graph")  # Set title

# Display top words associated with each topic
labelTopics(topic_mod_20)

# Retrieve example reviews that match Topic 15
findThoughts(model = topic_mod_20, texts = Games_reviews_train$Review, topics = 15, n = 5)

# Estimate the correlation between topics and binary star ratings
stmEffects <- estimateEffect(1:topicNum ~ Rating_binary,
                             topic_mod_20,
                             meta = Games_reviews_train %>% dplyr::select(Rating_binary))

# Extract effect estimates and visualize topic correlations
bind_rows(lapply(summary(stmEffects)$tables, function(x) x[2, 1:2])) %>%
  mutate(topic = factor(topic_names, ordered = TRUE, levels = topic_names),  # Use custom topic names
         se_u = Estimate + `Std. Error`,
         se_l = Estimate - `Std. Error`) %>%
  ggplot(aes(x = topic, y = Estimate, ymin = se_l, ymax = se_u)) +
  geom_point() +
  geom_errorbar() +
  coord_flip() +
  geom_hline(yintercept = 0) +
  theme_bw() +
  labs(y = "Correlation with STARRATING", x = "Topic") +
  theme(panel.grid = element_blank(),
        axis.text = element_text(size = 16),  # Adjust axis text size
        axis.title = element_text(size = 20))  # Adjust axis title size



# Extract topic proportions for each document from the trained STM model
topic_prop_train <- topic_mod_20$theta

dim(topic_prop_train)  # Display dimensions of topic proportion matrix

# Assign column names to topic proportions based on defined topic names
colnames(topic_prop_train) <- topic_names

# Train a logistic regression model using topic proportions as features
app_model_stm <- glmnet::cv.glmnet(
  x = topic_prop_train,
  y = Games_reviews_train$Rating_binary,
  family = "binomial"
)

# Visualize cross-validation results for the STM-based model
plot(app_model_stm)

# Compute topic proportions for the test dataset using the trained STM model
topic_prop_test <- fitNewDocuments(
  topic_mod_20,
  Games_dfm_test %>%
    convert(to = "stm") %>%
    `$`(documents)
)

# Predict test set ratings using the trained STM-based model
test_stm_predict <- predict(app_model_stm,
                            newx = topic_prop_test$theta)[,1]

# Evaluate STM model performance using Kendall's accuracy
acc_stm <- kendall_acc(Games_reviews_test$Rating_binary, test_stm_predict)
acc_stm  # Display accuracy result


# Train a logistic regression model using DFM features
Games_model_dfm <- glmnet::cv.glmnet(
  x = Games_dfm_train,
  y = Games_reviews_train$Rating_binary,
  family = "binomial"
)

# Visualize cross-validation results for the DFM-based model
plot(Games_model_dfm)

# Predict test set ratings using the trained DFM-based model
Games_test_dfm_predict <- predict(Games_model_dfm,
                                  newx = Games_dfm_test)[,1]

# Evaluate DFM model performance using Kendall's accuracy
acc_dfm <- kendall_acc(Games_reviews_test$Rating_binary, Games_test_dfm_predict)
acc_dfm  # Display accuracy result


# Evaluate sentiment-based classification performance
acc_sentiment <- kendall_acc(Games_reviews_test$Rating_binary, Games_reviews_test$sentiment)
acc_sentiment  # Display accuracy result


# Evaluate word count-based classification performance
acc_wdct <- kendall_acc(Games_reviews_test$Rating_binary, Games_reviews_test$desc_wdct)
acc_wdct  # Display accuracy result


# Store accuracy results in a dataframe for comparison
accuracy_results <- data.frame(
  Model = c("STM", "DFM (glmnet)", "Sentiment", "Word Count"),
  Accuracy = c(
    acc_stm[["acc"]],
    acc_dfm[["acc"]],
    acc_sentiment[["acc"]],
    acc_wdct[["acc"]]
  ),
  Lower = c(
    acc_stm[["lower"]],
    acc_dfm[["lower"]],
    acc_sentiment[["lower"]],
    acc_wdct[["lower"]]
  ),
  Upper = c(
    acc_stm[["upper"]],
    acc_dfm[["upper"]],
    acc_sentiment[["upper"]],
    acc_wdct[["upper"]]
  )
)


# Create a bar plot comparing model accuracy
ggplot(accuracy_results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.7, width = 0.6, color = "black") +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2, color = "black") +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison", y = "Accuracy (%)", x = "Model") +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 14),  # Adjust X-axis text size
    axis.text.y = element_text(size = 14),  # Adjust Y-axis text size
    axis.title.x = element_text(size = 18),  # Increase X-axis title size
    axis.title.y = element_text(size = 18),  # Increase Y-axis title size
    plot.title = element_text(size = 20, face = "bold")  # Increase title font size
  )






