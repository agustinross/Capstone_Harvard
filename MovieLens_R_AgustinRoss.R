##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(rafalib)) install.packages("refalib", repos = "http://cran.us.r-project.org")
if(!require(ggpubr)) install.packages("ggpubr", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(dataCompareR)) install.packages("dataCompareR", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(markdown)) install.packages("markdown", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")


library(dslabs)
library(tidyverse)
library(caret)
library(data.table)
library(rafalib)
library(ggpubr)
library(knitr)  
library(dataCompareR)
library(ggplot2)
library(dplyr)
library(markdown)
library(kableExtra)
library(tidyr)
library(stringr)
library(ggthemes)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###
### Let's start with some exploratory analysis

# Fist I create a full dataframe, including all the data. This will be used ONLY for exploratory
# analysis purposes, in the modeling section we will use the split data ("edx" and "validation")
full_data <- rbind(edx,validation)

# First, it is important to know how many movies and users there are:
full_data %>%
  summarize(number_of_users = n_distinct(userId),  number_of_movies = n_distinct(movieId))

# Each movie is categorize in a genre, the list of genres are:
str_extract_all(unique(full_data$genres), "[^|]+") %>% unlist() %>% unique()

# Then, we know the users can punctuate the movies between 0,5 and 5 stars, it would be 
# interesting to know how this punctuation is distributed. For this, I will use the ggplot2
# package and show the distribution in a simple but very illustrative plot.

# NOTE: I will use the validation set to make the plot because it is possible that on some 
# computers the memory is not enough if I use the full_data set. By the law of Large Numbers 
# the conclusions obtained are the same.

validation %>% ggplot(aes(rating)) +
  geom_histogram(fill = "brown") +
  labs(title = "Distribution of ratings in a histogram",
  x = "Ratings", y = "Count", fill = element_blank()) +
  theme_economist()

# If I fuse the information of genres and rating, I will be able to reach to a very nice plot
# of the number of ratings per genre, here it is:

ggplot(validation %>% separate_rows(genres, sep = "\\|", convert = TRUE),
       aes(x = reorder(genres, genres, function(x) -length(x)))) +
       geom_bar(fill = "brown") +
       labs(title = "Distribution of ratings per genre", x = "Genre", y = "Counts") +
       coord_flip() +
       theme_economist()

# Combining the two plots showed before, I could get a very nice boxplot that shows the mean 
# rating per genre, here it is:
# NOTE: This powerful plot can't be reached with the 10 million database (it consumes too much
# RAM, and a "normal" computer could crash) so I used a small sample to do it.

sample_data <- full_data[1:(0.01*nrow(full_data)),]
ggplot(sample_data %>% separate_rows(genres, sep = "\\|", convert = TRUE), aes(genres, rating)) +
  geom_boxplot(fill = "steelblue", varwidth = TRUE) +
  labs(
    title = "Movie ratings per genre",
    x = "Genre", y = "Rating", fill = element_blank()
  ) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# Now we understand better the dataset, let's proceed to the creation of the recommendation model:
# From this point further, I will use the "edx" dataset to create the model and the "validation"
# dataset to test and to do the comparison between models.

graphics.off()
rm(full_data, sample_data)

# The methodology I decided to follow is: Create first very basic models (the ones given in the 
# "Machine Learning" course) 
# and then, in a progressive way, create more advanced ones.

## 1st Model: Only the rating's mean

ratings_mean <- mean(edx$rating)

model_1_prediction <- ratings_mean

# Let's see the RMSE:

RMSE_MODEL_1 <- RMSE(validation$rating, model_1_prediction)
RMSE_MODEL_1
rm(model_1_prediction)

# Let's store the RMSE in a dataframe. The idea is to store all the RMSE there and then we will be
# able to compare them.

ALL_RMSE <- data_frame(method = "Model Number 1", RMSE = RMSE_MODEL_1)


## 2nd Model: Let's include to the 1st model the movie's effect: 

movies_contribution<- edx %>% group_by(movieId) %>%
  summarize(Mg_contribution_of_movies = mean(rating - ratings_mean))

model_2_prediction <- ratings_mean + validation %>% left_join(movies_contribution, by='movieId') %>%
                      .$Mg_contribution_of_movies

# Let's see the RMSE:

RMSE_MODEL_2 <- RMSE(validation$rating, model_2_prediction)
RMSE_MODEL_2
rm(model_2_prediction)

# Let's add it to the general RMSE dataframe:

ALL_RMSE <- bind_rows(ALL_RMSE, data_frame(method="Model Number 2",RMSE = RMSE_MODEL_2 ))


## 3rd Model: Let's include to the 2nd model the user's effect:

user_contribution <- edx %>% left_join(movies_contribution, by='movieId') %>% group_by(userId) %>% 
      summarize(Mg_contribution_of_users = mean(rating - ratings_mean - Mg_contribution_of_movies))

model_3_prediction <- validation %>% left_join(movies_contribution, by='movieId') %>%
  left_join(user_contribution, by='userId') %>%
  mutate(M3_pred = ratings_mean + Mg_contribution_of_movies + Mg_contribution_of_users) %>%
  .$M3_pred

# Let's see the RMSE:

RMSE_MODEL_3 <- RMSE(validation$rating, model_3_prediction)
RMSE_MODEL_3
rm(model_3_prediction)

# Let's add it to the general RMSE dataframe:

ALL_RMSE <- bind_rows(ALL_RMSE, data_frame(method="Model Number 3",RMSE = RMSE_MODEL_3))


## 4th Model: I have already included: Rating's effect (1st model), Movie's effect (2nd model) and
## User's effect (3rd model).
## In this 4th model I will include the Genre, that is the only variable that I did not use already.

# Now, continuing with the method I used in the previous models, let's create the 4th model:

genres_contribution <- edx %>% left_join(movies_contribution, by = "movieId") %>%
                       left_join(user_contribution, by = "userId") %>% group_by(genres) %>%
                       summarize(Mg_contribution_of_genres = mean(rating - ratings_mean -
                                                    Mg_contribution_of_movies - Mg_contribution_of_users))


model_4_prediction <- validation %>% left_join(movies_contribution, by = "movieId") %>%
                      left_join(user_contribution, by = "userId") %>%
                      left_join(genres_contribution, by = c("genres")) %>%
                      mutate(M4_pred = ratings_mean + Mg_contribution_of_movies + Mg_contribution_of_users +
                               Mg_contribution_of_genres) %>%
                      .$M4_pred
                     

# Let's see the RMSE:

RMSE_MODEL_4 <- RMSE(validation$rating, model_4_prediction)
RMSE_MODEL_4

# Let's add it to the general RMSE dataframe:

ALL_RMSE <- bind_rows(ALL_RMSE, data_frame(method="Model Number 4",RMSE = RMSE_MODEL_4))


### I arrived to four different models, each one adds a new feature to the one before. The RMSEs
### obtain in each  model can be seen here:

ALL_RMSE

## The best model and his RMSE is the Model Number 4, as we can see here:

ALL_RMSE[which.min(ALL_RMSE$RMSE),1]

#The RMSE is approx 0.8649

print(RMSE_MODEL_4)

# The best model is the one that includes the contribution of the rating's mean, and the 
# combined effect of: movies, users and genres. 


