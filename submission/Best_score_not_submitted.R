library(tidyverse) # data manipulation
library(lubridate) # date conversion
library(tictoc) # to measure time for model training
library(benford.analysis) # Amount analysis
library(lightgbm)
library(MLmetrics)

char_columns <- cols(ProductCD = col_character(),
                     card1 = col_character(),
                     card2 = col_character(),
                     card3 = col_character(),
                     card4 = col_character(),
                     card5 = col_character(),
                     card6 = col_character(),
                     addr1 = col_character(),
                     addr2 = col_character(),
                     M1 = col_character(),
                     M2 = col_character(),
                     M3 = col_character(),
                     M4 = col_character(),
                     M5 = col_character(),
                     M6 = col_character(),
                     M7 = col_character(),
                     M8 = col_character(),
                     M9 = col_character(),
                     .default = col_guess())

train_transaction <- read_csv("../input/ieee-fraud-detection/train_transaction.csv", col_types = char_columns)
train_identity <- read_csv("../input/ieee-fraud-detection/train_identity.csv", col_types = cols(.default = col_character()))

test_transaction <- read_csv("../input/ieee-fraud-detection/test_transaction.csv", col_types = char_columns)
test_identity <- read_csv("../input/ieee-fraud-detection/test_identity.csv", col_types = cols(.default = col_character()))

num_col <- c(seq(1:12))
train_identity[,num_col] <- sapply(train_identity[,num_col], as.numeric)
test_identity[,num_col] <- sapply(test_identity[,num_col], as.numeric)


## SAVE FOR LIGHTGBM

save_TransactionDT <- train_transaction$TransactionDT

# train_transaction.save <- train_transaction
# test_transaction.save <- test_transaction
# train_transaction <- train_transaction.save
# test_transaction <- test_transaction.save

## DATE 

add_date <- function(df){
  
  train_dt <- df$TransactionDT
  
  base_date <- as.numeric(as.POSIXct("2017-12-01", format="%Y-%m-%d"))
  train_dt <- as.POSIXct(train_dt, origin="1970-01-01")
  
  train_dt <- as.POSIXct((base_date + train_dt), origin="1970-01-01")
  
  df$day <- wday(train_dt)
  df$week <- week(train_dt)
  df$hour <- hour(train_dt)
  df$date <- date(train_dt)
  
  return(df)
}


# feature list
list_fe <- read_csv("../input/ref-imp/ref_imp.csv")
list_fe <- list_fe %>%
  arrange(desc(Gain)) %>%
  select(Feature) %>%
  head(250) %>% 
  unlist()




###########################
## FEATURE ENGINEERING  ##
###########################


traintra_mx <- sapply(train_transaction, is.na)
train_transaction$sumna <- traintra_mx %>%
  as_tibble() %>%
  mutate(sumna = rowSums(.)) %>%
  select(sumna) %>% 
  pull()

testtra_mx <- sapply(test_transaction, is.na)
test_transaction$sumna <- testtra_mx %>%
  as_tibble() %>%
  mutate(sumna = rowSums(.)) %>%
  select(sumna) %>%
  pull()

rm(list=c("traintra_mx", "testtra_mx"))
gc()

# card1_addr1 count encoding
train_transaction <- train_transaction %>%
  group_by(card1, addr1) %>%
  mutate(card1_addr1_count = n()) %>%
  ungroup()
test_transaction <- test_transaction %>%
  group_by(card1, addr1) %>%
  mutate(card1_addr1_count = n()) %>%
  ungroup()

# card1_dist1 count encoding
train_transaction <- train_transaction %>%
  group_by(card1, dist1) %>%
  mutate(card1_dist1_count = n()) %>%
  ungroup()
test_transaction <- test_transaction %>%
  group_by(card1, dist1) %>%
  mutate(card1_dist1_count = n()) %>%
  ungroup()

# card1_addr1 label encoding
train_transaction <- train_transaction %>%
  mutate(card1_addr1_label = str_c(card1, addr1))
test_transaction <- test_transaction %>%
  mutate(card1_addr1_label = str_c(card1, addr1))
# Probleme : some values don't exist in test
levels <- sort(unique(c(train_transaction$card1_addr1_label, test_transaction$card1_addr1_label)))
train_transaction$card1_addr1_label <- as.integer(factor(train_transaction$card1_addr1_label, levels = levels))
test_transaction$card1_addr1_label <- as.integer(factor(test_transaction$card1_addr1_label, levels = levels))
# So we need to encode both :
# encoding all character features into ids with sort by name


# card1_dist1 label encoding
train_transaction <- train_transaction %>%
  mutate(card1_dist1_label = str_c(card1, dist1))
test_transaction <- test_transaction %>%
  mutate(card1_dist1_label = str_c(card1, dist1))

levels <- sort(unique(c(train_transaction$card1_dist1_label, test_transaction$card1_dist1_label)))
train_transaction$card1_dist1_label <- as.integer(factor(train_transaction$card1_dist1_label, levels = levels))
test_transaction$card1_dist1_label <- as.integer(factor(test_transaction$card1_dist1_label, levels = levels))


## R_emaildomain
levels <- sort(unique(c(train_transaction$R_emaildomain, test_transaction$R_emaildomain)))
train_transaction$R_emaildomain_enc <- as.integer(factor(train_transaction$R_emaildomain, levels = levels))
test_transaction$R_emaildomain_enc <- as.integer(factor(test_transaction$R_emaildomain, levels = levels))
train_transaction <- train_transaction %>% select(-R_emaildomain)
test_transaction <- test_transaction %>% select(-R_emaildomain)

# P_emaildomain
levels <- sort(unique(c(train_transaction$P_emaildomain, test_transaction$P_emaildomain)))
train_transaction$P_emaildomain_enc <- as.integer(factor(train_transaction$P_emaildomain, levels = levels))
test_transaction$P_emaildomain_enc <- as.integer(factor(test_transaction$P_emaildomain, levels = levels))

process_email <- function(df) {
  email <- df %>% 
    select(P_emaildomain) %>%
    separate(P_emaildomain, c("domain", "ext"), sep = "\\.", extra = "merge")
  
  df <- df %>%
    bind_cols(email) %>%
    select(-P_emaildomain)
  
  return(df)
} 

train_transaction <- train_transaction %>%
  process_email()
test_transaction <- test_transaction %>%
  process_email()

levels <- sort(unique(c(train_transaction$domain, test_transaction$domain)))
train_transaction$domain <- as.integer(factor(train_transaction$domain, levels = levels))
test_transaction$domain <- as.integer(factor(test_transaction$domain, levels = levels))
levels <- sort(unique(c(train_transaction$ext, test_transaction$ext)))
train_transaction$ext <- as.integer(factor(train_transaction$ext, levels = levels))
test_transaction$ext <- as.integer(factor(test_transaction$ext, levels = levels))


# card1, card2, card3, card5, addr1, addr2 label encoding
train_transaction <- train_transaction %>%
  mutate(card_addr_label = str_c(card1, card2, card3, card5, addr1, addr2))
test_transaction <- test_transaction %>%
  mutate(card_addr_label = str_c(card1, card2, card3, card5, addr1, addr2))

levels <- sort(unique(c(train_transaction$card_addr_label, test_transaction$card_addr_label)))
train_transaction$card_addr_label <- as.integer(factor(train_transaction$card_addr_label, levels = levels))
test_transaction$card_addr_label <- as.integer(factor(test_transaction$card_addr_label, levels = levels))

# card1, card2, card3, card5label encoding
train_transaction <- train_transaction %>%
  mutate(cards_label = str_c(card1, card2, card3, card5))
test_transaction <- test_transaction %>%
  mutate(cards_label = str_c(card1, card2, card3, card5))

levels <- sort(unique(c(train_transaction$cards_label, test_transaction$cards_label)))
train_transaction$cards_label <- as.integer(factor(train_transaction$cards_label, levels = levels))
test_transaction$cards_label <- as.integer(factor(test_transaction$cards_label, levels = levels))

train_transaction <- train_transaction %>%
  group_by(card1, card2, card3, card5) %>%
  mutate(cards_mean_tra = mean(TransactionAmt),
         cards_sd_tra = sd(TransactionAmt),
         cards_diff_tra = (TransactionAmt - cards_mean_tra) / cards_sd_tra ) %>%
  ungroup()

test_transaction <- test_transaction %>%
  group_by(card1, card2, card3, card5) %>%
  mutate(cards_mean_tra = mean(TransactionAmt),
         cards_sd_tra = sd(TransactionAmt),
         cards_diff_tra = (TransactionAmt - cards_mean_tra) / cards_sd_tra ) %>%
  ungroup()  

## Add date

train_transaction <- train_transaction %>% 
  add_date()
test_transaction <- test_transaction %>% 
  add_date()

## Add hour in periodic format
conv_hour <- read_csv("../input/periodic-hours-2/periodic_hours_2.csv")
train_transaction <- train_transaction %>% 
  left_join(conv_hour) %>%
  select(-upper) %>%
  rename(hour_circ = value)
test_transaction <- test_transaction %>% 
  left_join(conv_hour) %>%
  select(-upper) %>%
  rename(hour_circ = value)

train_transaction <- train_transaction %>%
  group_by(week) %>%
  mutate(week_mean = mean(TransactionAmt),
         week_sd = sd(TransactionAmt),
         week_diff = TransactionAmt - week_mean) %>%
  ungroup()
test_transaction <- test_transaction %>%
  group_by(week) %>%
  mutate(week_mean = mean(TransactionAmt),
         week_sd = sd(TransactionAmt),
         week_diff = TransactionAmt - week_mean) %>%
  ungroup()

train_transaction <- train_transaction %>%
  group_by(week, hour_circ) %>%
  mutate(whour_circ_mean = mean(TransactionAmt),
         whour_circ_sd = sd(TransactionAmt),
         whour_circ_diff = TransactionAmt - week_mean) %>%
  ungroup()
test_transaction <- test_transaction %>%
  group_by(week, hour_circ) %>%
  mutate(whour_circ_mean = mean(TransactionAmt),
         whour_circ_sd = sd(TransactionAmt),
         whour_circ_diff = TransactionAmt - week_mean) %>%
  ungroup()

add_benford <- function(df) {
  
  amount.ben <- df %>% select(TransactionAmt)
  amount.ben.1 <- benford(as_vector(amount.ben), number.of.digits = 1)
  amount.ben.2 <- benford(as_vector(amount.ben), number.of.digits = 2)
  df$ben1 <- amount.ben.1$data$data.digits
  df$ben2 <- amount.ben.2$data$data.digits
  
  return(df)
}

train_transaction <- train_transaction %>% add_benford()
test_transaction <- test_transaction %>% add_benford()

## Remove variables

remove_transaction <-c("TransactionDT", "date", "week", "hour", "hour_circ")
train_transaction <- train_transaction %>% select(-remove_transaction)
test_transaction <- test_transaction %>% select(-remove_transaction)


# Join dataframes
train_transaction <- train_transaction %>% 
  left_join(train_identity) %>%
  select(-TransactionID)

test_transaction <- test_transaction %>% 
  left_join(test_identity) %>%
  select(-TransactionID)


list_fe <- c(list_fe, "sumna", "card1_addr1_count", "card1_addr1_label", "card1_dist1_label", "R_emaildomain_enc",
             "P_emaildomain_enc", "domain", "ext", "card_addr_label", "cards_label", "cards_mean_tra",
             "cards_sd_tra", "cards_diff_tra", "week_mean", "week_sd", "week_diff", "whour_circ_mean",
             "whour_circ_sd", "whour_circ_diff", "ben1", "ben2")

test_transaction <- test_transaction[, which(names(test_transaction) %in% list_fe)]
list_fe <- c(list_fe, "isFraud")
train_transaction <- train_transaction[, which(names(train_transaction) %in% list_fe)]


categorical <- names(train_transaction)[sapply(train_transaction, is.character)]

# Encode isFraud as a factor
train_transaction$isFraud <- as.factor(as.character((train_transaction$isFraud)))


## Free memory
rm(list=setdiff(ls(), c("train_transaction", "test_transaction", "save_TransactionDT", "categorical")))
gc()


######################################
## lightgbm
######################################


traindf <- train_transaction
traindf.save <- traindf

traindf$TransactionDT <- save_TransactionDT

tryCatch(
  {tr_idx <- which(traindf$TransactionDT < quantile(traindf$TransactionDT, 0.7))},
  error = function(e){print(e)}
)

y <- as.numeric(as.character(traindf$isFraud))
traindf$isFraud <- NULL
traindf$TransactionDT <- NULL


tic()
d0 <- lgb.Dataset(as.matrix(traindf[tr_idx,]), label = y[tr_idx], free_raw_data=F, categorical_feature = categorical)
dval <- lgb.Dataset(as.matrix(traindf[-tr_idx,]), label = y[-tr_idx], free_raw_data=F, categorical_feature = categorical) 


lgb_param <- list(boosting_type = 'dart',
                  objective = "binary" ,
                  metric = "AUC",
                  boost_from_average = "false",
                  tree_learner  = "serial",
                  max_depth = -1,
                  learning_rate = 0.01,
                  num_leaves = 197,
                  feature_fraction = 0.3,          
                  bagging_freq = 1,
                  bagging_fraction = 0.7,
                  min_data_in_leaf = 100,
                  bagging_seed = 11,
                  max_bin = 255,
                  verbosity = -1)


#  train 
valids <- list(train = d0, valid = dval)
lgb <- lgb.train(params = lgb_param, data = d0, nrounds = 15000, categorical_feature = categorical, valids = valids, eval_freq = 200, early_stopping_rounds = 400, verbose = 1, seed = 123)

oof_pred <- predict(lgb, data.matrix(traindf[-tr_idx,]))
cat("best iter :" , lgb$best_iter, "best score :", AUC(oof_pred, y[-tr_idx]) ,"\n" )
iter <- lgb$best_iter

toc()

rm(lgb,d0,dval) ; invisible(gc())

# full 

tic()
d0 <- lgb.Dataset(data.matrix(traindf), label = y, categorical_feature = categorical)
lgb <- lgb.train(params = lgb_param, data = d0, nrounds = iter * 1.05, categorical_feature = categorical, verbose = -1, seed = 123)
preds <- predict(lgb, data.matrix(test_transaction))
toc()

submission <- read.csv('../input/ieee-fraud-detection/sample_submission.csv')
submission$isFraud <- preds
write.csv(submission, "submission_lgbm_13.csv", row.names = FALSE)

imp <- lgb.importance(lgb)
write.csv(iter, "iter.csv")
write.csv(imp, "imp.csv")

imp %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  arrange(desc(Gain)) %>%
  head(50) %>%
  ggplot(aes(Feature, Gain, fill = Feature)) +
  geom_col() +
  coord_flip()