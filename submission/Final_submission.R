
library(tidyverse) # data manipulation
library(lubridate) # date conversion
library(tictoc) # to measure time for model training
library(benford.analysis) # Amount analysis
library(lightgbm)
library(MLmetrics)


train_transaction <- read_csv("train_transaction.csv")
train_identity <- read_csv("train_identity.csv")

test_transaction <- read_csv("test_transaction.csv")
test_identity <- read_csv("test_identity.csv")


num_col <- c(seq(1:12))
train_identity[,num_col] <- sapply(train_identity[,num_col], as.numeric)
test_identity[,num_col] <- sapply(test_identity[,num_col], as.numeric)

num_rows <- NROW(train_transaction)

y <- train_transaction$isFraud 
train_transaction$isFraud <- NULL

## SAVE FOR LIGHTGBM

save_TransactionDT <- train_transaction$TransactionDT

train_transaction <- train_transaction %>% bind_rows(test_transaction)
train_identity <- train_identity %>% bind_rows(test_identity)


train_transaction.save <- train_transaction
test_transaction.save <- test_transaction
y.save <- y
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

calc_smooth_mean <- function(df, by, on, m){
  
  global_mean <- mean(df[[on]])
  on <- sym(on)
  smooth <-df %>%
    group_by(.dots = by) %>%
    summarise(counts = n(),
              means = mean(!!on)) %>%
    ungroup() %>%
    mutate(smooth = (counts * means + m * global_mean) / (counts + m) ) 
  
  return(smooth)
}

# feature list
# list_fe <- read_csv("imp_24.csv")
# list_fe <- list_fe %>%
#   arrange(desc(Gain)) %>%
#   select(Feature) %>%
#   head(250) %>% 
#   unlist()




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



train_transaction$card4[which(is.na(train_transaction$card4))] <- "other"
train_transaction$card6[which(is.na(train_transaction$card6))] <- "other"
train_cards <- train_transaction %>% 
  select(c(card1:card6)) %>%
  mutate_all(~replace(., is.na(.), 0)) %>%
  mutate(cards_id = str_c(card1, card2, card3, card4, card5, card6)) 
train_transaction <- train_transaction %>% bind_cols(select(train_cards, cards_id))

test_transaction$card4[which(is.na(test_transaction$card4))] <- "other"
test_transaction$card6[which(is.na(test_transaction$card6))] <- "other"
test_cards <- test_transaction %>% 
  select(c(card1:card6)) %>%
  mutate_all(~replace(., is.na(.), 0)) %>%
  mutate(cards_id = str_c(card1, card2, card3, card4, card5, card6)) 
test_transaction <- test_transaction %>% bind_cols(select(test_cards, cards_id))

levels <- sort(unique(c(train_transaction$cards_id, test_transaction$cards_id)))
train_transaction$cards_id <- as.integer(factor(train_transaction$cards_id, levels = levels))
test_transaction$cards_id <- as.integer(factor(test_transaction$cards_id, levels = levels))




smoothenc <- train_transaction %>% 
  calc_smooth_mean("card4", "isFraud", 300) %>%
  rename(smooth_card4 = smooth)
train_transaction <- train_transaction %>% left_join(smoothenc)

test_transaction <- test_transaction %>% left_join(smoothenc)

smoothenc <- train_transaction %>% 
  calc_smooth_mean("card6", "isFraud", 300) %>%
  rename(smooth_card6 = smooth)
train_transaction <- train_transaction %>% left_join(smoothenc)

test_transaction <- test_transaction %>% left_join(smoothenc)




# card1_dist1 label encoding
train_transaction <- train_transaction %>%
  mutate(distance = ifelse(ProductCD == "W", dist1, dist2))
test_transaction <- test_transaction %>%
  mutate(distance = ifelse(ProductCD == "W", dist1, dist2))


# card1_dist1 count encoding
train_transaction <- train_transaction %>%
  group_by(card1, distance) %>%
  mutate(card1_dist_count = n()) %>%
  ungroup()
test_transaction <- test_transaction %>%
  group_by(card1, distance) %>%
  mutate(card1_dist_count = n()) %>%
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
  mutate(card1_dist_label = str_c(card1, distance))
test_transaction <- test_transaction %>%
  mutate(card1_dist_label = str_c(card1, distance))

levels <- sort(unique(c(train_transaction$card1_dist_label, test_transaction$card1_dist_label)))
train_transaction$card1_dist_label <- as.integer(factor(train_transaction$card1_dist_label, levels = levels))
test_transaction$card1_dist_label <- as.integer(factor(test_transaction$card1_dist_label, levels = levels))


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

# train_transaction <- train_transaction %>%
#   group_by(card1, card2, card3, card5) %>%
#   mutate(cards_mean_tra = mean(TransactionAmt),
#          cards_sd_tra = sd(TransactionAmt),
#          cards_diff_tra = (TransactionAmt - cards_mean_tra) / cards_sd_tra ) %>%
#   ungroup()
# 
# test_transaction <- test_transaction %>%
#   group_by(card1, card2, card3, card5) %>%
#   mutate(cards_mean_tra = mean(TransactionAmt),
#          cards_sd_tra = sd(TransactionAmt),
#          cards_diff_tra = (TransactionAmt - cards_mean_tra) / cards_sd_tra ) %>%
#   ungroup()  

train_transaction <- train_transaction %>%
  group_by(card1, card2, card3, card5) %>%
  mutate(cards_mean_tra = mean(TransactionAmt),
         cards_sd_tra = sd(TransactionAmt)) %>%
  ungroup() %>%
  mutate(cards_diff_tra = TransactionAmt / cards_mean_tra)

test_transaction <- test_transaction %>%
  group_by(card1, card2, card3, card5) %>%
  mutate(cards_mean_tra = mean(TransactionAmt),
         cards_sd_tra = sd(TransactionAmt)) %>%
  ungroup() %>%
  mutate(cards_diff_tra = TransactionAmt / cards_mean_tra)




## Add date

train_transaction <- train_transaction %>% 
  add_date()
test_transaction <- test_transaction %>% 
  add_date()

## Add hour in periodic format
conv_hour <- read_csv("periodic_hours_2.csv")
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
         week_count = n(),
         week_diff = TransactionAmt / week_mean) %>%
  ungroup()
test_transaction <- test_transaction %>%
  group_by(week) %>%
  mutate(week_mean = mean(TransactionAmt),
         week_sd = sd(TransactionAmt),
         week_count = n(),
         week_diff = TransactionAmt / week_mean) %>%
  ungroup()

train_transaction <- train_transaction %>%
  group_by(week, hour_circ) %>%
  mutate(whour_circ_mean = mean(TransactionAmt),
         whour_circ_sd = sd(TransactionAmt),
         whour_circ_diff = TransactionAmt / week_mean) %>%
  ungroup()
test_transaction <- test_transaction %>%
  group_by(week, hour_circ) %>%
  mutate(whour_circ_mean = mean(TransactionAmt),
         whour_circ_sd = sd(TransactionAmt),
         whour_circ_diff = TransactionAmt / week_mean) %>%
  ungroup()


train_transaction <- train_transaction %>%
  group_by(card1, card2, card3, card5, week) %>%
  mutate(cards_mean_week = mean(TransactionAmt),
         cards_sd_week = sd(TransactionAmt),
         cards_count_week = n()) %>%
  ungroup()
train_transaction <- train_transaction %>%
  mutate(cards_week_diff = cards_count_week / week_count,
         cards_week_diffamt = cards_mean_tra / week_mean)

test_transaction <- test_transaction %>%
  group_by(card1, card2, card3, card5, week) %>%
  mutate(cards_mean_week = mean(TransactionAmt),
         cards_sd_week = sd(TransactionAmt),
         cards_count_week = n()) %>%
  ungroup()
test_transaction <- test_transaction %>%
  mutate(cards_week_diff = cards_count_week / week_count,
         cards_week_diffamt = cards_mean_tra / week_mean)


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


decimalnumcount<-function(x){stopifnot(class(x)=="character")
  x<-gsub("(.*)(\\.)|([0]*$)","",x)
  nchar(x)
}

train_transaction$integral <- floor(train_transaction$TransactionAmt)
test_transaction$integral <- floor(test_transaction$TransactionAmt)
train_transaction$decimal <- train_transaction$TransactionAmt - train_transaction$integral
test_transaction$decimal <- test_transaction$TransactionAmt - test_transaction$integral
train_transaction$num_decimal <- decimalnumcount(as.character(train_transaction$TransactionAmt))
test_transaction$num_decimal <- decimalnumcount(as.character(test_transaction$TransactionAmt))

train_transaction <- train_transaction %>% 
  group_by(ProductCD, decimal) %>%
  mutate(prod_dec = str_c(ProductCD, as.character(decimal))) %>%
  ungroup() %>%
  mutate(prod_decw = ifelse(ProductCD == "W", prod_dec, 0))

# train_v <- train_transaction %>%
#   select(c(V1:V339))
# 
# train_v <- train_v %>% 
#   mutate_all(~replace(., is.na(.), -1))
# 
# v.pr <- prcomp(train_v, center = TRUE, scale = TRUE)
# vpcacol <- v.pr$x %>% 
#   as_tibble() %>%
#   select(VPC1 = PC1, VPC2 = PC2, VPC3 = PC3, VPC4 = PC4, VPC5 = PC5, VPC6 = PC6,
#          VPC7 = PC7, VPC8 = PC8)

# saveRDS(v.pr$x, "c:\\temp\\train_v_pca_full.RData")
# vpca <- readRDS("c:\\temp\\train_v_pca_full.RData")
# 
# vpcacol <- vpca %>% 
#   as_tibble() %>%
#   select(VPC1 = PC1, VPC2 = PC2, VPC3 = PC3, VPC4 = PC4, VPC5 = PC5, VPC6 = PC6,
#          VPC7 = PC7, VPC8 = PC8)
# 
# train_transaction <- train_transaction %>%
#   select(-c(V1:V339)) %>%
#   bind_cols(vpcacol)




train_v <- train_transaction %>%
  select(c(V1:V339))

test_v <- test_transaction %>%
  select(c(V1:V339))

prop_na_train <- sapply(lapply(train_v[,names(train_v)],is.na), function(x) {sum(x)/NROW(x)*100}) 
prop_na_test <- sapply(lapply(test_v[,names(test_v)],is.na), function(x) {sum(x)/NROW(x)*100})

#train_v <- train_v[, prop_na_train < 30]
#test_v <- test_v[, prop_na_train[names(prop_na_train) != "isFraud"] < 30]


list2char <- function(v) {
  reduce(.f=paste, .x = v, sep = "#")
}

list_50p <- t(prop_na_train) %>% 
  as_tibble() %>% 
  gather(varname, value) %>%
  group_by(value) %>% 
  summarise(val = unique(value),
            var_all = list2char(as.character(varname))) %>% 
  filter(value > 50) %>%
  select(list_var = var_all) %>% 
  unlist() %>%
  sapply(str_split, pattern = "#")


sum_na_col <- function(list_var, df){
  
  na_mx <- sapply(df[,list_var], is.na )
  col_name <- paste0(attributes(na_mx)$dimnames[[2]][[1]], "_NA")
  
  sum_na <- na_mx %>%
    as_tibble() %>%
    mutate(na = rowMeans(select(., colnames(.)))) %>%
    select(na) %>%
    rename(!!col_name := na)
  
  list_remove <- unlist(list_var)
  
  df <- df[, -which(names(df) %in% list_remove)] %>%
    bind_cols(sum_na)
  
  return(df)
}


for(list_na in list_50p){
  train_v <- sum_na_col(list_na, train_v)
  test_v <- sum_na_col(list_na, test_v)
}

rem_v <- unlist(list_50p)
col_keep <- which(!(names(train_transaction) %in% rem_v))

# train_transaction <- train_transaction %>%
#   select(-c(V1:V339)) %>%
#   bind_cols(train_v)



pca_vfe <- function(df){
  
  pcadf <- df %>%
    select(c(V12:V321))
  
  pcadf <- pcadf %>% 
    mutate_all(~replace(., is.na(.), -1))
  
  v12pca <- select(pcadf, c(V12:V125), -V107)
  v12.pr <- prcomp(v12pca, center = TRUE, scale = TRUE)
  colv12 <- v12.pr$x %>% 
    as_tibble() %>%
    select(V1_PC1 = PC1, V1_PC2 = PC2, V1_PC3 = PC3, V1_PC4 = PC4)
  df <- df %>% bind_cols(colv12)
  
  
  v126pca <- select(pcadf, c(V126:V137))
  v126.pr <- prcomp(v126pca, center = TRUE, scale = TRUE)
  colv126 <- v126.pr$x %>% 
    as_tibble() %>%
    select(V126_PC1 = PC1, V126_PC2 = PC2, V126_PC3 = PC3, V126_PC4 = PC4)
  df <- df %>% bind_cols(colv126)
  
  
  v279pca <- select(pcadf, c(V279:V305))
  v279.pr <- prcomp(v279pca, center = TRUE, scale = TRUE)
  colv279 <- v279.pr$x %>% 
    as_tibble() %>%
    select(V279_PC1 = PC1, V279_PC2 = PC2, V279_PC3 = PC3, V279_PC4 = PC4)
  df <- df %>% bind_cols(colv279)
  
  
  v306pca <- select(pcadf, c(V306:V321))
  v306.pr <- prcomp(v306pca, center = TRUE, scale = TRUE)
  colv306 <- v306.pr$x %>% 
    as_tibble() %>%
    select(V306_PC1 = PC1, V306_PC2 = PC2, V306_PC3 = PC3, V306_PC4 = PC4)
  df <- df %>% bind_cols(colv306)
  

  return(df)
}

train_transaction <- train_transaction %>% pca_vfe()

Vcol_keep <- train_transaction %>%
  select(c(V1:V7), V258, V294)


train_transaction <- train_transaction %>%
  select(-c(V1:V339)) %>%
  bind_cols(Vcol_keep)
test_transaction <- test_transaction %>%
  select(-c(V1:V339)) %>%
  bind_cols(Vcol_keep)



train_transaction <- train_transaction %>%
  mutate(C1_log = log1p(C1),
         C2_log = log1p(C2),
         C13_log = log1p(C13))
test_transaction <- test_transaction %>%
  mutate(C1_log = log1p(C1),
         C2_log = log1p(C2),
         C13_log = log1p(C13))


## Remove variables

remove_transaction <-c("TransactionDT", "date", "week", "hour")
train_transaction <- train_transaction %>% select(-remove_transaction)
test_transaction <- test_transaction %>% select(-remove_transaction)


# Join dataframes
train_transaction <- train_transaction %>% 
  left_join(train_identity) %>%
  select(-TransactionID)

test_transaction <- test_transaction %>% 
  left_join(test_identity) %>%
  select(-TransactionID)


train_transaction.save <- train_transaction
test_transaction.save <- test_transaction
# train_transaction <- train_transaction.save
# test_transaction <- test_transaction.save

## ID NA

list_id <- c("id_21", "id_22", "id_23", "id_24", "id_25", "id_27")

id_na <- is.na(train_transaction[, names(train_transaction) %in% list_id])
id_na <- as_tibble(id_na) %>%
   mutate(id_na = rowMeans(select(., colnames(.))))

train_transaction <- train_transaction[, !(names(train_transaction) %in% list_id)] %>%
  bind_cols(select(id_na, id_na))

## M NA

list_M <- c("M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9")

M_na <- is.na(train_transaction[, names(train_transaction) %in% list_M])
M_na <- as_tibble(M_na) %>%
  mutate(M_na = rowMeans(select(., colnames(.))))

train_transaction <- train_transaction[, !(names(train_transaction) %in% list_M)] %>%
  bind_cols(select(M_na, M_na))

# 
# list_fe <- c(list_fe, "sumna", "card1_addr1_count", "card1_addr1_label", "card1_dist1_label", "R_emaildomain_enc",
#              "P_emaildomain_enc", "domain", "ext", "card_addr_label", "cards_label", "cards_mean_tra",
#              "cards_sd_tra", "cards_diff_tra", "week_mean", "week_sd", "week_diff", "whour_circ_mean",
#              "whour_circ_sd", "whour_circ_diff", "ben1", "ben2")
# 
# test_transaction <- test_transaction[, which(names(test_transaction) %in% list_fe)]
# #list_fe <- c(list_fe, "isFraud")
# train_transaction <- train_transaction[, which(names(train_transaction) %in% list_fe)]





levels <- sort(unique(c(train_transaction$card1, test_transaction$card1)))
train_transaction$card1 <- as.integer(factor(train_transaction$card1, levels = levels))
test_transaction$card1 <- as.integer(factor(test_transaction$card1, levels = levels))

levels <- sort(unique(c(train_transaction$card2, test_transaction$card2)))
train_transaction$card2 <- as.integer(factor(train_transaction$card2, levels = levels))
test_transaction$card2 <- as.integer(factor(test_transaction$card2, levels = levels))

levels <- sort(unique(c(train_transaction$card1, test_transaction$card1)))
train_transaction$card3 <- as.integer(factor(train_transaction$card3, levels = levels))
test_transaction$card3 <- as.integer(factor(test_transaction$card3, levels = levels))

levels <- sort(unique(c(train_transaction$card5, test_transaction$card5)))
train_transaction$card5 <- as.integer(factor(train_transaction$card5, levels = levels))
test_transaction$card5 <- as.integer(factor(test_transaction$card5, levels = levels))

categorical <- names(train_transaction)[sapply(train_transaction, is.character)]

# Encode isFraud as a factor
# train_transaction$isFraud <- as.factor(as.character((train_transaction$isFraud)))


## Free memory
rm(list=setdiff(ls(), c("train_transaction", "test_transaction", "save_TransactionDT", "categorical", "num_rows", "y",
                        "train_transaction.save", "test_transaction.save", "y.save")))
gc()




######################################
## lightgbm
######################################


traindf <- train_transaction[c(1:num_rows),]
test_transaction <- train_transaction[-c(1:num_rows),]
traindf.save <- traindf

traindf$TransactionDT <- save_TransactionDT

tryCatch(
  {tr_idx <- which(traindf$TransactionDT < quantile(traindf$TransactionDT, 0.8))},
  error = function(e){print(e)}
)

#y <- as.numeric(as.character(traindf$isFraud))
traindf$isFraud <- NULL
traindf$TransactionDT <- NULL


tic()
d0 <- lgb.Dataset(as.matrix(traindf[tr_idx,]), label = y[tr_idx], free_raw_data=F, categorical_feature = categorical)
dval <- lgb.Dataset(as.matrix(traindf[-tr_idx,]), label = y[-tr_idx], free_raw_data=F, categorical_feature = categorical) 


lgb_param <- list(boosting_type = 'dart', # gbdt
                  objective = "binary" ,
                  metric = "AUC",
                  boost_from_average = TRUE,
                  learning_rate = 0.005, #0.005
                  max_depth = -1,  # remove
                  num_leaves = 197,    # 192
                  min_gain_to_split = 0,
                  feature_fraction = 0.5,   # 0.3
                  # feature_fraction_seed = 666666,
                  bagging_freq = 1,
                  bagging_fraction = 0.7,
                  # min_sum_hessian_in_leaf = 0,
                  min_data_in_leaf = 100
                  #scale_pos_weight = 5, # test
                  #lambda_l1 = 0.3,
                  #lambda_l2 = 0.1
                  )


lgb_param <- list(boosting_type = 'gbdt', # gbdt
                  objective = "binary" ,
                  metric = "AUC",
                  boost_from_average = TRUE,
                  learning_rate = 0.005, #0.005
                  max_depth = -1,  # remove
                  num_leaves = 197,    # 192
                  min_gain_to_split = 0,
                  feature_fraction = 1,   # 0.3
                  # feature_fraction_seed = 666666,
                  bagging_freq = 1,
                  bagging_fraction = 0.7,
                  # min_sum_hessian_in_leaf = 0,
                  min_data_in_leaf = 100
                  #scale_pos_weight = 5, # test
                  #lambda_l1 = 0.3,
                  #lambda_l2 = 0.1
)

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

submission <- read.csv('sample_submission.csv')
submission$isFraud <- preds
write.csv(submission, "c:\\temp\\submission_32_lgbm.csv", row.names = FALSE)

imp <- lgb.importance(lgb)
write.csv(iter, "iter_32.csv")
write.csv(imp, "imp_32.csv")

toc()


imp %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  arrange(desc(Gain)) %>%
  head(50) %>%
  ggplot(aes(Feature, Gain, fill = Feature)) +
  geom_col() +
  coord_flip()
