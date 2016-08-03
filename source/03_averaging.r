
# setting
dir_name <- "submission_pre"
filenames <- list.files(file.path("..", "submission", dir_name))
print(filenames)

# check result
result <- data.frame(filenames)
result$train <- as.numeric(gsub("submission_[0-9]+_s[0-9]+_train(.+)_valid(.+).csv","\\1",filenames))
result$valid <- as.numeric(gsub("submission_[0-9]+_s[0-9]+_train(.+)_valid(.+).csv","\\2",filenames))

# sort
result <- result[order(result$valid),]
print(result)

# number of input
num_model <- 20

# averaging
for(i in 1:num_model){
	target_file <- as.character(result$filename[i])
	print(target_file)
	datai <- read.csv(file.path("..", "submission", dir_name, target_file))
	if(i==1){
		data <- datai
	}else{
		data[,-1] <- data[,-1] + datai[,-1]
	}
}
data[,-1] <- data[,-1] / num_model

# display scores
train_scores <- result$train[1:num_model]
valid_scores <- result$valid[1:num_model]
cat("train: ", sprintf("%.4f +- %.4f", mean(train_scores), sd(train_scores)), "\n")
cat("valid: ", sprintf("%.4f +- %.4f", mean(valid_scores), sd(valid_scores)), "\n")

# save output
meanvalid <- mean(valid_scores)
filename <- paste0("submission_nummodel", num_model, "_mvalid", sprintf("%.4f", meanvalid), ".csv")
write.table(data, file.path(".." , "submission", filename), row.names = FALSE, sep=",", quote = FALSE)
