dataset = read.csv('Salary_Data.csv')

#taking care of missing data

#Splitting the dataset into the Training set test set 
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting Simple Linear Regression to the training set 
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

#predicting the test set  results
y_pred = predict(regressor, newdata = test_set)

#Visulisaing the traning set results
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x= training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_point(aes(x= training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
             colour = 'blue') +
  geom_line(aes(x= training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
             colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')
  
#Visulisaing the test set results
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x= test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_point(aes(x= training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
             colour = 'blue') +
  geom_line(aes(x= training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (test set)') +
  xlab('Years of experience') +
  ylab('Salary')

 