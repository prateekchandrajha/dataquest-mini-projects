"""Kaggle First Lesson

Source: https://campus.datacamp.com/courses/kaggle-python-tutorial-on-machine-learning/"""

# Passengers that survived vs passengers that passed away
print(train["Survived"].value_counts())

# As proportions
print(train["Survived"].value_counts(normalize= True))

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True))

# Normalized female survival
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True))

"""Second Lesson"""

# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.

train["Child"][train["Age"]<18] = 1
train["Child"][train["Age"]>=18] = 0
print(train["Child"])


# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))


"""Third Lesson"""

# Create a copy of test: test_one
test_one = test

# Initialize a Survived column to 0
test_one["Survived"] = 0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one["Survived"][test_one["Sex"] == "female"] = 1
test_one["Survived"][test_one["Sex"] == "male"] = 0
print(test_one["Survived"])

"""Fourth Lesson"""

"""A decision tree automates this process for you and outputs a classification model or classifier.

Conceptually, the decision tree algorithm starts with all the data at the root node and scans all the variables for the best one to split on. Once a variable is chosen, you do the split and go down one level (or one node) and repeat. The final nodes at the bottom of the decision tree are known as terminal nodes, and the majority vote of the observations in that node determine how to predict for new observations that end up in that terminal node.

Missingness is a whole subject with and in itself, but we will use a simple imputation technique where we substitute each missing value with the median of the all present values.

train["Age"] = train["Age"].fillna(train["Age"].median()) ---> One strategy of filling the missing values

Another problem is that the Sex and Embarked variables are categorical but in a non-numeric format. Thus, we will need to assign each class a unique integer so that Python can handle the information. 

Embarked also has some missing values which you should impute witht the most common class of embarkation, which is "S"."""

# Elementary Data Cleaning is done below for reference

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"]=="female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

#Print the Sex and Embarked columns
print(train["Sex"])
print(train["Embarked"])

""" DECISION TREES """

Scikit-Learn can be used to create tree objects from the DecisionTreeClassifier class. The methods that we will use take numpy arrays as inputs and therefore we will need to create those from the DataFrame that we already have. 

# Print the train data to see the available features
print(train)

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))

""" LET'S INVESTIGATE OUR DECISION TREES A LITTLE BIT FURTHER """

"""The feature_importances_ attribute make it simple to interpret the significance of the predictors you include. 

Based on your decision tree, what variable plays the most important role in determining whether or not a passenger survived?"""

my_tree_one
Out[1]: 
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='best')
            
my_tree_one.feature_importances_
Out[3]: array([0.1269655 , 0.31274009, 0.23147703, 0.32881738])

"""Luckily, with our decision tree, we can make use of some simple functions to "generate" our answer without having to manually perform subsetting."""

# Impute the missing value with the median
test.Fare[152] = test["Fare"].median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

""" OVERFITTING AND HOW TO CONTROL IT BY HAVING SECOND LOOK AT THE DATA 

Overfitting and how to control it

When you created your first decision tree the default arguments for max_depth and min_samples_split were set to None. This means that no limit on the depth of your tree was set. That's a good thing right? Not so fast. We are likely overfitting. This means that while your model describes the training data extremely well, it doesn't generalize to new data, which is frankly the point of prediction. Just look at the Kaggle submission results for the simple model based on Gender and the complex decision tree. Which one does better? 

Maybe we can improve the overfit model by making a less complex model? 

In DecisionTreeRegressor, the depth of our model is defined by two parameters:

the max_depth parameter determines when the splitting up of the decision tree stops.
the min_samples_split parameter monitors the amount of observations in a bucket. If a certain threshold is not reached (e.g minimum 10 passengers) no further splitting can be done.

By limiting the complexity of your decision tree you will increase its generality and thus its usefulness for prediction!"""

""" Something to be Kept in Mind: You can always use train.describe() in the console to check the names of the features. """

# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))

""" FEATURE ENGINEERING 

Data Science is an art that benefits from a human element. Enter feature engineering: creatively engineering your own features by combining the different existing variables."""

# Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = 1 + train_two["SibSp"] + train_two["Parch"]

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))


"RANDOM FOREST ANALYSIS OF THE DATASET

In layman's terms, the Random Forest technique handles the overfitting problem you faced with decision trees. It grows multiple (very deep) classification trees using the training set. At the time of prediction, each tree is used to come up with a prediction and every outcome is counted as a vote. For example, if you have trained 3 trees with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as a survivor. This approach of overtraining trees, but having the majority's vote count as the actual classification decision, avoids overfitting.

Building a random forest in Python looks almost the same as building a decision tree; so we can jump right to it. There are two key differences, however. Firstly, a different class is used. And second, a new argument is necessary. Also, we need to import the necessary library from scikit-learn."""


# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

"""Remember how we looked at .feature_importances_ attribute for the decision trees? Well, you can request the same attribute from your random forest as well and interpret the relevance of the included variables. 

You might also want to compare the models in some quick and easy way. For this, we can use the .score() method. 

The .score() method takes the features data and the target vector and computes mean accuracy of your model. You can apply this method to both the forest and individual trees. Remember, this measure should be high but not extreme because that would be a sign of overfitting."""

#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print(my_forest.score(features_two,target))
































