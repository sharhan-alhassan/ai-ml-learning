
# Machine Learning
- Devided into two main Areas

## 1. Supervised Learning
- The machine learning model/algorithm is provided with a `labeled` dataset (input and output)
- With labeled inputs/outputs as dataset, the model learns from initial data and know the correct results
- It then applies the principles/rules/methods on new/other data with to apply same processes

### Types of Supervised Learning
- 1. Regression: A learning model that gives a continous linear value such as price, average price to sell a product, distance to be covered by child, what time it'll rain in a windy weather, etc
- Example of Algorithms: Linear Regression and Logistic regression

- 2. Classification: A learning model that gives a discrete outsome such as, an email is spam or not spam, a custoemr will pay back loarn or not. Output is usually True/Fale, Yes/No, 1 or 2, binary
- Examples of Models/Algorithms are: `Linear classifiers, Vector Machines, or SPMs, decision trees, random forests `

## 2. Unsupervised Learning
- The model is provided with `unlabeled` dataset and the model will find patterns categorized this data
- Note that, `Unsupervised Learning Models` DO NOT make `predictions`, they group `similar datasets` together  

### Types of Unsupervised learning
- 1. `Clustering`: Grouping like terms of data. E.g grouping people by age, country, religion, etc

- 2. `Association`: Where the Algorithm looks for relationships between variables in the data. Eg; Which items bought together in cart. Hence, the idea that "customers who buy sponge also buy soap". That's associate data points together by associating them 

- 3. `Dimensional Reduction`: The algorithm reduces the number of variables in the data while preserving as much of the information as possible. This usually happens in the preprocessing at the data stage. Eg; using autoencoders to remove noice from visual images to improve picture quality

## Comparison: Supervised Models vs Unsupervised Models
- Usually, `Supervised Models` are largely uses because they're more accurate 

- Two major advantages of `Unsupervised Model` over supervised models are:

```sh
# 1. Unsupervised models are used in training unlabeled data, which mimics real-world scenarios
# 2. Can be used to find hidden patters in data that supervised learning modesl would otherwise never find
```

## 3. Semi-Supervised Learning
- A combination of the two learning models to leverage both advantages of these two models


## Variables
- A variable is considered dependent if it depends on an independent variable

- In mathematics, The most common symbol for the input is `x`, and the most common symbol for the output is `y`; the function itself is commonly written `y = f(x)`

- It is possible to have multiple independent variables or multiple dependent variables. 

- For instance, in `multivariable calculus`, one often encounters functions of the form `z = f(x,y)`, where `z` is a dependent variable and `x` and `y` are independent variables.[8] 

- Functions with `multiple outputs` are often referred to as `vector-valued functions`, `VVF`
```sh
                        Antonym pairs
INDEPENDENT	                                    DEPENDENT
    (x)                                             (y)
input	                                        output
regressor	                                    regressand
predictor	                                    predicted
explanatory	                                    explained
exogenous	                                    endogenous
manipulated	                                    measured
exposure	                                    outcome
feature or pattern recorgnition	                label or target
```

## Feature Selection
- The `input variables` that we give to our machine learning models are called `features`. Each `column` in our dataset constitutes a feature

- Example : Select the right features in determining how old a car is
```sh
                                    COLUMS
Model                   Year Made                   Mileage                    Owner  (should be dropped)

# Owner here can be eliminated as "Noise" because of lack of proper correlation with the other features
```

- To train an optimal model, we need to make sure that we use only the essential features. If we have too many features, the model can capture the unimportant patterns and learn from noise.

- A specific variant of `ACO` called `MLACO` is designed as a fil ter method for FS in ML data. `MLACO` leverages the principles of ACO to `evaluate the relevance and importance of features based on their performance in the classification task`

## CIFAR10 DATA SET
1. `CIFAR10` -- Canadian Institute For Advanced Research is a collection of images that are commonly used to train machine learning and computer vision algorithms. 

2. It is one of the most widely used datasets for machine learning research. The `CIFAR-10 `dataset contains `60,000 32x32 color images in 10 different classes`

3. The `10 different classes` represent:

```sh
airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. 
```

4. There are `6,000 images` of each class

## HYPERPARAMETERS AND PARAMETERS
1. `PARAMETER`
- A model parameter is a variable whose value is estimated from the dataset. Parameters are the values learned during training from the historical data sets.

- The values of model parameters are not set manually. They are estimated from the training data. These are variables, that are internal to the machine learning model.

- In case a model has a fixed number of parameters, the system is called `“parametric.”` On the other hand, `“non-parametric”` systems do not have a fixed number of parameters.


2. `HYPERPARAMETER`
- A `hyperparameter` is a configuration variable that is external to the model. 

- It is defined manually before the training of the model with the historical dataset. Its value cannot be evaluated from the datasets.

- It is not possible to know the best value of the hyperparameter. But we can use rules of thumb or select a value with trial and error for our system.

- `Hyperparameters` affect the `speed` and `accuracy` of the learning process of the model. Different systems need different numbers of hyper-parameters. Simple systems might not need any hyperparameters at all.

- Examples of hyper-parameters:
```sh
1. Learning rate, 
2. the value of K in k-nearest neighbors,
3. batch-size
```

- Example of Real world Example 
```sh
# 1. Learning how to drive with the help of an Instructor
# 2. Instructor facilitates and speed up you learning how to drive quicker
# 3. Once you mastered driving, you'll no longer need the Instructor to drive on the road by yourself
# 4. Instructor ----> Hyperparameter
# 5. You (Student) ----> Parameter
```

### Important Notes on Hyperparameters and Parameters
- The values of hyper-parameters are pre-set. The values of hyper-parameters are independent of the dataset. These values do not change during training.

- Hyper-parameter is not a part of the trained or the final model. The values of model parameters estimated during training are saved with the trained model.

- `Hyper-parameters are external configuration variables, whereas model parameters are internal to the system.`

## Hyperparameters fo ✅ Deep learning
- number of hidden layers
- units per layer
- loss
- optimizer
- Activation
- learning_rate
- dropout rate
- epochs
- batch_size
- early_stop_patience

### 
```sh
#########################################################################################
    Now, don’t get parameters confused with variables. Think of it his way:
    PARAMETERS are passed
    VARIABLES are defined 
##########################################################################################

```

## CLASSIFIERS and MODELS
1. A `CLASSIFIER` is like a techer who teaches/helps the machine understand how to identify different things base on a set of rules or instructions. 

- Example is a Fruit Identifier, the Classifier(Teacher) says, when you see an item that is "red, round and has a small stem, that's most likely an Apple"

2. A `MODEL` is like the machine memory that stores all the different information/patterns/rules. 
- When you show it an item, it makes use of the past data that is has learned and base on that predict what the item might be

## Example of a CLASSIFIER approach: Simple Example (Fruit Classifier)
1. Collect Data:
    - We gather pictures of different fruits: apples, bananas, and oranges.

2. Train the Classifier:
    - We show the machine many pictures of apples, bananas, and oranges and tell it the name of each fruit.

3. Build the Model:
    - The machine learns from these examples and builds a model that remembers the characteristics of each fruit.

4. Make Predictions:
    - Now, when we show the machine a new picture of a fruit, it uses the model to predict what kind of fruit it is.

```sh
Data ---> Machine(Classifier) ---> Model ---> Predictions (Machine use Model to Predict)
```

## TRAINING AND TESTING DATA
- Imagine you have a big bag of fruit. 

- You want to use some of the fruit to teach the machine what they are `(training),` and then you want to keep some fruit aside to test if the machine learned correctly `(testing).`

### Variables:
```sh
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

1. X_train and X_test:

- X_train: These are the fruits we use to teach the machine. It includes features of the fruits like their weight and color.
- X_test: These are the fruits we use to test the machine later. It also includes features of the fruits like their weight and color.

2. y_train and y_test:
- y_train: These are the labels (names) of the fruits we use to teach the machine. For example, apple, banana, orange.
- y_test: These are the labels (names) of the fruits we use to test the machine later.
```

### X and Y: Explanation of X and y
```sh
X: Represents the features of the fruits, like their weight and color. Think of it as the description of the fruits.

y: Represents the labels, or names, of the fruits.
```

### Parameters
```sh
test_size=0.3:

- This means we are keeping 30% of the fruits for testing and using 70% of the fruits for training.
- If you have 10 fruits, you would use 7 fruits to teach the machine and 3 fruits to test the machine.


random_state=42:
- This is like setting a specific way to pick which fruits go into the training group and which go into the testing group.
- By using the number 42, we ensure that every time we run the code, the fruits are split in the same way. This helps us get consistent results.


classifier = KNeighborsClassifier(n_neighbors=3)
- This is our fruit-guessing friend (the machine) who learns to recognize fruits.

n_neighbors=3:

This tells our friend (the classifier) to look at the three nearest fruits when trying to guess a new fruit.
If you show the machine a new fruit, it will find the three fruits that are most similar and use their labels to make a prediction.
```

### Simple Explanation of Classifier: KNeighborsClassifier
```sh
Imagine we have a friendly neighbor who is really good at recognizing fruits. When we show them a new fruit, they look at the three closest fruits they know well and use that information to guess what the new fruit is. This is how the KNeighborsClassifier works.
```

## Multi-Class vs Multi-Label Classification Problems
### Multi-Class Problem
- A Problem can be assigned an Instance of many classes at onces and not more that onces
- Eg; A movie can be rated `A(Adults)`, `U(Unrestricted)`, or `A/U(Adults with Parental Guidance)`
- However, a movie can't rated more with more than one instanc of (A, U and AU)

### Multi-Lable Classification
- A Problem can be assigned with one or more Instances of a Label
- A movie in both `Comedy`, `Action` and `Thriller` Genre
## Performance Metrics in MLKNN(Multi-Label K-Nearest Neighbors)
- Hamming Loss (HL)
- Ranking Loss (RL)
- Average Precision (AV)
- One-Error (OE)
- Coverage ()


## Data Comprehension of Data set

```py
# 1. load and output data
from skmultilearn.dataset import load_dataset
X, y, _, _ = load_dataset('emotions', 'train')

# Output
emotions:train - exists, not redownloading

# 2. Print X and y
X, y

# Ouput
(<391x72 sparse matrix of type '<class 'numpy.float64'>'
 	with 28059 stored elements in List of Lists format>,
 <391x6 sparse matrix of type '<class 'numpy.int64'>'
 	with 709 stored elements in List of Lists format>)
```

1. Taking the "X" dataset for instiance
- `391x72`: Meaning `391` rows(instances) of the data with `72` columns each of the instance that represent each of the features

- For example:

```sh
(0, 0)    0.034741
(0, 1)    0.089665
(0, 2)    0.091225

(0, 0) 0.034741: The value at row 0, column 0 is 0.034741.
(0, 1) 0.089665: The value at row 0, column 1 is 0.089665.
(0, 3) -73.302422: The value at row 0, column 3 is -73.302422.

...                         All the way to the 72nd column of row 0
(0, 69)	0.245457
(0, 70)	0.105065
(0, 71)	0.405399
(1, 0)	0.081374            Then it jumps to row 1 (or second instance) 
(1, 1)	0.272747
(1, 2)	0.085733
```


- IN summary, Each pair `(i, j)` value indicates that for sample `i (the ith row)`, feature `j (the jth column)` has the given value.

2. Taking the "y" dataset for instance
- The output:

```sh
(0, 1)  1  -> Sample 0 has label 1
(0, 2)  1  -> Sample 0 has label 2
(1, 0)  1  -> Sample 1 has label 0
(1, 5)  1  -> Sample 1 has label 5
(2, 1)  1  -> Sample 2 has label 1
(2, 5)  1  -> Sample 2 has label 5

## Detailed Breakdown
Sample 0:

Labels: 1, 2 (indicating that this sample is associated with emotions 1 and 2).
Sample 1:

Labels: 0, 5 (indicating that this sample is associated with emotions 0 and 5).
Sample 2:

Labels: 1, 5 (indicating that this sample is associated with emotions 1 and 5).
Sample 3:

Label: 2 (indicating that this sample is associated with emotion 2).

And so on and so forth ...
```

### Data Comprehension -- Concept 

1. Features (X)
- Each row represents an instance (e.g., an audio clip).
- Each column represents a feature extracted from that instance (e.g., pitch, tempo, rhythm).

2. Labels (y):
- Each row represents an instance.
- Each column represents an emotion label (e.g., 'happy', 'sad').
- The value is 1 if the emotion is present, and 0 if it is not.


## Sparse Matrix Format
- In the sparse_matrix, we only store the row, column, and value of non-zero elements.
- The output is given in the format `(row, column) value`, where:

```py
# Full matrix representation (mostly zeros, not efficient)
full_matrix = [
    [0, 0, 0, 0, 1],  # Only one non-zero element (1) in the last position
    [0, 0, 0, 0, 0],  # All zeros
    [0, 1, 0, 0, 0],  # One non-zero element (1) in the second position
    [0, 0, 0, 0, 0],  # All zeros
    [0, 0, 0, 0, 1],  # One non-zero element (1) in the last position
]

# Sparse matrix representation (only storing non-zero values)
sparse_matrix = [
    (0, 4, 1),  # Row 0, Column 4, Value 1
    (2, 1, 1),  # Row 2, Column 1, Value 1
    (4, 4, 1),  # Row 4, Column 4, Value 1
]
```

```sh
row: Represents the index of the sample (or instance) in the dataset.
column: Represents the index of the label.
value: The value at that position, which is 1 in this case, indicating the presence of a particular label for that sample.
```


## Feature Vector
- `Feature Vector:` A feature vector is a list of numbers that represent the characteristics (`features`) of a single instance (or example) in your dataset.

