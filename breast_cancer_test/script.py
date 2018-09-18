# import modules for this project
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# load the breast cancer data
breast_cancer_data = load_breast_cancer()

# the dataset comes with the .data and .feature_names methods, let's use them to explore the data
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

# exploring the pre-built set some more, we can see that the first data point was tagged as being malignant
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

# prepare the data for classification
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data,
                                                                                      breast_cancer_data.target,
                                                                                      train_size=0.8, random_state=100)

# confirm the preparation was successful by confirming that the lengths of training_data and training_labels at the same size
print(len(training_data), len(training_labels))

# create the classifier model
classifier = KNeighborsClassifier(n_neighbors=3)

# load the training data to the model
classifier.fit(training_data, training_labels)

# print out how accurate the model is
print(classifier.score(validation_data, validation_labels))

"""
The classifier is pretty good when k=3, but is there a better k?

Use a for loop to cycle through the accuracies of models when k = i for range(1, 100), store the scores in an empty
list so we can plot the results.
"""
accuracies = list()
for i in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(training_data, training_labels)
    accuracies.append((classifier.score(validation_data, validation_labels)))

# create our x-axis labels
k_list = list(i for i in range(1, 100))

# graph the results
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classification Accuracy")
plt.show()