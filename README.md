# Decision Tree Model - Machine Learning with ID3

The goal of this project is to use machine learning techniques, specifically the ID3 algorithm, to build a decision tree model. This model will classify whether a person’s yearly income is ≤ 50K or > 50K. This project is a great opportunity to explore the practical application of decision trees in machine learning and gain hands-on experience with implementing algorithms from scratch.

## Objective

A decision tree is a flowchart-like structure in which each internal node represents a feature (or attribute), each branch represents a decision rule, and each leaf node represents an outcome. The topmost node in a decision tree is known as the root node. It learns to partition on the basis of the attribute value. It partitions the tree recursively in a manner called recursive partitioning.

In this project, the ID3 (Iterative Dichotomiser 3) algorithm is used to construct the decision tree. ID3 is one of the simplest and most common algorithms for constructing decision trees. It uses Entropy and Information Gain as the statistical test to determine on what attributes to split data. The algorithm is implemented from scratch as described in the book by Kelleher et al, without using packaged libraries like Scikit-Learn. This approach provides a deeper understanding of the inner workings of the algorithm and the principles of machine learning.

## Data
The training and test data are found in the enclosed subfolder named “data”. There are 4 files:
- **census_training.csv**: This dataset (30110 training examples) is used to build the decision tree model using ID3 and Info Gain.
- **census_training_test.csv**: This dataset (15028 examples) is used to test the accuracy of the decision tree model.
- **playtennis.csv**: A small dataset for which the tree shape is known. This is used to validate that the algorithm is working correctly.
- **emails.csv**: Another small dataset for which the tree shape is known. This is also used for validation.

## Implementation Hints
- Build your code incrementally: create helper functions that you are likely to call (calculating entropy, information gain, best feature to choose for a split, etc.). 
- Code your algorithm on small datasets that you know how their resulting decision trees look like: `playtennis.csv` and `emails.csv` were provided to you for this purpose.
- When building the decision tree, save it in memory as a dictionary of dictionaries.
- After verifying that your algorithm is building the correct PlayTennis and Email trees, then you can test your algorithm on training data `census_training.csv`.
- Now use `census_training_test.csv` to test the accuracy of your Decision Tree model. Count the number of accurate classifications and the number of inaccurate classifications.
- If you represent your tree as a Python dictionary, you can use the function `draw_decision_tree_dictionary` to get a visual of the tree. It is not a requirement to get a visual of the tree. However, after spending many hours you are probably curious to see how your tree looks like. In some cases, looking at the tree can reveal the existence of bugs.

## Model Accuracy & Pruning
The decision tree model without pruning achieved an accuracy of over 80% on the test data. This is a strong performance, indicating that the model was able to learn the underlying patterns in the training data and generalize them to unseen data.

To further improve the model's performance, pre-pruning was implemented. Pre-pruning is a technique that involves stopping the tree-building process early when the current partition (node) size falls below a certain threshold. In this case, a threshold of 5 was used.

After implementing pre-pruning, there was a slight increase in the model's accuracy on the test data. This suggests that pre-pruning helped to prevent overfitting by stopping the model from learning too much from the noise in the training data, leading to better generalization on unseen data.

It's important to note that the optimal pre-pruning threshold can vary depending on the dataset and the problem at hand. Therefore, it's always a good idea to experiment with different thresholds and evaluate the model's performance on a validation set to find the best value.
