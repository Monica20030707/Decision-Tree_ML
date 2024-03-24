#!/usr/bin/env python
# coding: utf-8

# # Build a Decision Tree Model

# In[1]:


import pandas as pd

# Read all csv file 

training_df= pd.read_csv("data/census_training.csv")
testing_df= pd.read_csv("data/census_training_test.csv")
tennis_df= pd.read_csv("data/playtennis.csv")
email_df= pd.read_csv("data/emails.csv")


# In[2]:


from math import log2
# Testing out how to caculate the Information Gain from tennis_df and email_df

# Show header of playtennis and emails
print("Header of playtennis dataset:",tennis_df.columns)
print("Header of email dataset:",email_df.columns)

# Use Information Gain from Assignment 1 with a few modify to get call as a function

# Define entropy funtion to use on the feature that we look into
def entropy(feature_prob):
    return -sum(p * log2(p) for p in feature_prob)

def IG(model):
    
    # Store Information Gain in a dictionary
    IG_dict = {}  
    
    # Identify the target variable (last column)
    target_feature = model.columns[-1]
    
    # Calculate the entropy of the target variable 'PlayTennis'
    target_entropy = entropy(model[target_feature].value_counts(normalize=True))

    # Get the all features (excluding the target variable)
    feature_all = model.columns[:-1]

    for feature in feature_all:

        # Calculate weight entropy
        weighted_entropy = 0
        for value in model[feature].unique():
        
            subset = model[model[feature] == value]
            subset_entropy = entropy(subset[target_feature].value_counts(normalize=True))
        
            weighted_entropy += (len(subset) / len(model)) * subset_entropy

        # Calculate Information Gain
        information_gain = target_entropy - weighted_entropy
        
        # Format IG to 3 decimal places
        formatted_ig = '{:.3f}'.format(information_gain)

        # Store Information Gain in the dictionary
        IG_dict[feature] = formatted_ig
    
    return IG_dict


# In[3]:


# Print Information Gain of play tennis set and emails set

#this tennis dataset have Day feature, so I will update df with this feature removed

removeDay= tennis_df.drop('Day', axis= 1)
tennis_df= removeDay

print("Information Gains of playtennis dataset:", IG(tennis_df))

#this email dataset have ID feature, so I will update df with this feature removed

removeID= email_df.drop('ID', axis= 1)
email_df= removeID

print("Information Gains of email dataset:", IG(email_df))


# In[4]:


# Print Information Gain of training dataset

training_all = training_df.columns[:-1]

print("Information Gains of training dataset:", IG(training_df))


# In[5]:


# Try to build decision tree and save it as dictionary 

#Testing set-up rootkey



# Add best_feature as the root key
best_feature = max(IG(email_df), key=IG(email_df).get)

root_key= list(IG(email_df).keys())[0]
print("root_key:", root_key)



#tennis set

best_feature = max(IG(email_df), key=IG(email_df).get)

# Add best_feature as the root key

root_key= list(IG(tennis_df).keys())[0]
print("root_key:", root_key)



# Make the tree here and return it as a dict
def decision_tree(model, current_depth=0):
    # Check if there are any features left to split on
    if len(model.columns) == 1:
        # If there are no more features to split on, return the mode of the target variable
        return model[model.columns[-1]].mode().iloc[0]
    
    # Define root key
    best_feature = max(IG(model), key=IG(model).get)
    current_node = {best_feature: {}}
    
    # Iterate through unique values of the best_feature
    for value in model[best_feature].unique():
        subset = model[model[best_feature] == value]
        
        if entropy(subset[model.columns[-1]].value_counts(normalize=True)) == 0:
            # If entropy is 0, it's a leaf node
            current_node[best_feature][value] = subset[model.columns[-1]].mode().iloc[0]
        else:
            # Drop the current feature and recursively build child nodes with increased depth
            subset = subset.drop(columns=[best_feature])
            current_node[best_feature][value] = decision_tree(subset, current_depth + 1)

    return current_node


# In[6]:


# Testing result
result_email=decision_tree(email_df)
print("Email Tree:",result_email)

result_tennis=decision_tree(tennis_df)
print("Tennis Tree:",result_tennis)

# Use this decision_tree on traning_df

result_training=decision_tree(training_df)
print("Training Tree:",result_training)


# In[7]:


# Build a graphical tree (from Module 3)
from graphviz import Digraph

def draw_decision_tree_dictionary(tree_dictionary):
    if not isinstance(tree_dictionary, dict):
        raise TypeError("Argument tree_dictionary must be of type dictionary")
    if not tree_dictionary:
        raise ValueError("Dictionary tree_dictionary is empty")
        
    dot= Digraph(strict= True)
    draw_tree(dot, tree_dictionary, None)
    
    return dot

def draw_tree(dot, tree_dictionary, parent_node_name):
    if isinstance(tree_dictionary, dict):
        for key in tree_dictionary:
            no_spaces_key= str(key).replace(" ", "")
            
            dot.node(no_spaces_key, str(key), shape= "ellipse")
            
            if parent_node_name != None:
                dot.edge(parent_node_name, no_spaces_key)
                
            draw_tree(dot, tree_dictionary[key], no_spaces_key)
    else:
        val= str(tree_dictionary)
        dot.node(val, val, shape= "plaintext")
        dot.edge(parent_node_name, val)


# In[8]:


drawEmail= draw_decision_tree_dictionary(result_email)
drawEmail


# In[9]:


drawTennis= draw_decision_tree_dictionary(result_tennis)
drawTennis


# In[10]:


drawTraining= draw_decision_tree_dictionary(result_training)
drawTraining

# Just for curious but this was scary to see, like a spider web


# In[11]:


# Test accuracy with testing dataset

def accuracy_data(model, test_df, training_df, print_accuracy=True):
    correct_predictions = 0
    incorrect_predictions = 0

    # Find the most common class in the training set
    default_class = training_df[training_df.columns[-1]].mode().iloc[0]

    for index in range(len(test_df)):
        # Start at the root of the decision tree
        current_node = model
        test_instance = test_df.iloc[index]
        
        # While the current_node is a decision node (dictionary), keep traversing the tree
        while isinstance(current_node, dict):
            feature = list(current_node.keys())[0]
            value = test_instance[feature]
            current_node = current_node[feature].get(value, default_class)
            
        # If we've reached a leaf node (class label), compare it with the true label
        if current_node == test_instance[test_df.columns[-1]]:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    # Calculate accuracy
    total_examples = len(test_df)
    accuracy = correct_predictions / total_examples * 100 # so that it is into %

    if print_accuracy:
        print("Number of testing examples:", total_examples)
        print("Correct classification count:", correct_predictions)
        print("Incorrect classification count:", incorrect_predictions)
        print("Accuracy:", accuracy, "%")

    return accuracy


# In[12]:


# Test result for the training data

accuracy = accuracy_data(result_training, testing_df, training_df)


# In[21]:


# Pre-pruning tree with current partition (node) size at 5. 

def pre_prune(model, current_depth=0, max_depth= 5):
    # Check if there are any features left to split on or if the current depth has reached the maximum depth
    if len(model.columns) == 1 or current_depth == max_depth:
        # If there are no more features to split on or if the maximum depth is reached, return the mode of the target variable
        return model[model.columns[-1]].mode().iloc[0]
    
    # Define root key
    best_feature = max(IG(model), key=IG(model).get)
    current_node = {best_feature: {}}
    
    # Iterate through unique values of the best_feature
    for value in model[best_feature].unique():
        subset = model[model[best_feature] == value]
        
        if entropy(subset[model.columns[-1]].value_counts(normalize=True)) == 0:
            # If entropy is 0, it's a leaf node
            current_node[best_feature][value] = subset[model.columns[-1]].mode().iloc[0]
        else:
            # Drop the current feature and recursively build child nodes with increased depth
            subset = subset.drop(columns=[best_feature])
            current_node[best_feature][value] = pre_prune(subset, current_depth + 1, max_depth)

    return current_node


# In[22]:


# Prune the decision tree
pruned_tree = pre_prune(training_df)

# Calculate the accuracy of the pruned tree
accuracy = accuracy_data(pruned_tree, testing_df, training_df)

