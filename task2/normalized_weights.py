from pprint import pprint
def normalized_weights(labels):
    '''
    param: labels - a list (or iterable object) of _labels_
    returns: a list of weights 'weights',  of length len(labels), where weights[i] = appropriate normalized weight

    Here is how appropriate normalized weight is defined - if there are c unique classes
    in the labels list, then each label is assigned 1/c amount of weight total.
    Think of 1 as the "total supply" of weight, and We're redistributing it
    So that each label receives the same amount, regardless of the number of samples
    of that label in the labels list.

    Example: there are 3 labels, so each label receives 0.33 total weight assignment.
    Then, if there is only one sample of label 1, then that sample receives weight 0.33
    If there are three samples of label two, then each sample receives a weight of 0.11
    to sum to 0.33 etc.
    '''
    #keeps track of unique labels and their counts
    unique_labels = {}
    for i in labels:
        if i not in unique_labels:
            unique_labels[i] = 1
        else:
            unique_labels[i] += 1

    #each label gets the same amount of "total weight", 1/number of unique labels
    balanced_weight_allocation = 1/len(unique_labels.keys())
    for key in unique_labels.keys():
        #unique_labels[label] now holds the appropriate weight each of it's samples should have
        #which is balanced_weight_allocation/number of samples of that label
        unique_labels[key] = balanced_weight_allocation/unique_labels[key]

    weights = []
    for i in range(0, len(labels)):
        weights.append(unique_labels[labels[i]])
    return weights
    
    #assert(sum(weights) == 1)
