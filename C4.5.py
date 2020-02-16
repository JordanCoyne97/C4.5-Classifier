import pandas as pd
import math
from sklearn.model_selection import train_test_split
from collections import Counter
from random import randrange


class Node:
    def __init__(self, leaf, threshold, column, value):
        self.value = value
        self.leaf = leaf
        self.threshold = threshold
        self.column = column
        self.children = []


def split_data(data):
    train, test = train_test_split(data, test_size=0.33, random_state=randrange(100))
    return train, test


def calc_dataset_entropy(data, target_name):
    value_counts = data[target_name].value_counts()
    total_count = len(data)

    entropy = 0
    for i in value_counts:
        entropy += i / total_count * math.log2(i / total_count)
    return -entropy


def get_index_names(data):
    return list(data.columns.values)


def get_split_info(less_than, greater_than):
    total = less_than + greater_than

    if greater_than != 0:  # problems when the greater_than list is empty "math.log2(0)"
        return -less_than / total * math.log2(less_than / total) - greater_than / total * math.log2(
            greater_than / total)
    else:
        return 1


def gain_ratio(target_lesser, target_greater, less_values, greater_values, entropy):
    target_greater_count = Counter(target_greater)
    target_less_count = Counter(target_lesser)

    less_total_count = len(less_values)
    greater_total_count = len(greater_values)

    total_count = less_total_count + greater_total_count

    less_than_entropy = 0.0
    greater_than_entropy = 0.0

    for i in target_greater_count:
        greater_than_entropy += -(target_greater.count(i) / greater_total_count *
                                  math.log2(target_greater.count(i) / greater_total_count))

    for j in target_less_count:
        less_than_entropy += -(target_lesser.count(j) / less_total_count *
                               math.log2(target_lesser.count(j) / less_total_count))

    less_proportion = less_total_count / total_count
    greater_proportion = greater_total_count / total_count

    info_gain = entropy - (less_than_entropy * less_proportion) - (greater_than_entropy * greater_proportion)

    split_info = get_split_info(less_total_count, greater_total_count)

    gain_ratio_for_threshold = float(info_gain) / float(split_info)

    return gain_ratio_for_threshold


def get_gain(sorted_by_attribute, value, target, entropy):
    data_len = len(sorted_by_attribute)
    gain, threshold = 0, 0
    column_of_best_gain = ""

    for i in range(data_len):
        current_val = sorted_by_attribute[value].iloc[i]

        less_than = []
        less_than_targets = []
        greater_than = []
        greater_than_targets = []

        for j in range(data_len):
            if sorted_by_attribute[value].iloc[j] <= current_val:
                less_than = less_than + [sorted_by_attribute[value].iloc[j]]
                less_than_targets = less_than_targets + [sorted_by_attribute[target].iloc[j]]

            else:
                greater_than = greater_than + [sorted_by_attribute[value].iloc[j]]
                greater_than_targets = greater_than_targets + [sorted_by_attribute[target].iloc[j]]

        new_gain = gain_ratio(less_than_targets, greater_than_targets, less_than, greater_than, entropy)

        if gain < new_gain:
            threshold = sorted_by_attribute[value].iloc[i]
            column_of_best_gain = value
            gain = gain_ratio(less_than_targets, greater_than_targets, less_than, greater_than, entropy)

    return gain, column_of_best_gain, threshold


def count_targets(data, target_name):
    target_list = [i for i, j in Counter(data[target_name]).most_common()]
    return target_list


def get_majority_class(data, target_name):
    max_value_type = data[target_name].value_counts().idmax()
    return max_value_type


def tree(data, index_names, target_name):
    return create_tree(data, index_names, target_name)


def create_tree(data, index_names, target_name):
    threshold, new_threshold = 0, 0
    best_gain, new_gain = 0, 0
    column, new_column = " ", " "

    target_classes = count_targets(data, target_name)

    # If there is nothing in the data set return a failed node
    if len(target_classes) == 0:
        node = Node(True, None, None, "Failed")

    # If there is only one type of target set it as a leaf node
    if len(target_classes) == 1:
        node = Node(True, None, target_name, target_classes[0])

    # If there more than 1 type of target
    elif target_classes:

        # Finds the best gain_ratio and the threshold for each column(attributes)
        for i in range(len(index_names)):
            if index_names[i] != target_name:
                entropy = calc_dataset_entropy(data, target_name)
                sorted_by_attribute = data.sort_values(by=index_names[i])
                new_gain, new_column, new_threshold = get_gain(sorted_by_attribute, index_names[i], target_name,
                                                               entropy)

            if best_gain < new_gain:
                best_gain = new_gain
                threshold = new_threshold
                column = new_column

        left = [left_set for left_set in data[column] if threshold >= left_set]
        data_left = data.loc[data[column].isin(left)]

        right = [right_set for right_set in data[column] if threshold < right_set]
        data_right = data.loc[data[column].isin(right)]

        node = Node(False, threshold, column, None)
        node.children = [create_tree(data_left, index_names, target_name),
                            create_tree(data_right, index_names, target_name)]

    return node


def set_up_predict(node, test_data, target):
    correct_count = 0
    results_list = "Predicted, Actual, Correct"

    total = len(test_data)
    for i in range(total):
        row = test_data.iloc[i]
        predicted, actual, answer = predict(node, row, target)

        results_list = results_list + "\n" + predicted + ", " + actual + ", " + str(answer)

        if answer:
            correct_count = correct_count + 1

    accuracy = float(correct_count/total)
    results_list = ["Total accuracy = " + str(accuracy)] + [results_list]

    with open('results.txt', 'w') as f:
        for row in results_list:
            f.write("%s\n" % str(row))

    print("total accuracy is: " + str(accuracy))


def predict(node, data_row, target):
    column = node.column
    threshold = node.threshold

    left_child = node.children[0]
    right_child = node.children[1]

    answer = False
    predicted = " "
    actual = " "

    value = data_row[column]

    if value <= threshold:
        if left_child.leaf:

            if data_row[target] == left_child.value:
                answer = True
                predicted = data_row[target]
                actual = left_child.value
            else:
                answer = False
                predicted = data_row[target]
                actual = left_child.value
        else:
            return predict(left_child, data_row, target)

    else:
        if right_child.leaf:

            if data_row[target] == right_child.value:
                answer = True
                predicted = data_row[target]
                actual = right_child.value
            else:
                answer = False
                predicted = data_row[target]
                actual = right_child.value
        else:
            return predict(right_child, data_row, target)

    return predicted, actual, answer


def main():
    file = (input("Please enter the path to csv file or leave blank for hazelnuts.csv: ") or 'hazelnuts.csv')
    data_set = pd.read_csv(file)

    print(data_set.iloc[0])
    target = (input("\nPlease enter the classifier target name: "))

    print("Building tree and making prediction ...")

    index_names = get_index_names(data_set)
    train_data, test_data = split_data(data_set)

    classifier = tree(train_data, index_names, target)

    set_up_predict(classifier, test_data, target)


if __name__ == "__main__":
    main()
