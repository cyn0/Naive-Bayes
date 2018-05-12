from scipy.io import arff
import math
import sys
import pdb


def do_predict(row, attributes_list, count, class_count, class_attribute):
    prob = [1, 1]
    den = 2
    for cls in class_attribute[1]:
        den += class_count[cls]

    for i in range(len(attributes_list)):
        attr = attributes_list[i]
        feature_value = row[i]

        num_attr = len(count[attr])
        for index, value in enumerate(class_attribute[1]):
            prob[index] *= ( (count[attr][feature_value][value] +1) / (class_count[value] + num_attr) )

    for index, value in enumerate(class_attribute[1]):
        prob[index] *= ( (class_count[value] + 1) / den)

    if prob[0] > prob[1]:
        # print prob[0]/(prob[0] + prob[1])
        return 0, prob[0]/(prob[0] + prob[1]), prob[0]/(prob[0] + prob[1])
    else:
        # print prob[1] / (prob[0] + prob[1])
        return 1, prob[1]/(prob[0] + prob[1]), prob[0]/(prob[0] + prob[1])


def do_count(rows, attributes_list, count, class_count):
    for row in rows:
        for i in range(len(attributes_list)):
            attr = attributes_list[i]
            value = row[i]
            count[attr][value][row[-1]] += 1
        class_count[row[-1]] += 1


def initialise_count(attributes, count, class_count, class_value, class_attribute):
    for attribute in attributes:
        count[attribute] = {}
        for value in attributes[attribute][1]:
            count[attribute][value] = {}
            for c in attributes[class_value][1]:
                count[attribute][value][c] = 0.0

    for cls in class_attribute[1]:
        class_count[cls] = 0.0


def count_three_attr_occurence(attributes_list, feature_a, feature_a_value, feature_b, feature_b_value, rows, class_attribute):
    ind1 = attributes_list.index(feature_a)
    ind2 = attributes_list.index(feature_b)

    class_count = {}
    for cls in class_attribute[1]:
        class_count[cls] = 0.0

    for row in rows:
        if row[ind1] == feature_a_value and row[ind2] == feature_b_value:
            class_count[row[-1]] += 1

    return class_count


def calculate_conditional_mutual_info(attributes_list, attributes, feature_a, feature_b, class_attribute, rows, count, total_attribute_count):
    if feature_a == feature_b:
        return -1.0
    feature_a_attributes = attributes[feature_a][1]
    feature_b_attributes = attributes[feature_b][1]

    info = 0.0
    for attr1 in feature_a_attributes:
        for attr2 in feature_b_attributes:
            class_count = count_three_attr_occurence(attributes_list, feature_a, attr1, feature_b, attr2, rows, class_attribute)
            num_attr_a = len(feature_a_attributes)
            num_attr_b = len(feature_b_attributes)

            for cls in class_attribute[1]:

                p_a1_a2_given_y = (class_count[cls] + 1) / (total_attribute_count[cls] + (num_attr_a * num_attr_b))
                p_a1_given_y = (count[feature_a][attr1][cls] + 1) / (total_attribute_count[cls] + num_attr_a)
                p_a2_given_y = (count[feature_b][attr2][cls] + 1) / (total_attribute_count[cls] + num_attr_b)
                right_part = math.log(p_a1_a2_given_y / (p_a1_given_y * p_a2_given_y), 2)
                left_part = (class_count[cls] + 1) / (len(rows) + (num_attr_a * num_attr_b * 2))

                info += left_part * right_part

    return info


def calculate_info_gain_matrix(rows, count, class_count, attributes, attributes_list, class_attribute ):
    mutual_info = []
    for attr1 in attributes_list:
        mutual_info_row = []
        for attr2 in attributes_list:
            mutual_info_val = calculate_conditional_mutual_info(attributes_list, attributes, attr1, attr2,
                                                                class_attribute, rows, count, class_count)
            mutual_info_row.append(mutual_info_val)
        mutual_info.append(mutual_info_row)
    return mutual_info


def do_predict_tan(train_rows, parent_tree, row, attributes_list, count, total_class_count, class_attribute):
    prob = [1, 1]
    den = 2
    for cls in class_attribute[1]:
        den += total_class_count[cls]

    for i in range(len(attributes_list)):
        attr = attributes_list[i]
        feature_value = row[i]
        num_attr = len(count[attr])

        parent_attr = parent_tree.get(attr, None)

        if parent_attr:
            parent_attr_index = attributes_list.index(parent_attr)
            parent_attr_val = row[parent_attr_index]
            class_count = count_three_attr_occurence(attributes_list, attr, feature_value, parent_attr, parent_attr_val,
                                                     train_rows, class_attribute)

            for index, value in enumerate(class_attribute[1]):
                # prob[index] *= ((class_count[value] + 1) / (count[parent_attr][parent_attr_val][value] + num_attr))
                t = ((class_count[value] + 1) / (count[parent_attr][parent_attr_val][value] + num_attr))
                prob[index] *= t

        else:
            for index, value in enumerate(class_attribute[1]):
                prob[index] *= ((count[attr][feature_value][value] + 1) / (total_class_count[value] + num_attr))

    for index, value in enumerate(class_attribute[1]):
        prob[index] *= ((total_class_count[value] + 1) / den)

    if prob[0] > prob[1]:
        # print prob[0]/(prob[0] + prob[1])
        return 0, prob[0] / (prob[0] + prob[1]), prob[0] / (prob[0] + prob[1])
    else:
        # print prob[1] / (prob[0] + prob[1])
        return 1, prob[1] / (prob[0] + prob[1]), prob[0] / (prob[0] + prob[1])


def construct_mst(mutual_info, attributes_list):
    tree = {}
    tree_nodes = [0]

    while len(tree_nodes) < len(attributes_list):
        max_val = -1
        max_node = None

        for current_node in tree_nodes:
            current_row = mutual_info[current_node]

            for j in range(len(current_row)):
                if j in tree_nodes:
                    continue

                if current_row[j] > max_val:
                    max_val = current_row[j]
                    max_node = (current_node, j)

        tree[attributes_list[max_node[1]]] = attributes_list[max_node[0]]
        tree_nodes.append(max_node[1])

    return tree


def main():
    args = sys.argv
    # rows, meta = arff.loadarff('lymph_train.arff')
    rows, meta = arff.loadarff(args[1])

    attributes_list = meta._attrnames
    attributes = meta._attributes

    class_value = attributes_list[-1]
    attributes_list.remove(class_value)
    class_attribute = attributes[class_value]

    count = {}
    class_count = {}
    initialise_count(attributes, count, class_count, class_value, class_attribute)
    do_count(rows, attributes_list, count, class_count)

    correct_prediction = 0
    test_rows, meta = arff.loadarff(args[2])

    algo = args[3]
    tree = None

    if algo == "n":
        # Do naive bayes
        for attr in attributes_list:
            print "%s %s" % (attr, class_value)

    else:
        # Do TAN
        mutual_info = calculate_info_gain_matrix(rows, count, class_count, attributes, attributes_list, class_attribute)
        tree = construct_mst(mutual_info, attributes_list)

        for attr in attributes_list:
            parent_attr = tree.get(attr, None)
            if not parent_attr:
                print "%s %s" % (attr, class_value)
            else:
                print "%s %s %s" % (attr, parent_attr, class_value)

    print ""

    t_rows = []

    for test_row in test_rows:
        index, prob = None, None
        if algo == "n":
            index, prob, meta_prob = do_predict(test_row, attributes_list, count, class_count, class_attribute)

        else:
            index, prob, meta_prob = do_predict_tan(rows, tree, test_row, attributes_list, count, class_count, class_attribute)

        t_rows.append([test_row[-1], meta_prob])
        print "%s %s %0.12f" % (class_attribute[1][index], test_row[-1], prob)
        if class_attribute[1][index] == test_row[-1]:
            correct_prediction += 1

    print "\n%d" % (correct_prediction,)


if __name__ == "__main__":
    main()
