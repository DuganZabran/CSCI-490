import sys
import numpy
import numpy as np

training_images_file = open('trainingimages.txt', 'r').read().splitlines()
training_data_file = open('traininglabels.txt', 'r').read().splitlines()

for i in range(len(training_images_file)):
    training_images_file[i] = training_images_file[i].replace("+", "1")
    training_images_file[i] = training_images_file[i].replace("#", "1")
    training_images_file[i] = training_images_file[i].replace(" ", "0")

training_data = [int(numeric_string) for numeric_string in training_data_file]
training_images_data = []
for i in range(0, len(training_images_file), 28):
    np_image =np.empty((0, 28), int)
    for j in range(28):
        np_image_line = np.array([int(numeric_string) for numeric_string in training_images_file[i+j]])
        np_image = np.append(np_image, [np_image_line], axis=0)
    training_images_data.append(np_image)

training_nums = {}
for num in training_data:
    if num in training_nums:
        training_nums[num] += 1
    else:
        training_nums[num] = 0

# probability matrix of Fij given C
probability_matrix_1 = []
probability_matrix_0 = []
for i in range(10):
    probability_matrix_1.append(numpy.zeros((28, 28)))
    probability_matrix_0.append(numpy.zeros((28, 28)))

for i in range(len(training_data)):
    num = training_data[i]
    for k in range(28):
        for j in range(28):
            if training_images_data[i][k][j] == 1:
                probability_matrix_1[num][k][j] += 1
            elif training_images_data[i][k][j] == 0:
                probability_matrix_0[num][k][j] += 1


k = 0.1
for i in range(len(probability_matrix_1)):
    for l in range(28):
        for j in range(28):
            probability_matrix_1[i][l][j] = ((k + probability_matrix_1[i][l][j]) / ((2 * k) + training_nums[i]))
            probability_matrix_0[i][l][j] = ((k + probability_matrix_0[i][l][j]) / ((2 * k) + training_nums[i]))

for i in range(10):
    training_nums[i] /= 5000



test_images_file = open('testimages.txt', 'r').read().splitlines()
test_data_file = open('testlabels.txt', 'r').read().splitlines()

for i in range(len(test_images_file)):
    test_images_file[i] = test_images_file[i].replace("+", "1")
    test_images_file[i] = test_images_file[i].replace("#", "1")
    test_images_file[i] = test_images_file[i].replace(" ", "0")

test_data = [int(numeric_string) for numeric_string in test_data_file]
test_images_data = []
for i in range(0, len(test_images_file), 28):
    np_image = np.empty((0, 28), int)
    for j in range(28):
        np_image_line = np.array([int(numeric_string) for numeric_string in test_images_file[i+j]])
        np_image = np.append(np_image, [np_image_line], axis=0)
    test_images_data.append(np_image)



max_probability = -sys.maxsize - 1
number = -1

# classifier = 15
#
# for i in range(10):
#     probability = np.log(training_nums[i])
#     for l in range(28):
#         for j in range(28):
#             if test_images_data[classifier][l][j] == 1:
#                 probability += np.log(probability_matrix_1[i][l][j])
#             elif test_images_data[classifier][l][j] == 0:
#                 probability += np.log(probability_matrix_0[i][l][j])
#     print(i)
#     print(probability)
#     if probability > max_probability:
#         max_probability = probability
#         number = i
#
# print("\n\n")
# print(test_images_data[classifier])
# print("\n")
# print(max_probability)
# print(str(number) + " = " + str(test_data[classifier]))

correct = 0
for classifier in range(len(test_data)):
    max_probability = -sys.maxsize - 1
    number = -1
    for i in range(10):
        probability = np.log(training_nums[i])
        for l in range(28):
            for j in range(28):
                if test_images_data[classifier][l][j] == 1:
                    probability += np.log(probability_matrix_1[i][l][j])
                elif test_images_data[classifier][l][j] == 0:
                    probability += np.log(probability_matrix_0[i][l][j])
        if probability > max_probability:
            max_probability = probability
            number = i
    if number == test_data[classifier]:
        correct += 1

print(len(test_data))
print(correct)
correct /= len(test_data)
print("Percent Correct " + str(correct*100) + "%")