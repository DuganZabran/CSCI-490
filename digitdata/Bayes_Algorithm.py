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

print(training_nums)

# probability matrix of Fij given C
probability_matrix = []


for i in range(len(training_data)):
    num = training_data[i]


