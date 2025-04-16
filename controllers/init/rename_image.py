import os

folder_path = ".\\data\\easy1\\"

prefix = 'easy1_'

i = 0

for filename in os.listdir(folder_path):
    i += 1
    os.rename(os.path.join(folder_path + filename), os.path.join(folder_path + prefix + str(i) + '.png'))
        
print("Completed")