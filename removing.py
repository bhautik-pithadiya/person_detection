import os
path = os.listdir('output_frames/')
img_path = []
txt_path = []
for i in path:
    if i.endswith('.jpg'):
        img_path.append('output_frames/' + i)
    else:
        txt_path.append('output_frames/' + i)


for i in img_path:
    # print(i[:-4])
    if i[:-4] not in txt_path:
        # print('in loop')
        os.remove(i[:-4] + '.txt')
        print('removed' + i[:-4] + '.txt')