import os
import random
import cv2
import numpy as np
from PIL import Image
import pdb
from torch.utils.data import Dataset
import IPython

class BaseDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all paths in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.name = 'single_stream'
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.open(self.images[index]).convert('RGB')
        x = self.transform(x)
        y = self.labels[index]
        return x, y

class BaseMultiStreamDataset(Dataset):
    """Dataset for loading multistream images"""
    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images_rgb = data['x_rgb']
        self.images_edge = data['x_edge']
        self.name = 'multi_stream'
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images_rgb)

    def __getitem__(self, index):
        """Generates one sample of data"""
        
        # print(self.images_rgb[index])
        # print(self.images_edge[index])
        # print("========")

        x_rgb = Image.open(self.images_rgb[index]).convert('RGB')
        x_edge = Image.open(self.images_edge[index])

        # transform with corresponding type
        x_rgb = self.transform['rgb'](x_rgb)
        x_edge = self.transform['edge'](x_edge)

        # get color histogram of image
        # tmp_img = cv2.imread(self.images_rgb[index])
        # tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        # hist = cv2.calcHist([tmp_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # hist = cv2.normalize(hist, hist).flatten()
        # hist = np.expand_dims(hist, axis=0)

        y = self.labels[index]
        return x_rgb, x_edge, y

class BaseMultiStreamHistoDataset(Dataset):
    """Dataset for loading multistream images"""
    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images_rgb = data['x_rgb']
        # extract histo here 

        self.histo_info = []
        for rgb_img_path in self.images_rgb:
            tmp_img = cv2.imread(rgb_img_path)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            hist = cv2.calcHist([tmp_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            self.histo_info.append(hist)


        # self.images_edge = data['x_edge']
        self.name = 'multi_stream'
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images_rgb)

    def __getitem__(self, index):
        """Generates one sample of data"""
        
        # print(self.images_rgb[index])
        # print(self.images_edge[index])
        # print("========")

        x_rgb = Image.open(self.images_rgb[index]).convert('RGB')
        # x_edge = Image.open(self.images_edge[index])

        # transform with corresponding type
        x_rgb = self.transform['rgb'](x_rgb)
        # x_edge = self.transform['edge'](x_edge)

        # get color histogram of image
        # tmp_img = cv2.imread(self.images_rgb[index])
        # tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        # hist = cv2.calcHist([tmp_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # hist = cv2.normalize(hist, hist).flatten()
        hist = self.histo_info[index]
        # hist = np.expand_dims(hist, axis=0)

        y = self.labels[index]
        return x_rgb, hist, y

class BaseMultiStreamContourDataset(Dataset):
    """Dataset for loading multistream images"""
    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images_rgb = data['x_rgb']
        self.images_edge = data['x_edge']

        self.name = 'multi_stream'
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images_rgb)

    def __getitem__(self, index):
        """Generates one sample of data"""
        
        # print(self.images_rgb[index])
        # print(self.images_edge[index])
        # print("========")

        x_rgb = Image.open(self.images_rgb[index]).convert('RGB')
        x_edge = Image.open(self.images_edge[index])

        # transform with corresponding type
        x_rgb = self.transform['rgb'](x_rgb)
        x_edge = self.transform['edge'](x_edge)

        y = self.labels[index]
        return x_rgb, x_edge, y


class Base3StreamHistoDataset(Dataset):
    """Dataset for loading multistream images"""
    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images_rgb = data['x_rgb']
        self.images_edge = data['x_edge']
        # extract histo here 

        self.histo_info = []
        for rgb_img_path in self.images_rgb:
            tmp_img = cv2.imread(rgb_img_path)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            hist = cv2.calcHist([tmp_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            self.histo_info.append(hist)


        self.name = 'multi_stream'
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images_rgb)

    def __getitem__(self, index):
        """Generates one sample of data"""
        
        # print(self.images_rgb[index])
        # print(self.images_edge[index])
        # print("========")

        x_rgb = Image.open(self.images_rgb[index]).convert('RGB')
        x_edge = Image.open(self.images_edge[index])

        # transform with corresponding type
        x_rgb = self.transform['rgb'](x_rgb)
        x_edge = self.transform['edge'](x_edge)

        # get color histogram of image
        # tmp_img = cv2.imread(self.images_rgb[index])
        # tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        # hist = cv2.calcHist([tmp_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # hist = cv2.normalize(hist, hist).flatten()
        hist = self.histo_info[index]
        # hist = np.expand_dims(hist, axis=0)

        y = self.labels[index]
        return x_rgb, x_edge, hist, y

def get_data(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []
    is_multistream_data = False


    if not isinstance(path, dict):
        # read filenames and labels
        trn_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)

    else:
        # pill multistream data
        is_multistream_data = False # now set false for testing histogram

        # both txt must be the same
        tmp = path['rgb']
        trn_lines = np.loadtxt(os.path.join(tmp, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(tmp, 'test.txt'), dtype=str)

    # IPython.embed()
    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure, multistream option here
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)

        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}


    # ALL OR TRAIN
    for this_image, this_label in trn_lines:

        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in tst_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order

def get_data_multistream(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []
    is_multistream_data = False

    if not isinstance(path, dict):
        # read filenames and labels
        trn_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)

    else:
        # pill multistream data
        is_multistream_data = True

        # both txt must be the same
        tmp = path['rgb']
        trn_lines = np.loadtxt(os.path.join(tmp, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(tmp, 'test.txt'), dtype=str)

    
    # IPython.embed()
    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure, multistream option here
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)

        if is_multistream_data:
            data[tt]['trn'] = {'x_rgb': [], 'x_edge': [], 'y': []}
            data[tt]['val'] = {'x_rgb': [], 'x_edge': [], 'y': []}
            data[tt]['tst'] = {'x_rgb': [], 'x_edge': [], 'y': []}
        
        else:
            data[tt]['trn'] = {'x': [], 'y': []}
            data[tt]['val'] = {'x': [], 'y': []}
            data[tt]['tst'] = {'x': [], 'y': []}


    # ALL OR TRAIN
    
    for this_image, this_label in trn_lines:
        # add path of rgb and edge image
        img_name_only = this_image.split('/')[-1]
        this_image_rgb = os.path.join(path['rgb'], 'train', img_name_only)
        this_image_edge = os.path.join(path['edge'], 'train', img_name_only)

        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x_rgb'].append(this_image_rgb)
        data[this_task]['trn']['x_edge'].append(this_image_edge)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in tst_lines:
        
        # add path of rgb and edge image
        img_name_only = this_image.split('/')[-1]
        this_image_rgb = os.path.join(path['rgb'], 'test', img_name_only)
        this_image_edge = os.path.join(path['edge'], 'test', img_name_only)

        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x_rgb'].append(this_image_rgb)
        data[this_task]['tst']['x_edge'].append(this_image_edge)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x_rgb'].append(data[tt]['trn']['x_rgb'][rnd_img[ii]])
                    data[tt]['val']['x_edge'].append(data[tt]['trn']['x_edge'][rnd_img[ii]])

                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x_rgb'].pop(rnd_img[ii])
                    data[tt]['trn']['x_edge'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order

def get_data_multistream_his(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []
    is_multistream_data = False

    if not isinstance(path, dict):
        # read filenames and labels
        trn_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)

    else:
        # pill multistream data
        is_multistream_data = True

        # both txt must be the same
        tmp = path['rgb']
        trn_lines = np.loadtxt(os.path.join(tmp, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(tmp, 'test.txt'), dtype=str)

    
    # IPython.embed()
    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure, multistream option here
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)

        if is_multistream_data:
            data[tt]['trn'] = {'x_rgb': [], 'x_his': [], 'y': []}
            data[tt]['val'] = {'x_rgb': [], 'x_his': [], 'y': []}
            data[tt]['tst'] = {'x_rgb': [], 'x_his': [], 'y': []}
        
        else:
            data[tt]['trn'] = {'x': [], 'y': []}
            data[tt]['val'] = {'x': [], 'y': []}
            data[tt]['tst'] = {'x': [], 'y': []}


    # ALL OR TRAIN
    
    for this_image, this_label in trn_lines:
        # add path of rgb and edge image
        img_name_only = this_image.split('/')[-1]
        this_image_rgb = os.path.join(path['rgb'], 'train', img_name_only)
        # this_image_edge = os.path.join(path['edge'], 'train', img_name_only)

        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x_rgb'].append(this_image_rgb)
        # data[this_task]['trn']['x_edge'].append(this_image_edge)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in tst_lines:
        
        # add path of rgb and edge image
        img_name_only = this_image.split('/')[-1]
        this_image_rgb = os.path.join(path['rgb'], 'test', img_name_only)
        # this_image_edge = os.path.join(path['edge'], 'test', img_name_only)

        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x_rgb'].append(this_image_rgb)
        # data[this_task]['tst']['x_edge'].append(this_image_edge)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x_rgb'].append(data[tt]['trn']['x_rgb'][rnd_img[ii]])
                    # data[tt]['val']['x_edge'].append(data[tt]['trn']['x_edge'][rnd_img[ii]])

                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x_rgb'].pop(rnd_img[ii])
                    # data[tt]['trn']['x_edge'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order

def get_data_multistream_contour(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []
    is_multistream_data = False

    if not isinstance(path, dict):
        # read filenames and labels
        trn_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)

    else:
        # pill multistream data
        is_multistream_data = True

        # both txt must be the same
        tmp = path['rgb']
        trn_lines = np.loadtxt(os.path.join(tmp, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(tmp, 'test.txt'), dtype=str)

    
    # IPython.embed()
    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure, multistream option here
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)

        if is_multistream_data:
            data[tt]['trn'] = {'x_rgb': [], 'x_edge': [], 'y': []}
            data[tt]['val'] = {'x_rgb': [], 'x_edge': [], 'y': []}
            data[tt]['tst'] = {'x_rgb': [], 'x_edge': [], 'y': []}
        
        else:
            data[tt]['trn'] = {'x': [], 'y': []}
            data[tt]['val'] = {'x': [], 'y': []}
            data[tt]['tst'] = {'x': [], 'y': []}


    # ALL OR TRAIN
    
    for this_image, this_label in trn_lines:
        # add path of rgb and edge image
        img_name_only = this_image.split('/')[-1]
        this_image_rgb = os.path.join(path['rgb'], 'train', img_name_only)
        this_image_edge = os.path.join(path['edge'], 'train', img_name_only)

        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x_rgb'].append(this_image_rgb)
        data[this_task]['trn']['x_edge'].append(this_image_edge)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in tst_lines:
        
        # add path of rgb and edge image
        img_name_only = this_image.split('/')[-1]
        this_image_rgb = os.path.join(path['rgb'], 'test', img_name_only)
        this_image_edge = os.path.join(path['edge'], 'test', img_name_only)

        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x_rgb'].append(this_image_rgb)
        data[this_task]['tst']['x_edge'].append(this_image_edge)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x_rgb'].append(data[tt]['trn']['x_rgb'][rnd_img[ii]])
                    data[tt]['val']['x_edge'].append(data[tt]['trn']['x_edge'][rnd_img[ii]])

                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x_rgb'].pop(rnd_img[ii])
                    data[tt]['trn']['x_edge'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order

def get_data_3stream(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []
    is_multistream_data = False

    if not isinstance(path, dict):
        # read filenames and labels
        trn_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)

    else:
        # pill multistream data
        is_multistream_data = True

        # both txt must be the same
        tmp = path['rgb']
        trn_lines = np.loadtxt(os.path.join(tmp, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(tmp, 'test.txt'), dtype=str)

    
    # IPython.embed()
    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure, multistream option here
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)

        if is_multistream_data:
            data[tt]['trn'] = {'x_rgb': [], 'x_edge': [], 'x_his': [], 'y': []}
            data[tt]['val'] = {'x_rgb': [], 'x_edge': [], 'x_his': [], 'y': []}
            data[tt]['tst'] = {'x_rgb': [], 'x_edge': [], 'x_his': [], 'y': []}
        
        else:
            data[tt]['trn'] = {'x': [], 'y': []}
            data[tt]['val'] = {'x': [], 'y': []}
            data[tt]['tst'] = {'x': [], 'y': []}


    # ALL OR TRAIN
    
    for this_image, this_label in trn_lines:
        # add path of rgb and edge image
        img_name_only = this_image.split('/')[-1]
        this_image_rgb = os.path.join(path['rgb'], 'train', img_name_only)
        this_image_edge = os.path.join(path['edge'], 'train', img_name_only)

        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x_rgb'].append(this_image_rgb)
        data[this_task]['trn']['x_edge'].append(this_image_edge)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in tst_lines:
        
        # add path of rgb and edge image
        img_name_only = this_image.split('/')[-1]
        this_image_rgb = os.path.join(path['rgb'], 'test', img_name_only)
        this_image_edge = os.path.join(path['edge'], 'test', img_name_only)

        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x_rgb'].append(this_image_rgb)
        data[this_task]['tst']['x_edge'].append(this_image_edge)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x_rgb'].append(data[tt]['trn']['x_rgb'][rnd_img[ii]])
                    data[tt]['val']['x_edge'].append(data[tt]['trn']['x_edge'][rnd_img[ii]])

                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x_rgb'].pop(rnd_img[ii])
                    data[tt]['trn']['x_edge'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order
