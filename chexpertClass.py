import csv
from torch.utils.data import Dataset
from PIL import Image

class CheXpertData(Dataset):
    """
    A customized data loader for Chexpert.
    """
    def __init__(self,
                 root,
                 transform=None,
                 preload=False,
                 policy="ones"):
        """ Intialize the cheXpert dataset
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.policy = policy

        # read filenames
        with open(self.root, 'r') as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            k = 0
            for line in csvReader:
                k+= 1
                image_name = line[0]
                label  = line[5:]
                self.filenames.append((image_name,label))
        
        # if preload dataset into memory
        if preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
    def _applypolicy(self, label):
        policy = self.policy
        for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
        return label
        
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
        for image_fn, label in self.filenames:            
            # load images
            image = Image.open('datasets/chexpert-small/' + image_fn).convert('RGB')
            self.images.append(image.copy())
            # avoid too many opened files bug
            image.close()
            label = self._applypolicy(label)
            self.labels.append(label)

    # probably the most important to customize.
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            image_fn, label = self.filenames[index]
            label = self._applypolicy(label)
            image = Image.open('datasets/chexpert-small/' + image_fn).convert('RGB')
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
        # return image and label
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len