from torchvision import datasets, transforms
from base import BaseDataLoader
from pathlib import Path

class DataSet(Dataset):
        def __init__(self, datalist):
            self.datalist = datalist
            
        def __getitem__(self, index):
            exposure1, exposure2, exposure3, hdr = self.datalist[index] 
            image_tensor1 = img_transform(PIL.Image.open(exposure1))
            image_tensor2 = img_transform(PIL.Image.open(exposure2))
            image_tensor3 = img_transform(PIL.Image.open(exposure3))
            image_tensor4 = img_transform(PIL.Image.open(hdr))
            image_tensor4 = torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])(image_tensor4)
            return image_tensor1, image_tensor2, image_tensor3, image_tensor4
        
        def __len__(self):
            return len(self.datalist)

class HdrDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        img_transform = transforms.Compose([
                    transforms.Resize((270, 480)),
                    transforms.ToTensor()
                    ])
        self.data_dir = data_dir
        self.data_list=[]
        for video_dir in Path(self.data_dir).iter_dir():
            if video_dir.is_dir()
                image_files = [file for file in natsorted(list(video_dir.iter_dir())) if 'png' in file.name]
                for i in range(0,(int(len(image_files)/4)-2)):
                    for k in range(1,4):
                        for j in range(3):
                            try:
                                exp1 = f"{self.data_dir}/{image_files[(i*4)+(j)]}"
                                exp2 = f"{self.data_dir}/{image_files[(i+1*k)*4+((j+1)%3)]}"
                                exp3 = f"{self.data_dir}/{image_files[(i+2*k)*4+((j+2)%3)]}"
                                hdr = f"{self.data_dir}/{image_files[(i+1*k)*4+3]}"
                                self.data_list.append((exp1, exp2, exp3, hdr))
                            except IndexError:
                                continue
        self.dataset = DataSet(self.datalist)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
