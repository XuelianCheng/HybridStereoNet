from torch.utils.data import DataLoader
from dataloaders.datasets import stereo
import pdb

def make_data_loader(args, **kwargs):
        ############################ sceneflow ###########################
        if args.dataset == 'sceneflow':              
            train_list = 'dataloaders/lists/sceneflow_train.list'  #original training set: 35,454
            test_list  = 'dataloaders/lists/sceneflow_test.list'   #original test set:4,370
            train_set  = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
            # test_set   = stereo.DatasetFromList(args, test_list,  [576,960], False)
            test_set   = stereo.DatasetFromList(args, test_list,  [448,896], False)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
            return train_loader, test_loader
     
        ############################ scared 2019 ###########################
        elif args.dataset == 'scared':
            train_list= 'dataloaders/lists/scared2019_train.list'
            test_list = 'dataloaders/lists/scared2019_test.list'
            train_set = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
            test_set  = stereo.DatasetFromList(args, test_list,  [1024,1280], False)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader  = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
            return train_loader, test_loader
        else:
            raise NotImplementedError
