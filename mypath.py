class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'sceneflow':
            return './dataset/SceneFlow/'
        elif dataset == 'scared':
            return './dataset/SCARED2019/' 
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
