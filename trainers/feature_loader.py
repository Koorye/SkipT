import random
from collections import defaultdict


class FeatureLoader(object):
    def __init__(self, batch_size=4, mode='random'):
        self.batch_size = batch_size
        self.mode = mode
        self.features_all_epochs = []
        self.zero_shot_features_all_epochs = []
        self.labels_all_epochs = []
        self.batch_idxs_all_epochs = []
    
    def __len__(self):
        if self.one_epoch:    
            return len(self.batch_idxs_all_epochs)
        else:
            return len(self.batch_idxs_all_epochs[0])
        
    def add_features(self, features, labels, zero_shot_features=None):
        self.features_all_epochs.append(features)
        self.labels_all_epochs.append(labels)
        if zero_shot_features is not None:
            self.zero_shot_features_all_epochs.append(zero_shot_features)

        if self.mode == 'random':
            self.batch_idxs_all_epochs.append(self.generate_random_idxs(labels))
        elif self.mode == 'batch_random':
            self.batch_idxs_all_epochs.append(self.generate_random_batch_idxs(labels))
        else:
            raise NotImplementedError
        
        self.one_epoch = False
    
    def set_features(self, features, labels, zero_shot_features=None):
        self.features_all_epochs = features
        self.labels_all_epochs = labels
        if zero_shot_features is not None:
            self.zero_shot_features_all_epochs = zero_shot_features
        if self.mode == 'random':
            self.batch_idxs_all_epochs = self.generate_random_idxs(labels)
        elif self.mode == 'batch_random':
            self.batch_idxs_all_epochs = self.generate_random_batch_idxs(labels)
        else:
            raise NotImplementedError

        self.one_epoch = True
    
    def get_features(self, epoch=None):
        if epoch is not None:
            features = self.features_all_epochs[epoch]
            labels = self.labels_all_epochs[epoch]
            batch_idxs = self.batch_idxs_all_epochs[epoch]
            if len(self.zero_shot_features_all_epochs) > 0:
                zero_shot_features = self.zero_shot_features_all_epochs[epoch]
        else:
            features = self.features_all_epochs
            labels = self.labels_all_epochs
            batch_idxs = self.batch_idxs_all_epochs
            if len(self.zero_shot_features_all_epochs) > 0:
                zero_shot_features = self.zero_shot_features_all_epochs

        features_batch = [features[:, idx, :] for idx in batch_idxs]
        labels_batch = [labels[idx] for idx in batch_idxs]
        if len(self.zero_shot_features_all_epochs) > 0:
            zero_shot_features_batch = [zero_shot_features[idx, :] for idx in batch_idxs]
            batches = list(zip(features_batch, labels_batch, zero_shot_features_batch))
        else:
            batches = list(zip(features_batch, labels_batch))
        return batches

    def generate_random_idxs(self, labels):
        idxs = list(range(len(labels)))
        # a list of batches, each batch contains the indexes of samples
        # [[idx1, idx2, ...], [idx3, idx4, ...], ...]
        batch_idxs = []
        while True:
            selected_idxs = random.sample(idxs, self.batch_size)
            batch_idxs.append(selected_idxs)
            
            for idx in selected_idxs:
                idxs.remove(idx)
                
            if len(idxs) < self.batch_size:
                break
        
        return batch_idxs
    
    def generate_random_batch_idxs(self, labels):
        # build class_dict
        class_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            class_dict[label.item()].append(idx)
        
        # a list of batches, each batch contains the indexes of samples of the same class
        # [[class1_idx1, class1_idx2, ...], [class2_idx1, class2_idx2, ...], ...]
        batch_idxs = []
        while True:
            selected_class = random.choice(list(class_dict.keys()))
            idxs = class_dict[selected_class]
            selected_idxs = random.sample(idxs, self.batch_size)
            batch_idxs.append(selected_idxs)
            
            for idx in selected_idxs:
                class_dict[selected_class].remove(idx)
                
            if len(class_dict[selected_class]) < self.batch_size:
                del class_dict[selected_class]
            
            if len(class_dict) == 0:
                break
        
        return batch_idxs


class FeatureLoader2D(FeatureLoader):
    def get_features(self, epoch=None):
        if epoch is not None:
            features = self.features_all_epochs[epoch]
            labels = self.labels_all_epochs[epoch]
            batch_idxs = self.batch_idxs_all_epochs[epoch]
            if len(self.zero_shot_features_all_epochs) > 0:
                zero_shot_features = self.zero_shot_features_all_epochs[epoch]
        else:
            features = self.features_all_epochs
            labels = self.labels_all_epochs
            batch_idxs = self.batch_idxs_all_epochs
            if len(self.zero_shot_features_all_epochs) > 0:
                zero_shot_features = self.zero_shot_features_all_epochs

        features_batch = [features[idx, :] for idx in batch_idxs]
        labels_batch = [labels[idx] for idx in batch_idxs]
        if len(self.zero_shot_features_all_epochs) > 0:
            zero_shot_features_batch = [zero_shot_features[idx, :] for idx in batch_idxs]
            batches = list(zip(features_batch, labels_batch, zero_shot_features_batch))
        else:
            batches = list(zip(features_batch, labels_batch))
        return batches
