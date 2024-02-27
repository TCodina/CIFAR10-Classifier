if __name__ == '__main__':

    import os
    from torch import nn, optim

    from utils.util import load_data, split_train_val
    from utils.visualization import plot_metrics
    from models import DenseNet121Wrapper
    from dataset import CIFARDataset, CustomTransform
    from training_app import CIFARTrainingApp

    # Defines current working directory
    cwd = '.'
    # Load data all at once (TODO: should be changed in the future)
    data_dir = os.path.join(cwd, 'CIFAR10_data/')
    data_train_raw, data_test, classes = load_data(data_dir)

    # Split training set into training and validation sets
    val_size = 5000
    data_train, data_val = split_train_val(data_train_raw, val_size)

    # Initialize pre-processing transformations
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    trans_224imagenet = CustomTransform(imagenet_mean, imagenet_std, resize=224)
    trans_224imagenet_aug = CustomTransform(imagenet_mean, imagenet_std, resize=224, use_augmentation=True)

    # Define parameters. hyperparameters and design choices
    app_parameters = {'num_workers': 0,
                'val_freq': 1,
                'save_path': cwd + '/saved_models/densenet121_lr001',
                'seed': 42}

    hyperparameters = {'batch_size': 32,
                    'epochs': 20,
                    'model_parameters': {'drop_rate': 0.2},
                    'optimizer_parameters': {'lr': 0.01, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 1e-4},
                    'scheduler_parameters': {'step_size':5, 'gamma':0.5}}

    design = {'model': DenseNet121Wrapper,
            'criterion': nn.CrossEntropyLoss,
            'optimizer': optim.SGD,
            'scheduler': optim.lr_scheduler.StepLR}

    transformation = {'train': trans_224imagenet_aug, 'val': trans_224imagenet}

    # Initialize trainer app
    trainer = CIFARTrainingApp(data_train,
                                    data_val,
                                    app_parameters,
                                    hyperparameters,
                                    transformation,
                                    design)

    # Train model
    trainer.main(validate_bool=True, save_bool=True)
    # Plot metrics
    plot_metrics(*trainer.get_metrics(), w=50)


