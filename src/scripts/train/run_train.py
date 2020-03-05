import argparse
import configparser

from train_setup import train

def preprocess_config(c):
    conf_dict = {}
    int_params = ["data.batch_size", "data.episodes", "data.gpu", "data.cuda", "model.growth_rate", "train.epochs", "train.patience"]
    float_params = ["train.lr", "data.train_size", "data.test_size", "data.rotation_range",
                    "data.width_shift_range", "data.height_shift_range", "model.reduction"]

    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]

    conf_dict['model.nb_layers'] = list(map(int, c['model.nb_layers'].split(',')))
    
    return conf_dict


parser = argparse.ArgumentParser(description='Run training')
parser.add_argument("--config", type=str, default="./src/config/config_cifar10.conf",
                    help="Path to the config file.")

parser.add_argument("--data.dataset", type=str, default=None)
parser.add_argument("--data.split", type=str, default=None)
parser.add_argument("--data.batch_size", type=int, default=None)
parser.add_argument("--data.episodes", type=int, default=None)
parser.add_argument("--data.cuda", type=int, default=None)
parser.add_argument("--data.gpu", type=int, default=None)

parser.add_argument("--data.rotation_range", type=float, default=None)
parser.add_argument("--data.width_shift_range", type=float, default=None)
parser.add_argument("--data.height_shift_range", type=float, default=None)
parser.add_argument("--data.horizontal_flip", type=bool, default=None)

parser.add_argument("--data.train_size", type=float, default=None)
parser.add_argument("--data.test_size", type=float, default=None)

parser.add_argument("--data.weight_classes", type=bool, default=False)

parser.add_argument("--model.name", type=str, default=None)
parser.add_argument("--model.weights", type=str, default=None)
parser.add_argument("--model.nb_layers", type=str, default=None)
parser.add_argument("--model.growth_rate", type=int, default=None)
parser.add_argument("--model.reduction", type=int, default=None)

parser.add_argument("--train.patience", type=int, default=None)
parser.add_argument("--train.lr", type=float, default=None)

# Run training
args = vars(parser.parse_args())
config = configparser.ConfigParser()
config.read(args['config'])
filtered_args = dict((k, v) for (k, v) in args.items() if not v is None)
config = preprocess_config({ **config['TRAIN'], **filtered_args })
train(config)
