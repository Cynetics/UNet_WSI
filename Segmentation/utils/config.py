import argparse

def dense_unet_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--latent_hidden_n', type=int, default=512)
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--n_eval', type=int, default=250)
    parser.add_argument('--print_it', type=int, default=250)
    parser.add_argument('--val_num', type=int, default=128, help="how many datapoints to validate on")
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="model")
    parser.add_argument('--validate', default=0, type=int)
    args = parser.parse_args()
    print(args)
    with open("config.txt", 'w') as file:
        file.write(str(args))
    return args