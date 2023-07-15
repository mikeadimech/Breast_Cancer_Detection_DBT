from Breast_Cancer_Detection_DBT.src.utils import *

def main():

    # Define the accepted values for the arguments
    VALID_MODELS = ['Swin', 'ViT', 'ResNet', 'Convnext']
    VALID_TRANSFER_LEARNING = ['single', 'multi']

    # Create an ArgumentParser object and define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=VALID_MODELS, default='ResNet')
    parser.add_argument('--transferlearning', choices=VALID_TRANSFER_LEARNING, default='single')

    # Parse the arguments
    args = parser.parse_args()

    # Save the arguments as variables
    model = args.model
    transfer_learning = args.transferlearning


if __name__ == "__main__":
    main()