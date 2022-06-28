import argparse

def obtain_predict_args():

    parser = argparse.ArgumentParser(description='HybridStereoNet Prediction')
    parser.add_argument('--crop_height', type=int, required=True, help="crop height")
    parser.add_argument('--crop_width', type=int, required=True, help="crop width")
    parser.add_argument('--maxdisp', type=int, default=192, help="max disp")
    parser.add_argument('--resume', type=str, default='', help="resume from saved model")
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--sceneflow', type=int, default=0, help='sceneflow dataset? Default=False')
    parser.add_argument('--davinci', type=int, default=0, help='scared2019? Default=False')
    parser.add_argument('--scared2019', type=int, default=0, help='scared2019? Default=False')
    parser.add_argument('--scared2019_small', type=int, default=0, help='scared2019_small? Default=False')
    parser.add_argument('--data_path', type=str, required=True, help="data root")
    parser.add_argument('--test_list', type=str, required=True, help="training list")
    parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")

    #Transtereo
    parser.add_argument('--dataset', type=str, default='sceneflow', 
                        choices=['sceneflow', 'scared2019', 'scared2019_small','davinci'], help='dataset name')    
    parser.add_argument('--cfg', type=str, default="configs/transtereo_test.yaml", metavar="FILE", help='path to config file', )
    args = parser.parse_args()
    return args
