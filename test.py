import logging
import torch
from torch.utils.data import DataLoader
import albumentations
import albumentations.pytorch
from network import *
from sklearn.metrics import confusion_matrix,roc_auc_score
from load_data import *
from utils import *
import time
import datetime


def test(us_path,csv_path,batch_size, device, model_path,log_file):
    dataset = UltrasoundDataset(us_path,csv_path,train=False, transform=False)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ConvNet().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    with torch.no_grad():
        acc = 0

        y_preds = np.array([])
        labels = np.array([])
        y_probs = np.array([])
        total_start = time.time()
        for batch_index, (id,image, label) in enumerate(test_loader):
            case_start = time.time()
            image, label = image.to(device, dtype=torch.float32), label.to(device)

            output = model(image)
            prob = torch.softmax(output, 1)[:, 1].detach().cpu().numpy()
            _, preds = torch.max(output.data, 1)
            y_preds = np.hstack([y_preds, preds.cpu().numpy()])
            labels = np.hstack([labels, label.cpu().numpy()])
            y_probs = np.hstack([y_probs, prob])
            acc += torch.sum(preds == label.data)

            if label == 1:
                probability = prob[0]
            else:
                probability = 1 - prob[0]
            predicted_value = preds.cpu().numpy()[0]
            case_end = time.time()
            case_time = case_end - case_start
            now = datetime.datetime.now()
            now = now.strftime('%Y-%m-%d %H:%M:%S')
            case_log = 'Anonymized id : ' + str(id.cpu().numpy()[0]) + ', label : '+str(label.cpu().numpy()[0]) + ', probability : '+str(probability) + ', predicted value : '+str(predicted_value) + ', time : '+str(case_time)+', date : '+now+'\n'
            log_file.write(case_log)

        auc_score = roc_auc_score(labels, y_probs)
        tn, fp, fn, tp = confusion_matrix(labels, y_preds).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        total_end = time.time()
        total_time = total_end-total_start
        now = datetime.datetime.now()
        now = now.strftime('%Y-%m-%d %H:%M:%S')
        total_log =  'AUC : ' + str(auc_score) + ', Accuracy : '+str(acc.item() / len(test_loader.dataset)) + ', Sensitivity : '+str(sensitivity) + ', Specificity : '+str(specificity) + ', time : '+str(total_time)+', date : '+now
        log_file.write(total_log)

        print('AUC : {:.6f}, Accuracy : {:.6f}, Sensitivity : {:.6f}, Specificity : {:.6f}'.format(
            auc_score, acc.item() / len(test_loader.dataset),
            sensitivity,
            specificity))
    return auc_score, acc.item() / len(test_loader.dataset), sensitivity, specificity


if __name__ =='__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args,_ = get_args()
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    logging.info(f'Using device {device}')

    log_file = open('./log_file.txt','w')
    path = './us_checkpoint.pth'

    auc_score, acc, sen, spe = test(us_path=args.us,csv_path=args.csv,batch_size=args.batchsize, device=device, model_path=path,log_file=log_file)

    log_file.close()