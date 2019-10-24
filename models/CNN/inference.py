# inference module for char level cnn
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader
from torch.autograd import Variable
from char_cnn_2 import *
torch.cuda.set_device(0)

def inference(batch_size, datapath):

    test_params = {"batch_size": batch_size,
                   "shuffle": False,
                   "num_workers": 0}

    test_set = MyDataset(datapath + "cnn_test_10m.csv")

    # generators for our training and test sets
    test_generator = DataLoader(test_set, **test_params)

    model = torch.load(datapath + "trained_model")
    model.eval()
    with torch.no_grad():

        test_true = []
        test_prob = []

        for batch in test_generator:
            _, n_true_label = batch

            batch = [Variable(record).cuda() for record in batch]

            t_data, _ = batch
            t_predicted_label = model(t_data)

            test_prob.append(t_predicted_label)
            test_true.extend(n_true_label)

        test_prob = torch.cat(test_prob, 0)
        test_prob = test_prob.cpu().data.numpy()
        test_true = np.array(test_true)
        #test_pred = np.argmax(test_prob, -1)

    # fieldnames = ['True label', 'Predicted label', 'Text']
    # with open(datapath + "output_reddit.csv", 'w',encoding='utf-8') as csv_file:
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    #     writer.writeheader()
    #     for i, j, k in zip(test_true, test_pred, test_set.texts):
    #         writer.writerow(
    #             {'True label': i, 'Predicted label': j, 'Text': k})

    test_metrics = get_evaluation(test_true, test_prob,
                                  list_metrics=["accuracy", "loss", "confusion_matrix"])
    print("Prediction:\nLoss: {} Accuracy: {} \nConfusion matrix: \n{}".format(test_metrics["loss"],
                                                                               test_metrics["accuracy"],
                                                                               test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    inference(1024, "/home/rachneet/reddit/char_cnn/")

