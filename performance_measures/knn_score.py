import numpy as np

class Scores:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred
        self.accuracy = np.mean(y_pred == y_test)
        self.createConfusionMatrix(y_test, y_pred)
        self.calculateMicroScores()
        self.calculateMacroScores()

    def createConfusionMatrix(self, y_test, y_pred):
        test_unique = np.sort(np.unique(y_test))
        matrix = np.zeros((len(test_unique), len(test_unique)))
        for i in range(len(test_unique)):
            for j in range(len(test_unique)):
                matrix[i, j] = np.sum((y_pred == test_unique[i]) & (y_test == test_unique[j]))
            
        # for i in range(matrix.shape[0]):
        #     for j in range(matrix.shape[1]):
        #         print(f"{matrix[i, j]:.0f}", end=" ")
        #     print()
        self.confusion_matrix = matrix

    def calculateMicroScores(self):
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=1) - TP
        FN = np.sum(self.confusion_matrix, axis=0) - TP
        precision = np.sum(TP) / np.sum(TP + FP)
        recall = np.sum(TP) / np.sum(TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        self.micro_precision = precision
        self.micro_recall = recall
        self.micro_f1 = f1

    def calculateMacroScores(self):
        precision = np.zeros(self.confusion_matrix.shape[0])
        recall = np.zeros(self.confusion_matrix.shape[0])
        f1 = np.zeros(self.confusion_matrix.shape[0])
        for i in range(self.confusion_matrix.shape[0]):
            TP = self.confusion_matrix[i, i]
            FP = np.sum(self.confusion_matrix[i, :]) - TP
            FN = np.sum(self.confusion_matrix[:, i]) - TP
            if (TP + FP) != 0:
                precision[i] = TP / (TP + FP)
            else:
                precision[i] = 0
            if (TP + FN) != 0:
                recall[i] = TP / (TP + FN)
            else:
                recall[i] = 0
            if (precision[i] + recall[i]) != 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:
                f1[i] = 0
        self.macro_precision = np.mean(precision)
        self.macro_recall = np.mean(recall)
        self.macro_f1 = np.mean(f1)
