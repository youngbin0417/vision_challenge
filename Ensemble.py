import os
import joblib
import numpy as np
import faiss
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import normalize

class EnsembleFaissKNN:
    def __init__(self, fold_model_dir, n_folds=5):
        self.fold_model_dir = fold_model_dir
        self.n_folds = n_folds
        self.models = []

    def load_models(self):
        self.models = []
        for fold in range(self.n_folds):
            index_path = os.path.join(self.fold_model_dir, f'faiss_index_fold_{fold}.bin')
            param_path = os.path.join(self.fold_model_dir, f'faiss_params_fold_{fold}.joblib')

            faiss_index = faiss.read_index(index_path)
            params = joblib.load(param_path)

            self.models.append({
                'index': faiss_index,
                'y_train_labels': params['y_train_labels'],
                'n_neighbors': params['n_neighbors']
            })

        print(f"✔ {self.n_folds}개의 Fold 모델 로딩 완료.")

    def predict(self, X_test_feats):
        all_preds = []

        for model in self.models:
            index = model['index']
            y_train_labels = model['y_train_labels']
            k = model['n_neighbors']

            X_test_norm = normalize(X_test_feats, axis=1).astype(np.float32)
            _, indices = index.search(X_test_norm, k)

            y_pred = []
            for neighbor_indices in indices:
                neighbor_labels = y_train_labels[neighbor_indices]
                majority_label = np.bincount(neighbor_labels).argmax()
                y_pred.append(majority_label)
            all_preds.append(y_pred)

        # Hard Voting
        all_preds = np.array(all_preds)  # shape: (n_folds, n_samples)
        final_preds = []
        for i in range(all_preds.shape[1]):
            votes = all_preds[:, i]
            final_label = np.bincount(votes).argmax()
            final_preds.append(final_label)

        return np.array(final_preds)

    def evaluate(self, X_test_feats, y_test, le=None):
        y_pred = self.predict(X_test_feats)

        print("\n=== 앙상블 평가 결과 ===")
        print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        if le:
            print(classification_report(y_test, y_pred, target_names=le.classes_))
        else:
            print(classification_report(y_test, y_pred))

        return y_pred
