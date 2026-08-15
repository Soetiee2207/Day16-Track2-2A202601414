import time
import json
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import os

def main():
    results = {}
    
    # 1 & 3. Load dataset
    print("Loading data...")
    start_load = time.time()
    df = pd.read_csv(os.path.expanduser('~/ml-benchmark/creditcard.csv'))
    results['Thời gian load data'] = f"{time.time() - start_load:.4f} seconds"
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2 & 3. Train LGBM
    print("Training model...")
    start_train = time.time()
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(stopping_rounds=50)])
    results['Thời gian training'] = f"{time.time() - start_train:.4f} seconds"
    results['Best iteration'] = str(model.best_iteration_)
    
    # 4. Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results['AUC-ROC'] = f"{roc_auc_score(y_test, y_prob):.4f}"
    results['Accuracy'] = f"{accuracy_score(y_test, y_pred):.4f}"
    results['F1-Score'] = f"{f1_score(y_test, y_pred):.4f}"
    results['Precision'] = f"{precision_score(y_test, y_pred):.4f}"
    results['Recall'] = f"{recall_score(y_test, y_pred):.4f}"
    
    # 5. Inference Latency & Throughput
    print("Benchmarking inference...")
    # 1 row
    single_row = X_test.iloc[[0]]
    start_latency = time.time()
    model.predict(single_row)
    results['Inference latency (1 row)'] = f"{(time.time() - start_latency) * 1000:.4f} ms"
    
    # 1000 rows
    thousand_rows = X_test.head(1000)
    start_throughput = time.time()
    model.predict(thousand_rows)
    results['Inference throughput (1000 rows)'] = f"{(time.time() - start_throughput) * 1000:.4f} ms"
    
    # 6. Save results
    print("Saving results...")
    with open(os.path.expanduser('~/benchmark_result.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    main()
