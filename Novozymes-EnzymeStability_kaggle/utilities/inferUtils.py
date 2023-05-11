import csv

def kaggle_submission(preds, filename='new'):
    with open(filename+'_submission.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['seq_id','tm'])
        for row in preds:
            writer.writerow(row)
