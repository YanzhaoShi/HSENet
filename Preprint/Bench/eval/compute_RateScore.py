from RaTEScore import RaTEScore
import pandas as pd
import os

def RaTEScore_demo():
    """
    RaTEScore demo
    """
    pred_report = ['There are no intracranial hemorrhages.',
                'The musculature and soft tissues are intact.']

    gt_report = ['There is no finding to suggest intracranial hemorrhage.',
                'The muscle compartments are intact.']

    assert len(pred_report) == len(gt_report)

    ratescore = RaTEScore()
    # Add visualization_path here if you want to save the visualization result
    # ratescore = RaTEScore(visualization_path = '')

    scores = ratescore.compute_score(pred_report, gt_report)
    print('RaTEScore:', scores)

def RaTEScore_compute_from_csv(csv_path):
    """
    RaTEScore compute
    """
    load_csv_res = pd.read_csv(csv_path)
    load_csv_res = load_csv_res[['Ground Truth', 'pred']]
    gt_report = load_csv_res['Ground Truth'].tolist()
    pred_report = load_csv_res['pred'].tolist()

    assert len(pred_report) == len(gt_report)
    
    ratescore = RaTEScore()
    # Add visualization_path here if you want to save the visualization result
    # ratescore = RaTEScore(visualization_path = '')
    scores = ratescore.compute_score(pred_report, gt_report)

    return scores


if __name__ == '__main__':

    compute_dict = {
        "model_name":"eval_caption.csv",
    }
    for key, csv_path in compute_dict.items():
        print(f"############ Computing RaTEScore for: {key} ############ ")
        print(f"csv_path: {csv_path}")
        rate_scores = RaTEScore_compute_from_csv(csv_path)
        avg_rate_score = sum(rate_scores) / len(rate_scores)
        print('RaTEScore:', rate_scores)
        print('Avg RaTEScore:', avg_rate_score)
        print(f"############ Finished Computing RaTEScore for: {key} ############ ")
        print(" ")


