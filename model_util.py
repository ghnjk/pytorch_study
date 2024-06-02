import os
from experiments.decision_tree.metric_smooth_classify import MetricSmoothClassifyDecisionTreeModel



data_root_path = "/Users/jkguo/workspace/fit-ops-check/ops_metric_training_data"


def build_smooth_dct() -> MetricSmoothClassifyDecisionTreeModel:
    """
    创建平滑度决策树
    :return:
    """
    smooth_dct = MetricSmoothClassifyDecisionTreeModel()
    smooth_dct.load_model(os.path.join(data_root_path, "metric_classify_db/models/tree.20220717.pkl"))
    return smooth_dct
