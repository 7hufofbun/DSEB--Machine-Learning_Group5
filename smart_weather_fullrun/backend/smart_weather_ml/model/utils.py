
import os, random, warnings
import numpy as np
warnings.filterwarnings("ignore")
class DummyLogger:
    def report_table(self, *a, **k): pass
    def report_scalar(self, *a, **k): pass
def init_clearml(task_name="LGBM All Features Pipeline", project_name="Weather Forecast HCM"):
    try:
        from clearml import Task
        task = Task.init(project_name=project_name, task_name=task_name, task_type=Task.TaskTypes.training, output_uri=True)
        return task, task.get_logger()
    except Exception as e:
        print(f"ClearML not available: {e}")
        return None, DummyLogger()
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
