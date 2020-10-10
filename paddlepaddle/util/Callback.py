import paddle
from visualdl import LogWriter
class Visual(paddle.callbacks.Callback):
    def __init__(self,log_dir):
        self.log_dir = log_dir
        self.writer = LogWriter(logdir=self.log_dir)
    def printByStep(self,logs):
        step = logs.get("step")
        loss = str(logs.get("loss")[0]) if logs.get("loss")!=None else 0
        self.writer.add_scalar(tag="loss", step=step, value=loss)
        for key,value in {k:v for (k,v) in logs.items() if k not in["step","loss","batch_size"]}.items():
            self.writer.add_scalar(tag=key, step=step, value=str(value))
    def on_train_batch_begin(self, step, logs=None):
        """Called at the beginning of each batch in training.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is empty.
        """
        self.printByStep(logs)
    def on_train_batch_end(self, step, logs=None):
        """Called at the end of each batch in training.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of current batch.
        """

    def on_eval_batch_begin(self, step, logs=None):
        """Called at the beginning of each batch in evaluation.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is empty.
        """
        self.printByStep(logs)
    def on_eval_batch_end(self, step, logs=None):
        """Called at the end of each batch in evaluation.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of current batch.
        """

    def on_test_batch_begin(self, step, logs=None):
        """Called at the beginning of each batch in predict.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None.
        """
        self.printByStep(logs)
    def on_test_batch_end(self, step, logs=None):
        """Called at the end of each batch in predict.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None.
        """

