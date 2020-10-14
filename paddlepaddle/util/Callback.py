import paddle
from visualdl import LogWriter
class Visual(paddle.callbacks.Callback):
    def __init__(self,log_dir="source/paddlepaddle"):
        self.log_dir = log_dir
        self.writer = LogWriter(logdir=self.log_dir)
    def writeLogs(self,logs,type):
        """
        logs:{'loss': [0.3810439], 'acc': 0.8914392605633803, 'step': 1419, 'batch_size': 32}
        params:{'batch_size': None, 'epochs': 3, 'steps': 1875, 'verbose': 2, 'metrics': ['loss', 'acc']}
        
        """
        if type == 'train':
            step = logs.get("step")+self.params.get("steps")*self.epoch
        else:
            step = self.epoch
        for key in self.params.get("metrics"):
            if not isinstance(logs.get(key),float):
                for i,value in enumerate(logs.get(key)):
                    self.writer.add_scalar(tag="{}_{}_{}".format(type,key,i), step=step, value=value)
            else:
                self.writer.add_scalar(tag="{}_{}".format(type,key), step=step, value=logs.get(key))
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
    def on_train_batch_end(self, step, logs=None):
        """Called at the end of each batch in training.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of current batch.
        """
        self.writeLogs(logs,'train')
    def on_eval_batch_end(self, step, logs=None):
        """Called at the end of each batch in evaluation.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of current batch.
        """
    def on_eval_end(self, logs=None):
        self.writeLogs(logs,'eval')
        pass

    def on_test_batch_end(self, step, logs=None):
        """Called at the end of each batch in predict.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None.
        """
        self.writeLogs(logs,'predict')

