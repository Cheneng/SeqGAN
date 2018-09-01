import torch


class Visdom_line(object):
    def __init__(self, vis, win, start_step=0, name="Line_1"):
        """
        :param vis: (object) visdom.Visdom object
        :param win: (str) name of the window
        :param start_step: (int) the begin of the step
        :param name: (str) the name of the line
        """
        self._vis = vis
        self._win = win
        self._start_step = start_step
        self._name = name

    def update(self, y):
        if self._start_step == 0:
            self._vis.line(X=torch.Tensor([self._start_step]),
                           Y=y if isinstance(y, torch.Tensor) else torch.Tensor([y]),
                           win=self._win,
                           name="%s" % self._name,
                           opts=dict(legend=[self._name]))
        else:
            self._vis.updateTrace(X=torch.Tensor([self._start_step]),
                                  Y=y if isinstance(y, torch.Tensor) else torch.Tensor([y]),
                                  win=self._win,
                                  name="%s" % self._name)
        self._start_step += 1