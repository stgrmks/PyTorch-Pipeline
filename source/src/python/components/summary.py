_author__ = 'MSteger'

import torch
import numpy as np

class summary(object):

    def __init__(self, model, device = torch.device('cpu'), input_size=(1,1,256,256), verbose = True):
        self.model = model.to(device)
        self.input_size = input_size
        self.device = device
        self.iterate()
        if verbose: self.printer()

    def compute_output(self, input, layer):
        if isinstance(layer, torch.nn.Linear):
            try:
                input = input.resize(1, layer.in_features)
            except Exception as e:
                print 'Failure! {} >> using hack'.format(e)
                input = input.mean(-1).mean(-1) # ...
        return layer(input)

    def compute_no_params(self, layer):
        params_to_optim, params_frozen = [], []
        for p in layer.parameters():
            if p.requires_grad:
                params_to_optim.append(tuple(p.size()))
            else:
                params_frozen.append(tuple(p.size()))
        return [np.sum([np.prod(p) for p in params_to_optim]).astype(int), np.sum([np.prod(p) for p in params_frozen]).astype(int)]

    def iterate(self):
        summary = []
        with torch.no_grad():
            input = torch.autograd.Variable(torch.FloatTensor(*self.input_size)).to(self.device)
            for k, v in self.model._modules.iteritems():
                if isinstance(v, torch.nn.Sequential):
                    for layer in v:
                        output = self.compute_output(input, layer)
                        summary.append([k, type(layer).__name__, tuple(input.shape)[1:], tuple(output.shape)[1:]] + self.compute_no_params(layer))
                        input = output
                else:
                    layer = v
                    output = self.compute_output(input, layer)
                    summary.append([k, type(layer).__name__, tuple(input.shape)[1:], tuple(output.shape)[1:]] + self.compute_no_params(layer))
                    input = output
        self.summary = summary
        return summary

    def printer(self, summary = None):
        if summary is None: summary = self.summary
        total_params, trainable_params = 0, 0
        print 'Model Summary'
        print '---------------------------------------------------------------------------------------------------------------------------------'
        print '{:>2} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20}'.format('Id', 'Name', 'Type', 'Input', 'Output', 'Params', 'Params(Frozen)')
        print '---------------------------------------------------------------------------------------------------------------------------------'
        for idx, layer in enumerate(summary):
            print '{:>2} {:>20}  {:>20} {:>20} {:>20} {:>20} {:>20}'.format(*[idx]+layer)
            total_params += layer[-2] + layer[-1]
            trainable_params += layer[-2]
        print '================================================================================================================================='
        print 'Total params: {0:,}'.format(total_params)
        print 'Trainable params: {0:,}'.format(trainable_params)
        print 'Non-trainable params: {0:,}'.format(total_params - trainable_params)
        print '---------------------------------------------------------------------------------------------------------------------------------'

        return self

