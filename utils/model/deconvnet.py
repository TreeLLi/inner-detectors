'''
DeConvNet

To map activations back to input spaces

'''


class DeConvNet:
    
    def __init__(self, model):
        self.layers = list(reversed(model.layers))
        self.configs = model.configs
        self.max_pool_switches = model.max_pool_switches
        
    def build(self):
        print ("Building DeConvNet")

    def unpoolLayer(self):
        print ("Construct unpool layer")

    def transposeConvLayer(self):
        print ("Construct transpose conv layer")
