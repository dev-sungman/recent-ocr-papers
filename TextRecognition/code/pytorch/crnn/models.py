import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T*b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    
    def __init__(self, nclass):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
                # in_channels, out_channels, kernel_size, padding
                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=1, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(kernel_size=1, stride=2),
                nn.Conv2d(512, 512, kernel_size=2, stride=1),
                )

        self.rnn = nn.Sequential(
                BidirectionalLSTM(512, 256, 256),
                BidirectionalLSTM(256, 256, nclass),
                )

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2) #[b,c,w]
        conv = conv.permute(2,0,1) #[w,b,c]

        output = self.rnn(conv)

        return output
   
# for test
def test():
    net = CRNN(27)
    x = torch.randn(1,1,32,32)
    y = net(x)
    print(y.size())

#test()
