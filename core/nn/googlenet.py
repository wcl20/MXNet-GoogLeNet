import mxnet as mx

class GoogLeNet:

    @staticmethod
    def conv_module(data, filters, kernel_size, stride=(1, 1), pad=(0, 0)):
        # Apply convolution kernel_size (Kx, Ky)
        conv = mx.sym.Convolution(data=data, kernel=kernel_size, num_filter=filters, stride=stride, pad=pad)
        # Batch Normalization
        bn = mx.sym.BatchNorm(data=conv)
        # Activation
        act = mx.sym.Activation(data=bn, act_type="relu")
        return act


    @staticmethod
    def inception_module(data, filters1x1, filters3x3_reduce, filters3x3, filters5x5_reduce, filters5x5, filters1x1_proj):
        # First branch 1x1 convolution
        conv_1_1 = GoogLeNet.conv_module(data, filters1x1, (1, 1))

        # Second branch 1x1 convolution => 3x3 convolution
        conv_2_1 = GoogLeNet.conv_module(data, filters3x3_reduce, (1, 1))
        conv_2_2 = GoogLeNet.conv_module(conv_2_1, filters3x3, (3, 3), pad=(1, 1))

        # Third branch 1x1 convolution => 5x5 convolution
        conv_3_1 = GoogLeNet.conv_module(data, filters5x5_reduce, (1, 1))
        conv_3_2 = GoogLeNet.conv_module(conv_3_1, filters5x5, (5, 5), pad=(2, 2))

        # Fourth branch 3x3 max pooling => 1x1 convolution
        pool_4_1 = mx.sym.Pooling(data=data, pool_type="max", kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        conv_4_1 = GoogLeNet.conv_module(pool_4_1, filters1x1_proj, (1, 1))

        # Concatenate results
        concat = mx.sym.Concat(*[conv_1_1, conv_2_2, conv_3_2, conv_4_1])
        return concat

    @staticmethod
    def build(num_classes):

        data = mx.sym.Variable("data")

        # Block 1
        conv_1_1 = GoogLeNet.conv_module(data, 64, (7, 7), pad=(3, 3), stride=(2, 2))
        pool_1_1 = mx.sym.Pooling(data=conv_1_1, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2))
        conv_1_2 = GoogLeNet.conv_module(pool_1_1, 64, (1, 1))

        # Block 2
        conv_2_1 = GoogLeNet.conv_module(conv_1_2, 192, (3, 3), pad=(1, 1))
        pool_2_1 = mx.sym.Pooling(data=conv_2_1, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2))

        # Block 3
        inception_3_1 = GoogLeNet.inception_module(pool_2_1, 64, 96, 128, 16, 32, 32)
        inception_3_2 = GoogLeNet.inception_module(inception_3_1, 128, 128, 192, 32, 96, 64)
        pool_3_1 = mx.sym.Pooling(data=inception_3_2, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2))

        # Block 4
        inception_4_1 = GoogLeNet.inception_module(pool_3_1, 192, 96, 208, 16, 48, 64)
        inception_4_2 = GoogLeNet.inception_module(inception_4_1, 160, 112, 224, 24, 64, 64)
        inception_4_3 = GoogLeNet.inception_module(inception_4_2, 128, 128, 256, 24, 64, 64)
        inception_4_4 = GoogLeNet.inception_module(inception_4_3, 112, 144, 288, 32, 64, 64)
        inception_4_5 = GoogLeNet.inception_module(inception_4_4, 256, 160, 320, 32, 128, 128)
        pool_4_1 = mx.sym.Pooling(data=inception_4_5, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2))

        # Block 5
        inception_5_1 = GoogLeNet.inception_module(pool_4_1, 256, 160, 320, 32, 128, 128)
        inception_5_2 = GoogLeNet.inception_module(inception_5_1, 384, 192, 384, 48, 128, 128)
        pool_5_1 = mx.sym.Pooling(data=inception_5_2, pool_type="avg", kernel=(7, 7), stride=(1, 1))
        dropout_5_1 = mx.sym.Dropout(data=pool_5_1, p=0.4)

        flatten = mx.sym.Flatten(data=dropout_5_1)
        fc_6_1 = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes)
        model = mx.sym.SoftmaxOutput(data=fc_6_1, name="softmax")

        return model



        return model
