from keras.optimizers import Adam
from .loss import dummy_loss
from . import nets
from scipy.misc import imsave

from .img_util import preprocess_reflect_image, crop_image




class TransformNet():
    def __init__(self):
        self.style = None  # set_paramで指定
        self.output_file = None  # set_paramで指定
        self.input_file = None   # set_paramで指定

    def predict(self, style, output_file, input_file):
        self.style = style
        self.output_file = output_file
        self.input_file = input_file

        aspect_ratio, x = preprocess_reflect_image(input_file, size_multiple=4)

        img_width = img_height = x.shape[1]
        net = nets.image_transform_net(img_width, img_height)
        model = nets.loss_net(net.output, net.input, img_width, img_height, "", 0, 0)

        model.compile(Adam(), dummy_loss)

        model.load_weights("transfer/ns_model/pretrained_model/"+ self.style + "_weights.h5", by_name=False)

        y = net.predict(x)[0]
        y = crop_image(y, aspect_ratio)

        imsave('media/result/result.jpg', y)